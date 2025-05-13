import os
import datetime
import time
import json
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from app.validate_tf import validate_terraform_code
from app.rag_prompts import get_system_prompt, get_search_prompt, format_system_prompt, format_feedback_prompt
from app.vector_store import CHROMA_COLLECTION_NAME, CHROMA_DB_NAME, VectorStore
from app.schemas import Search, State
from app.checkov_validator import generate_compliance_report
from app.error_summarizer import summarize_validation_errors
from dotenv import load_dotenv
import logging
from app.models import LLMClient
from langchain_google_genai import ChatGoogleGenerativeAI


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient,
        checkov_output_dir: str = None,
        max_retries: int = 3,
        run_type: str = "default",
        compliance_threshold: float = 90.0,
    ):
        # Vector store
        self.vector_store = vector_store

        # LLM setup
        self.llm_client = llm_client
        chat_model = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.structured_llm = chat_model.with_structured_output(Search)
        
        # Output directory for Checkov results
        if checkov_output_dir is None:
            # Use a directory in the project root
            self.checkov_output_dir = os.path.join(os.getcwd(), "checkov_output")
        else:
            self.checkov_output_dir = checkov_output_dir

        # Feedback loop settings
        self.max_retries = max_retries
        self.run_type = run_type
        self.compliance_threshold = compliance_threshold
        
        # Feedback logging directory
        self.feedback_logs_dir = os.path.join(os.getcwd(), "feedback_logs")
        os.makedirs(self.feedback_logs_dir, exist_ok=True)
        
        # Real failures directory
        self.real_failures_dir = os.path.join(os.getcwd(), "real_failures")
        os.makedirs(self.real_failures_dir, exist_ok=True)

        os.makedirs(self.checkov_output_dir, exist_ok=True)

        # Load CIS benchmark data into the vector store so CIS docs are retrievable
        # Try multiple potential locations for the benchmark file
        cis_benchmark_locations = [
            "scripts/cis_benchmark_enhanced.json",
            "../scripts/cis_benchmark_enhanced.json",
            os.path.join(os.getcwd(), "scripts/cis_benchmark_enhanced.json")
        ]
        
        cis_loaded = False
        for benchmark_file in cis_benchmark_locations:
            logger.info(f"Attempting to load CIS benchmark data from: {benchmark_file}")
            if os.path.exists(benchmark_file):
                cis_loaded = self.vector_store.load_cis_benchmark_data(benchmark_file)
                if cis_loaded:
                    logger.info(f"Successfully loaded CIS benchmark data from {benchmark_file}")
                    break
        
        if not cis_loaded:
            logger.warning("Failed to load CIS benchmark data. Vector search for CIS controls may not work properly.")
        
        # Build and compile the LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(State)

        # Define the sequence of steps
        graph_builder.add_node("analyze_query", self.analyze_query)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_node("validate", self.validate)
        graph_builder.add_node("process_feedback", self.process_feedback)
        graph_builder.add_node("regenerate", self.regenerate)

        # Define the edges
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_edge("analyze_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", "validate")
        
        # Conditional routing based on validation results
        graph_builder.add_conditional_edges(
            "validate",
            self.should_retry,
            {
                True: "process_feedback",
                False: END,
            },
        )
        
        # Feedback loop
        graph_builder.add_edge("process_feedback", "regenerate")
        graph_builder.add_edge("regenerate", "validate")

        return graph_builder.compile()

    def should_retry(self, state: State) -> bool:
        """Determine if we should retry code generation based on validation results."""
        # Get retry count - default to 0 if not set
        retry_count = state.get("retry_count", 0)
        
        # Check if we have validation errors and haven't exceeded retry limit
        has_validation_errors = not state.get("validation_results", {}).get("syntax_valid", True) or \
                               not state.get("validation_results", {}).get("cis_compliant", True)
        
        # Return true if both conditions are met: has errors and under retry limit
        return has_validation_errors and retry_count < self.max_retries

    def analyze_query(self, state: State) -> State:
        """Parse the question into a structured Search object."""
        logger.info("Analyzing query...")

        # Add retry logic for API rate limits
        max_retries = 3
        backoff_time = 2  # seconds

        for attempt in range(max_retries):
            try:
                parsed_query = self.structured_llm.invoke(
                    get_search_prompt() + f"\n\nUser query: {state['question']}"
                )
                if "query" not in parsed_query:
                    parsed_query["query"] = state["question"]

                logger.info(f"Parsed query: {parsed_query}")
                
                # Initialize the retry count and max_retries
                return {
                    "search": parsed_query,
                    "retry_count": 0,
                    "max_retries": self.max_retries
                }
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit hit, retrying in {backoff_time} seconds (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    logger.error(f"Error analyzing query: {str(e)}")
                    # Fallback to using the raw question as the query
                    return {
                        "search": {"query": state["question"]},
                        "retry_count": 0,
                        "max_retries": self.max_retries
                    }

    def retrieve(self, state: State) -> State:
        """Retrieve documents based on the query and section."""
        logger.info("Retrieving relevant documents...")

        # Log the search query for debugging
        logger.info(f"Search query: {state['search']}")

        # Create filter for document retrieval
        # Allow retrieval of both AWS resources and CIS controls
        # First, build a filter for AWS resources
        aws_resource_filter = [{"section": {"$in": ["Example Usage"]}}]
        if state["search"]["subcategories"]:
            aws_resource_filter.append(
                {"subcategory": {"$in": state["search"]["subcategories"]}}
            )
        
        # Combine filters to allow both AWS resources and CIS controls
        # This approach uses a metadata field OR condition
        chroma_filter = {
            "$or": [
                # AWS resources filter (AND condition on section and subcategories)
                {"$and": aws_resource_filter},
                # CIS controls filter - include any documents with type=cis_control
                {"type": "cis_control"}
            ]
        }
        
        # Check if the query contains security-related terms to boost CIS controls
        security_terms = ["security", "compliance", "cis", "benchmark", "secure", "protect", "encryption", "encrypt"]
        query = state["search"]["query"].lower()
        
        # Expand search results if security-related terms are present
        k_results = 5
        if any(term in query for term in security_terms):
            k_results = 12  # Get more results for security-related queries
            logger.info("Security-related query detected, expanding search results")
        
        # Retrieve documents from the vector store
        try:
            retrieved_docs = self.vector_store.get_db_instance().similarity_search(
                query=state["search"]["query"],
                k=k_results,
                filter=chroma_filter,
            )
            
            # Log the number of retrieved documents
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Extract which CIS control IDs we retrieved
            cis_control_ids = []
            
            for doc in retrieved_docs:
                if doc.metadata.get("type") == "cis_control" and doc.metadata.get("control_id"):
                    cis_control_ids.append(doc.metadata.get("control_id"))
                    
            # Deduplicate the control IDs
            cis_control_ids = list(set(cis_control_ids))
            
            # Sort them for consistent ordering
            cis_control_ids.sort()
            
            logger.info(f"Retrieved {len(cis_control_ids)} unique CIS controls: {', '.join(cis_control_ids)}")
            
            # For security-related queries without specific CIS controls retrieved,
            # add some common ones to guide the LLM
            if any(term in query for term in security_terms) and not cis_control_ids:
                logger.info("Adding default CIS controls for security-related query")
                default_security_controls = ["2.1.1", "2.1.2", "2.2"]
                if "ec2" in query or "instance" in query:
                    default_security_controls.extend(["5.1", "5.2", "5.3"])
                cis_control_ids = default_security_controls
                logger.info(f"Added default CIS controls: {', '.join(cis_control_ids)}")
            
            return {"context": retrieved_docs, "referenced_cis_controls": cis_control_ids}
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            # Return an empty context as fallback
            return {"context": [], "referenced_cis_controls": []}

    def generate(self, state: State) -> State:
        """Generate an answer using the retrieved context."""
        logger.info("Generating Terraform code...")

        # Extract context from retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Get referenced CIS controls if available
        referenced_cis_controls = state.get("referenced_cis_controls", [])
        if referenced_cis_controls:
            logger.info(f"Including specific CIS controls in prompt: {referenced_cis_controls}")
        
        # Create prompt with question, context, and CIS controls
        formatted_prompt = format_system_prompt(
            context=docs_content,
            question=state.get("question", ""),
            referenced_cis_controls=referenced_cis_controls
        )
        
        # Log the full prompt that will be sent to the LLM for inspection
        logger.info("LLM system prompt:\n%s", formatted_prompt)
        # Use the system prompt template format
        answer = self.llm_client.run(formatted_prompt, formatted_prompt)
        return {"answer": answer, "referenced_cis_controls": referenced_cis_controls}

    def log_feedback_iteration(self, state: State, iteration: int) -> None:
        """Log feedback and code at each iteration to track improvements."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = hash(state.get("question", "")) % 10000
        
        # Get model type and temperature for filename
        model_type = self.llm_client.model.replace('-', '_').lower() if hasattr(self.llm_client, 'model') else "unknown"
        temp = str(self.llm_client.temperature).replace('.', '_') if hasattr(self.llm_client, 'temperature') else "unknown"
        
        log_data = {
            "timestamp": timestamp,
            "run_type": self.run_type,
            "run_id": run_id,
            "iteration": iteration,
            "question": state.get("question", ""),
            "code": state.get("answer", ""),
            "validation_feedback": state.get("validation_feedback", ""),
            "validation_results": state.get("validation_results", {})
        }
        
        # Save to a JSON file
        log_filename = f"{self.run_type}_{model_type}_temp_{temp}_iteration_{run_id}_{iteration}_{timestamp}.json"
        log_path = os.path.join(self.feedback_logs_dir, log_filename)
        
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
            
        logger.info(f"Feedback iteration {iteration} logged to {log_path}")

    def log_failed_generation(self, state: State) -> None:
        """Store failed generations from the LLM for later testing"""
        validation_results = state.get("validation_results", {})
        
        # Check if the generation failed validation 
        if not validation_results.get("syntax_valid", True) or not validation_results.get("cis_compliant", True):
            # Only store initial failures (not retry attempts)
            if state.get("retry_count", 0) == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                question = state.get("question", "")
                query_hash = hash(question) % 10000
                
                # Prepare failure data
                failure_data = {
                    "name": f"Auto-collected failure: {question[:50]}...",
                    "query": question,
                    "code": state.get("answer", ""),
                    "validation_results": validation_results,
                    "timestamp": timestamp,
                    "validation_feedback": summarize_validation_errors(validation_results)
                }
                
                # Ensure directory exists (double-check here for safety)
                os.makedirs(self.real_failures_dir, exist_ok=True)
                
                # Save to file with real_failure prefix
                failure_file = f"real_failure_{self.run_type}_{query_hash}_{timestamp}.json"
                failure_path = os.path.join(self.real_failures_dir, failure_file)
                
                with open(failure_path, "w") as f:
                    json.dump(failure_data, f, indent=2)
                    
                logger.info(f"Logged real failure to {failure_path}")

    def validate(self, state: State) -> State:
        """Validate the generated Terraform code against CIS controls."""
        logger.info("Validating generated Terraform code...")

        # Generate a timestamp for this validation
        import datetime as dt

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hash(state["question"]) % 10000

        # Validate the generated Terraform code
        validation_results = validate_terraform_code(
            state["answer"],
            output_dir=self.checkov_output_dir,
            result_file_prefix=f"validation_{self.run_type}_{query_hash}",
        )

        # Extract results for better reporting
        is_syntax_valid = validation_results.get("syntax_valid", False)
        is_cis_compliant = validation_results.get("cis_compliant", False)
        referenced_cis_controls = validation_results.get("referenced_cis_controls", [])

        # Log validation status
        logger.info(f"Terraform syntax valid: {is_syntax_valid}")
        logger.info(f"CIS compliant: {is_cis_compliant}")
        
        # Log iteration 0 (initial attempt)
        retry_count = state.get("retry_count", 0)
        self.log_feedback_iteration(state, retry_count)
        
        # Store failed generations for later testing
        state_with_results = {**state, "validation_results": validation_results}
        self.log_failed_generation(state_with_results)
        
        # Store validation results in state
        return {"validation_results": validation_results}

    def process_feedback(self, state: State) -> State:
        """Process validation errors and generate feedback for correction."""
        logger.info("Processing validation errors and generating feedback...")
        
        # Extract validation results
        validation_results = state.get("validation_results", {})
        
        # Generate human-readable error summary
        error_summary = summarize_validation_errors(validation_results)
        logger.info(f"Validation feedback generated:\n{error_summary}")
        
        # Add validation feedback to state
        state_update = {"validation_feedback": error_summary}
        
        # Update state and log
        updated_state = {**state, **state_update}
        
        # Log this iteration's feedback
        retry_count = updated_state.get("retry_count", 0)
        self.log_feedback_iteration(updated_state, retry_count)
        
        return state_update

    def regenerate(self, state: State) -> State:
        """Regenerate code based on validation feedback."""
        retry_count = state.get("retry_count", 0) + 1
        logger.info(f"Regenerating code (attempt {retry_count})...")
        
        # Extract necessary data from state
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        referenced_cis_controls = state.get("referenced_cis_controls", [])
        previous_code = state.get("answer", "")
        validation_feedback = state.get("validation_feedback", "")
        
        # Create feedback prompt
        feedback_prompt = format_feedback_prompt(
            context=docs_content,
            question=state.get("question", ""),
            previous_code=previous_code,
            validation_feedback=validation_feedback,
            referenced_cis_controls=referenced_cis_controls
        )
        
        # Generate improved code with feedback
        new_answer = self.llm_client.run(feedback_prompt, feedback_prompt)
        
        # Update state with new answer and increment retry counter
        state_update = {
            "answer": new_answer,
            "previous_attempt_code": previous_code,
            "retry_count": retry_count
        }
        
        # Update state and log after regeneration
        updated_state = {**state, **state_update}
        self.log_feedback_iteration(updated_state, retry_count)
        
        return state_update

    def log_final_result(self, state: State) -> None:
        """Log the final result after all iterations complete."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = hash(state.get("question", "")) % 10000
        
        # Get model type and temperature for filename
        model_type = self.llm_client.model.replace('-', '_').lower() if hasattr(self.llm_client, 'model') else "unknown"
        temp = str(self.llm_client.temperature).replace('.', '_') if hasattr(self.llm_client, 'temperature') else "unknown"
        
        # Get the most up-to-date validation results
        code = state.get("answer", "")
        question = state.get("question", "")
        retry_count = state.get("retry_count", 0)
        
        # Re-validate the code to get the latest validation results
        latest_validation = validate_terraform_code(
            code, 
            output_dir=self.checkov_output_dir, 
            run_id=run_id, 
            compliance_threshold=self.compliance_threshold,
            result_file_prefix=f"validation_{self.run_type}_{model_type}_temp_{temp}_{run_id}"
        )
        
        # Use the latest validation results
        is_syntax_valid = latest_validation.get("syntax_valid", False)
        is_cis_compliant = latest_validation.get("cis_compliant", False)
        checkov_results = latest_validation.get("checkov_results", {})
        validation_summary = checkov_results.get("validation_summary", {})
        
        # Create a summary object
        summary = {
            "timestamp": timestamp,
            "run_type": self.run_type,
            "run_id": run_id,
            "question": question,
            "total_iterations": retry_count + 1,  # +1 for initial attempt
            "final_code": code,
            "final_state": {
                "syntax_valid": is_syntax_valid,
                "cis_compliant": is_cis_compliant,
                "pass_rate": validation_summary.get("pass_rate", 0),
                "total_checks": validation_summary.get("total_checks", 0),
                "passed": validation_summary.get("passed", 0),
                "failed": validation_summary.get("failed", 0)
            }
        }
        
        # Save as final summary file
        summary_filename = f"{self.run_type}_{model_type}_temp_{temp}_final_summary_{run_id}_{timestamp}.json"
        summary_path = os.path.join(self.feedback_logs_dir, summary_filename)
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Final result logged to {summary_path}")
        logger.info(f"Final validation: Syntax valid: {is_syntax_valid}, CIS compliant: {is_cis_compliant}")
        
        if is_cis_compliant:
            pass_rate = validation_summary.get("pass_rate", 0)
            logger.info(f"Achieved compliance with a pass rate of {pass_rate:.1f}%")

    def run(self, question: str) -> State:
        """Run the RAG pipeline for a given question."""
        # Run the pipeline on a question
        logger.info(f"Processing question: {question}")
        result = self.graph.invoke({"question": question})
        
        # Log final summary
        self.log_final_result(result)
        
        return result


def test_and_validate_rag_pipeline(pipeline: RAGPipeline, question: str):
    """Test the RAG pipeline and validate the output."""
    # Run the pipeline
    result = test_rag_pipeline(pipeline, question)

    # Validate the answer
    validation_result = pipeline.validate({"answer": result["answer"], "question": question})
    
    # Add validation results to pipeline result
    result.update(validation_result)
    
    # Return combined results
    return result


def test_rag_pipeline(pipeline: RAGPipeline, question: str):
    """Test the RAG pipeline without validation."""
    # Run the pipeline
    try:
        return pipeline.run(question)
    except Exception as e:
        logger.error(f"Error running RAG pipeline: {str(e)}")
        return {"answer": f"Error: {str(e)}", "error": True}


# Example usage
if __name__ == "__main__":
    load_dotenv()

    llm_client = LLMClient(
        provider="gemini",
        model="gemini-2.5-pro-preview-03-25",
        temperature=0.7,
        # top_p=0.8,
    )

    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)

    pipeline = RAGPipeline(
        vector_store,
        llm_client=llm_client,
        run_type="main_example"
    )

    question = "How do I set up an AWS S3 bucket with versioning and encryption that complies with CIS benchmarks?"

    test_rag_pipeline(pipeline, question)

    # Uncomment the line below to run the test and validation pipeline
    # test_and_validate_rag_pipeline(pipeline, question)
