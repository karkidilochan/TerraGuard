import os
import datetime
import time
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from app.validate_tf import validate_terraform_code
from app.rag_prompts import get_system_prompt, get_search_prompt
from app.vector_store import CHROMA_COLLECTION_NAME, CHROMA_DB_NAME, VectorStore
from app.schemas import Search, State
from app.checkov_validator import generate_compliance_report
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

        os.makedirs(self.checkov_output_dir, exist_ok=True)

        # Build and compile the LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(State)

        # Define the sequence of steps
        graph_builder.add_node("analyze_query", self.analyze_query)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        # graph_builder.add_node("validate", self.validate)

        # Define the edges
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_edge("analyze_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        # graph_builder.add_edge("validate", END)

        return graph_builder.compile()

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
                return {"search": parsed_query}
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
                    return {"search": {"query": state["question"]}}

    def retrieve(self, state: State) -> State:
        """Retrieve documents based on the query and section."""
        logger.info("Retrieving relevant documents...")

        # Log the search query for debugging
        logger.info(f"Search query: {state['search']}")

        # Retrieve documents from the vector store
        chroma_filter = [{"section": {"$in": ["Example Usage"]}}]
        if state["search"]["subcategories"]:
            chroma_filter.append(
                {"subcategory": {"$in": state["search"]["subcategories"]}}
            )
        retrieved_docs = self.vector_store.get_db_instance().similarity_search(
            query=state["search"]["query"],
            k=5,
            filter={
                "$and": chroma_filter,
            },
        )

        # Log the number of retrieved documents
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        return {"context": retrieved_docs}

    def generate(self, state: State) -> State:
        """Generate an answer using the retrieved context."""
        logger.info("Generating Terraform code...")

        # Extract context from retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        # Create prompt with question and context
        system_prompt = get_system_prompt()
        prompt = system_prompt.format(
            question=state.get("question", ""), context=docs_content
        )
        answer = self.llm_client.run(prompt, system_prompt)
        return {"answer": answer}

    def validate(self, state: State) -> State:
        """Validate the generated Terraform code against CIS controls."""
        logger.info("Validating generated Terraform code...")

        # Generate a timestamp for this validation
        import datetime as dt

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = (
            hash(state["question"]) % 10000
        )  # Simple hash for query identification

        # Validate the generated Terraform code
        validation_results = validate_terraform_code(
            state["answer"],
            output_dir=self.checkov_output_dir,
            result_file_prefix=f"validation_{timestamp}_{query_hash}",
        )

        # Extract results for better reporting
        is_syntax_valid = validation_results.get("syntax_valid", False)
        is_cis_compliant = validation_results.get("cis_compliant", False)
        referenced_cis_controls = validation_results.get("referenced_cis_controls", [])
        checkov_output_file = validation_results.get("checkov_output_file", "")

        # Generate a more detailed compliance report if we have results to work with
        compliance_report = validation_results.get("compliance_report", "")
        if not compliance_report and "checkov_results" in validation_results:
            try:
                compliance_report = generate_compliance_report(
                    validation_results["checkov_results"], referenced_cis_controls
                )
                validation_results["compliance_report"] = compliance_report
            except Exception as e:
                logger.error(f"Error generating compliance report: {e}")

        # Set validation status and provide detailed information
        validation_status = (
            "passed" if is_syntax_valid and is_cis_compliant else "failed"
        )
        validation_issues = []

        if not is_syntax_valid:
            validation_issues.append("Terraform syntax validation failed")

        if not is_cis_compliant:
            validation_issues.append("CIS compliance validation failed")

        if validation_results.get("errors"):
            validation_issues.extend(validation_results["errors"])

        # Add validation results to the state
        return {
            "validation_results": validation_results,
            "validation_summary": {
                "status": validation_status,
                "syntax_valid": is_syntax_valid,
                "cis_compliant": is_cis_compliant,
                "referenced_cis_controls": referenced_cis_controls,
                "issues": validation_issues,
                "checkov_output_file": checkov_output_file,
            },
        }

    def run(self, question: str) -> State:
        """Run the RAG pipeline for a given question."""
        logger.info(f"Processing question: {question}")

        initial_state = {
            "question": question,
        }

        # Execute the graph with the initial state
        result = self.graph.invoke(initial_state)

        logger.info("RAG pipeline execution completed")
        return result


def test_and_validate_rag_pipeline(pipeline: RAGPipeline, question: str):

    # Run the pipeline
    result_state = pipeline.run(question)

    result = pipeline.validate(result_state)

    # Print results
    print("\nFinal Result:")
    print(f"Question: {result_state['question']}")
    print(f"Query: {result_state['search']}")
    print("Context:")
    for doc in result_state["context"]:
        print(f"- {doc.page_content[:100]}... (Metadata: {doc.metadata})")
    print(f"Answer: {result_state['answer']}")

    # Print validation results
    print("\nValidation Results:")
    print(f"Status: {result['validation_summary']['status']}")
    print(f"Syntax Valid: {result['validation_summary']['syntax_valid']}")
    print(f"CIS Compliant: {result['validation_summary']['cis_compliant']}")
    print(
        f"Referenced CIS Controls: {result['validation_summary']['referenced_cis_controls']}"
    )

    if result["validation_summary"].get("issues"):
        print("\nValidation Issues:")
        for issue in result["validation_summary"]["issues"]:
            print(f"- {issue}")

    if result["validation_summary"].get("checkov_output_file"):
        print(
            f"\nCheckov Output File: {result['validation_summary']['checkov_output_file']}"
        )

    if result["validation_results"].get("compliance_report"):
        print("\nCompliance Report:")
        print(result["validation_results"]["compliance_report"])


def test_rag_pipeline(pipeline: RAGPipeline, question: str):
    # Run the pipeline
    result_state = pipeline.run(question)

    # Print results
    print("\nFinal Result:")
    print(f"Question: {result_state['question']}")
    print(f"Query: {result_state['search']}")
    print("Context:")
    for doc in result_state["context"]:
        print(f"- {doc.page_content[:100]}... (Metadata: {doc.metadata})")
    print(f"Answer: {result_state['answer']}")


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
    )

    question = "How do I set up an AWS S3 bucket with versioning and encryption that complies with CIS benchmarks?"

    test_rag_pipeline(pipeline, question)

    # Uncomment the line below to run the test and validation pipeline
    # test_and_validate_rag_pipeline(pipeline, question)
