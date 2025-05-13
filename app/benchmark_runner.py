import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any
from datasets import load_dataset
from dotenv import load_dotenv
from app.vector_store import VectorStore, CHROMA_DB_NAME, CHROMA_COLLECTION_NAME
from app.rag_pipeline import RAGPipeline
from app.validate_tf import validate_terraform_code
from app.checkov_validator import generate_compliance_report

from enum import Enum
from langchain.chat_models import init_chat_model
from app.models import LLMClient
from app.prompt_loader import PromptLoader, PromptingStrategy
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and validation of the dataset from Hugging Face."""

    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> List[str]:
        """Load prompts from the Hugging Face dataset's 'Prompt' column."""
        try:
            dataset = load_dataset(self.dataset_name, split="test")
            if "Prompt" not in dataset.features:
                raise ValueError("Dataset must contain a 'Prompt' column")
            prompts = [item["Prompt"] for item in dataset if item["Prompt"]]
            logger.info(
                f"Loaded dataset '{self.dataset_name}' with {len(prompts)} prompts"
            )
            return prompts
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise


class ResultSaver:
    """Handles saving benchmark results to a JSON file."""

    def __init__(self, output_file: str):
        self.output_file = output_file

    def save(self, results: List[Dict[str, Any]]) -> None:
        """Save results to the output JSON file."""
        try:
            with open(self.output_file, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Results saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


class BenchmarkRunner:
    """Manages the benchmarking process using different prompting strategies."""

    def __init__(
        self,
        output_file: str,
        llm_client: LLMClient,
        checkov_output_dir: str = "checkov_output",
        test_mode: bool = False,
        prompting_strategy: PromptingStrategy = PromptingStrategy.ZERO_SHOT,
        vector_store: VectorStore = None,
        use_feedback_loop: bool = False, 
        max_iterations: int = 3,
    ):
        self.dataset = load_dataset("autoiac-project/iac-eval", split="test")
        self.result_saver = ResultSaver(output_file)
        self.prompt_loader = PromptLoader()
        self.checkov_output_dir = checkov_output_dir
        self.llm_client = llm_client
        self.pipeline = None
        self.results: List[Dict[str, Any]] = []
        self.test_mode = test_mode
        self.prompting_strategy = prompting_strategy
        self.vector_store = vector_store
        # Feedback loop settings
        self.use_feedback_loop = use_feedback_loop
        self.max_iterations = max_iterations
        self.feedback_logs_dir = os.path.join(os.getcwd(), "feedback_logs")
        os.makedirs(self.feedback_logs_dir, exist_ok=True)

    def initialize_pipeline(self) -> None:
        """Initialize the VectorStore and RAGPipeline if using RAG strategy."""
        if self.prompting_strategy == PromptingStrategy.ZERO_SHOT:
            self.full_prompt = self.prompt_loader.system_prompt
        elif self.prompting_strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
            self.full_prompt = (
                f"{self.prompt_loader.system_prompt}\n\n{self.prompt_loader.cot_prompt}"
            )
        elif self.prompting_strategy == PromptingStrategy.FEW_SHOT:
            self.full_prompt = f"{self.prompt_loader.system_prompt}\n\n{self.prompt_loader.few_shot_prompt}"
        elif self.prompting_strategy == PromptingStrategy.RAG:
            try:
                # Use existing vector store if provided, otherwise create a new one
                if self.vector_store is None:
                    logger.info("Creating new vector store for benchmarking")
                    self.vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
                else:
                    logger.info("Using provided vector store for benchmarking")
                
                self.pipeline = RAGPipeline(
                    vector_store=self.vector_store,
                    llm_client=self.llm_client,
                    checkov_output_dir=self.checkov_output_dir,
                    run_type="benchmark",
                )
                logger.info("Initialized RAGPipeline successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAGPipeline: {str(e)}")
                raise
    
    def log_feedback_iteration(self, prompt_id, iteration, code, validation_results):
        """Log an iteration of the feedback loop."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get strategy name and temperature for filename
        strategy_name = self.prompting_strategy.name.lower()
        temp = str(self.llm_client.temperature).replace('.', '_')
        
        log_data = {
            "prompt_id": prompt_id,
            "iteration": iteration,
            "code": code,
            "validation_results": validation_results,
            "timestamp": timestamp
        }
        
        log_filename = f"bench_{strategy_name}_temp_{temp}_iteration_{prompt_id}_{iteration}_{timestamp}.json"
        log_path = os.path.join(self.feedback_logs_dir, log_filename)
        
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        return log_path

    def generate_feedback(self, validation_results):
        """Generate feedback based on validation results."""
        from app.error_summarizer import summarize_validation_errors
        
        # Process any Terraform syntax errors more explicitly to highlight argument name issues
        terraform_errors = []
        if "validation_results" in validation_results:
            errors = validation_results["validation_results"].get("errors", [])
            for error in errors:
                if isinstance(error, str) and ("Missing required argument" in error or "Unsupported argument" in error):
                    # Extract important error details
                    lines = error.split('\n')
                    error_type = ""
                    resource_type = ""
                    resource_name = ""
                    argument_name = ""
                    
                    for line in lines:
                        if "Error:" in line:
                            error_type = line.split("Error:")[1].strip()
                        if "resource" in line:
                            parts = line.split("resource")
                            if len(parts) > 1:
                                resource_parts = parts[1].strip().split()
                                if len(resource_parts) >= 2:
                                    resource_type = resource_parts[0].strip('"')
                                    resource_name = resource_parts[1].strip('"')
                        if "argument" in line:
                            parts = line.split("argument")
                            if len(parts) > 1 and '"' in parts[1]:
                                argument_name = parts[1].split('"')[1]
                    
                    if error_type and resource_type and argument_name:
                        if "Missing required argument" in error:
                            terraform_errors.append(f"IMPORTANT: {error_type} for resource '{resource_type}.{resource_name}'. The argument '{argument_name}' is required but missing.")
                        elif "Unsupported argument" in error:
                            terraform_errors.append(f"IMPORTANT: {error_type} for resource '{resource_type}.{resource_name}'. The argument '{argument_name}' is not supported. Remove it or use a valid argument name.")
                            
                            # Add special handling for common aws_route53_query_log errors
                            if resource_type == "aws_route53_query_log" and argument_name == "name":
                                terraform_errors.append("CORRECTION EXAMPLE: For aws_route53_query_log resources, use 'zone_id' instead of 'name' to specify the Route53 zone. Example: zone_id = aws_route53_zone.primary.zone_id")
        
        # Get standard error summary
        standard_feedback = summarize_validation_errors(validation_results)
        
        # Combine specialized Terraform error feedback with standard feedback
        if terraform_errors:
            return "SPECIFIC TERRAFORM ERRORS:\n" + "\n".join(terraform_errors) + "\n\n" + standard_feedback
        
        return standard_feedback

    def regenerate_with_feedback(self, prompt, previous_code, feedback, referenced_cis_controls=None):
        """Regenerate code with feedback from validation."""
        logger.info("Regenerating code with feedback")
        
        if self.prompting_strategy == PromptingStrategy.RAG:
            # For RAG, use the pipeline which handles feedback internally
            state = {
                "question": prompt,
                "answer": previous_code,
                "validation_feedback": feedback,
                "referenced_cis_controls": referenced_cis_controls or []
            }
            result = self.pipeline.regenerate(state)
            return result.get("answer", "")
        else:
            # For non-RAG strategies, construct a structured feedback prompt
            
            # Check if there's an issue with aws_route53_query_log resource
            route53_example = ""
            if "aws_route53_query_log" in previous_code and ("name" in previous_code or "zone_id" not in previous_code):
                route53_example = """
SPECIFIC EXAMPLE FOR ROUTE53 QUERY LOG:
```terraform
# Correct syntax for aws_route53_query_log resource:
resource "aws_route53_query_log" "example" {
  zone_id                  = aws_route53_zone.primary.zone_id  # Required parameter
  cloudwatch_log_group_arn = aws_cloudwatch_log_group.example.arn
}
```
The 'zone_id' parameter is required and 'name' parameter is not supported.
"""
            
            feedback_prompt = f"""
I need you to fix the Terraform code for this request: {prompt}

The code you provided has these validation errors:
{feedback}

{route53_example}

IMPORTANT INSTRUCTIONS:
1. Pay careful attention to the error messages which identify:
   - Missing required arguments
   - Unsupported arguments
   - Incorrect attribute names or references
2. Make sure to use the exact argument names as specified in the errors
3. Ensure all required arguments are included
4. Remove any unsupported arguments
5. Check resource attribute references to use the correct property names

CIS COMPLIANCE REQUIREMENTS:
1. Always encrypt CloudWatch Log Groups with KMS using the 'kms_key_id' attribute
2. Set log retention to at least 365 days (1 year) using 'retention_in_days'
3. Enable DNSSEC signing for Route 53 zones where applicable
4. Ensure all resources have proper access controls and least privilege
5. Add secure condition blocks to IAM policies when granting permissions

Previous code:
```terraform
{previous_code}
```

Please provide only fixed Terraform code that resolves all errors and is CIS compliant. Start with ```terraform and end with ```.
"""
            # Use the strategy-specific system prompt and pass feedback as user prompt
            system_prompt = self.full_prompt
            
            # Generate new code using the feedback-enhanced prompt
            regenerated_code = self.llm_client.run(feedback_prompt, system_prompt)
            return regenerated_code

    def log_benchmark_final_state(self, prompt: str, prompt_id: int, code: str, validation: Dict[str, Any]) -> str:
        """Create a final summary of the benchmark run and save it to a file.
        
        This method uses the best validation results across all iterations.
        """
        # Generate timestamp and run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4().int)[:3]
        
        # Get strategy name and temperature for filename
        strategy_name = self.prompting_strategy.name.lower()
        temp = str(self.llm_client.temperature).replace('.', '_')
        
        # Extract validation information
        validation_results = validation.get("validation_results", {})
        validation_summary = validation.get("validation_summary", {})
        
        syntax_valid = validation_summary.get("syntax_valid", False)
        cis_compliant = validation_summary.get("cis_compliant", False)
        
        # Extract checkov results, handling different structures based on strategy
        checkov_results = {}
        if "checkov_results" in validation_results:
            checkov_results = validation_results.get("checkov_results", {})
        
        # For the RAG pipeline, the results structure might be different
        if self.prompting_strategy == PromptingStrategy.RAG and isinstance(checkov_results, list):
            # Extract the first item if it's a list
            checkov_results = checkov_results[0] if checkov_results else {}
        
        # Extract pass rate and other metrics, defaulting to sensible values if missing or syntax is invalid
        pass_rate = 0
        total_checks = 0
        passed = 0
        failed = 0
        
        if syntax_valid and isinstance(checkov_results, dict):
            # Get validation summary from checkov results
            checkov_summary = checkov_results.get("validation_summary", {})
            
            # Extract metrics
            pass_rate = checkov_summary.get("pass_rate", 0)
            total_checks = checkov_summary.get("total", 0)
            passed = checkov_summary.get("passed", 0)
            failed = checkov_summary.get("failed", 0)
        else:
            # Set sensible defaults when syntax validation fails
            # At least one check was run (the syntax check) and it failed
            total_checks = 1
            failed = 1
            pass_rate = 0
        
        # Create the summary object
        summary = {
            "timestamp": timestamp,
            "run_type": "benchmark",
            "run_id": int(run_id),
            "prompt_id": prompt_id,
            "question": prompt,
            "total_iterations": validation.get("total_iterations", 1) if self.use_feedback_loop else 1,
            "final_code": code,
            "final_state": {
                "syntax_valid": syntax_valid,
                "cis_compliant": cis_compliant,
                "pass_rate": pass_rate,
                "total_checks": total_checks,
                "passed": passed,
                "failed": failed
            }
        }
        
        # Add information about the used model
        if hasattr(self.llm_client, "model"):
            summary["model"] = self.llm_client.model
        
        # Handle different feedback log directories based on test mode
        feedback_logs_dir = "predefined_logs" if self.test_mode else "feedback_logs"
        os.makedirs(feedback_logs_dir, exist_ok=True)
        
        # Create a filename with a prefix based on the strategy
        prefix = f"bench_{strategy_name}_temp_{temp}"
        if self.test_mode:
            prefix = "predefined_test"
        elif hasattr(self, "current_real_failure") and self.current_real_failure:
            prefix = "real_failure"
        
        # Generate the full path for the summary file
        summary_file = os.path.join(
            feedback_logs_dir, 
            f"{prefix}_final_summary_{run_id}_{timestamp}.json"
        )
        
        # Save the summary to a file
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved final benchmark summary to {summary_file}")
        
        return summary_file

    def process_prompt(self, prompt: str, prompt_id: int) -> Dict[str, Any]:
        """Process a single prompt and return the results."""
        logger.info(f"Processing prompt {prompt_id}: {prompt[:50]}...")
        
        # Initialize RAG pipeline if not done already
        if not hasattr(self, "pipeline") or self.pipeline is None:
            self.initialize_pipeline()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start with a single generation
        current_code = ""
        iterations = []
        improved = False
        current_validation = None
        
        # Variables to track best results across iterations
        best_code = ""
        best_validation = None
        best_pass_rate = -1  # Start with a value that any real pass rate will exceed
        
        # Run pipeline on the prompt
        try:
            # Process using the RAG pipeline
            if self.prompting_strategy == PromptingStrategy.RAG:
                result = self.pipeline.run(prompt)
                current_code = result.get("answer", "")
                
                # With RAG pipeline, validation is already performed, so extract it
                if self.use_feedback_loop:
                    # The validation is already done by the pipeline, so use that result
                    current_validation = {
                        "validation_results": result.get("validation_results", {}),
                        "validation_summary": {
                            "status": "passed" if result.get("cis_compliant", False) else "failed",
                            "syntax_valid": result.get("syntax_valid", False),
                            "cis_compliant": result.get("cis_compliant", False),
                            "referenced_cis_controls": result.get("referenced_cis_controls", []),
                            "issues": result.get("issues", []),
                            "checkov_output_file": result.get("checkov_output_file", ""),
                        }
                    }
                else:
                    # Validate the generated code if feedback loop is not enabled
                    current_validation = self.validate(prompt, current_code)
            else:
                # For non-RAG strategies, use the appropriate prompt template
                # self.full_prompt was already set up in initialize_pipeline
                task_prompt = f"Generate Terraform code for the following: {prompt}"
                full_prompt = f"{self.full_prompt}\n\n{task_prompt}"
                current_code = self.llm_client.run(full_prompt, full_prompt)
                
                # Validate the generated code
                current_validation = self.validate(prompt, current_code)
            
            # Initialize best results with the initial generation
            best_code = current_code
            best_validation = current_validation.copy()
            
            # Calculate initial pass rate
            if current_validation["validation_summary"].get("syntax_valid", False):
                validation_results = current_validation.get("validation_results", {})
                checkov_results = validation_results.get("checkov_results", {})
                if isinstance(checkov_results, dict) and "validation_summary" in checkov_results:
                    best_pass_rate = checkov_results.get("validation_summary", {}).get("pass_rate", 0)
            
            # Handle feedback loop if enabled
            if self.use_feedback_loop and self.prompting_strategy != PromptingStrategy.RAG:
                # For non-RAG strategies, we need to handle the feedback loop manually
                iteration_counter = 1
                initial_validation = current_validation.copy()
                
                # Log initial iteration
                self.log_feedback_iteration(prompt_id, 0, current_code, current_validation)
                
                original_syntax_valid = current_validation.get("validation_summary", {}).get("syntax_valid", False)
                original_cis_compliant = current_validation.get("validation_summary", {}).get("cis_compliant", False)
                
                # Start feedback loop if validation failed
                while not (original_syntax_valid and original_cis_compliant) and iteration_counter <= self.max_iterations:
                    logger.info(f"Iteration {iteration_counter} for prompt {prompt_id}")
                    
                    # Generate feedback for improving the code
                    feedback = self.generate_feedback(current_validation)
                    
                    # Regenerate code with feedback
                    logger.info("Regenerating code with feedback")
                    referenced_controls = current_validation.get("validation_results", {}).get("referenced_cis_controls", [])
                    new_code = self.regenerate_with_feedback(prompt, current_code, feedback, referenced_controls)
                    
                    # Validate the new code
                    new_validation = self.validate(prompt, new_code)
                    
                    # Log iteration
                    self.log_feedback_iteration(prompt_id, iteration_counter, new_code, new_validation)
                    
                    # Update iteration data
                    iterations.append({
                        "iteration": iteration_counter,
                        "code": new_code,
                        "validation": new_validation
                    })
                    
                    # Check if code improved
                    new_syntax_valid = new_validation.get("validation_summary", {}).get("syntax_valid", False)
                    new_cis_compliant = new_validation.get("validation_summary", {}).get("cis_compliant", False)
                    
                    if (new_syntax_valid > original_syntax_valid or new_cis_compliant > original_cis_compliant):
                        improved = True
                    
                    # Check if this iteration has a better pass rate than our current best
                    current_pass_rate = -1
                    if new_syntax_valid:
                        validation_results = new_validation.get("validation_results", {})
                        checkov_results = validation_results.get("checkov_results", {})
                        if isinstance(checkov_results, dict) and "validation_summary" in checkov_results:
                            current_pass_rate = checkov_results.get("validation_summary", {}).get("pass_rate", 0)
                    
                    # Update best results if this iteration is better
                    if new_syntax_valid and current_pass_rate > best_pass_rate:
                        best_code = new_code
                        best_validation = new_validation.copy()
                        best_pass_rate = current_pass_rate
                        logger.info(f"New best result in iteration {iteration_counter} with pass rate {best_pass_rate:.1f}%")
                    
                    # Update current code and validation for next iteration
                    current_code = new_code
                    current_validation = new_validation
                    
                    # Break if validation succeeded
                    if new_syntax_valid and new_cis_compliant:
                        break
                    
                    iteration_counter += 1
                
                # Add iteration details to current validation
                current_validation["total_iterations"] = iteration_counter
                current_validation["improved"] = improved
                
                # Ensure best_validation has the total iterations count
                best_validation["total_iterations"] = iteration_counter
                best_validation["improved"] = improved
            
            # Create a final summary with the best results
            final_summary_path = self.log_benchmark_final_state(
                prompt, prompt_id, best_code, best_validation
            )
            
        except Exception as e:
            # Handle exceptions during processing
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}")
            current_code = f"Error: {str(e)}"
            current_validation = {
                "validation_summary": {
                    "status": "error",
                    "syntax_valid": False,
                    "cis_compliant": False,
                    "referenced_cis_controls": [],
                    "issues": [str(e)],
                }
            }
            best_code = current_code
            best_validation = current_validation
        
        # Extract CIS compliance report if available
        compliance_report = ""
        if "validation_results" in best_validation and "compliance_report" in best_validation["validation_results"]:
            compliance_report = best_validation["validation_results"]["compliance_report"]
        
        # Build the result dictionary
        result = {
            "prompt_id": prompt_id,
            "generated_code": best_code,  # Use best code in result
            "validation_summary": best_validation.get("validation_summary", {}),
            "compliance_report": compliance_report,
            "timestamp": timestamp,
            "strategy": self.prompting_strategy.name,
        }
        
        # Add feedback loop data if available
        if self.use_feedback_loop:
            result["iterations"] = iterations
            result["total_iterations"] = best_validation.get("total_iterations", 1)
            result["improved"] = improved
        
        return result

    def validate(self, prompt, generated_code):
        """Validate the generated Terraform code against CIS controls."""
        logger.info("Validating generated Terraform code...")

        # Generate a timestamp for this validation
        import datetime as dt

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hash(prompt) % 10000  # Simple hash for query identification

        # Validate the generated Terraform code
        validation_results = validate_terraform_code(
            generated_code,
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

    def run(self) -> None:
        """Run the benchmark for all prompts."""
        load_dotenv()
        self.initialize_pipeline()
        os.makedirs(self.checkov_output_dir, exist_ok=True)

        for idx, item in enumerate(tqdm(self.dataset, desc="Processing prompts")):
            result = self.process_prompt(item["Prompt"], idx + 1)

            # Create the benchmark result entry with standard fields
            benchmark_result = {
                "Prompt ID": result["prompt_id"],
                "Resource": item["Resource"],
                "Prompt": item["Prompt"],
                "Rego Intent": item["Rego intent"],
                "Difficulty": item["Difficulty"],
                "Reference Output": item["Reference output"],
                "Intent": item["Intent"],
                "Generated Code": result["generated_code"],
                "Validation Summary": result["validation_summary"],
                "Compliance Report": result["compliance_report"],
                "Timestamp": result["timestamp"],
                "Strategy": result["strategy"],
            }
            
            # Add feedback loop data if available
            if self.use_feedback_loop:
                if "iterations" in result:
                    benchmark_result["Iterations"] = result["iterations"]
                if "total_iterations" in result:
                    benchmark_result["Total Iterations"] = result["total_iterations"]
                if "improved" in result:
                    benchmark_result["Improved"] = result["improved"]
            
            self.results.append(benchmark_result)

            if self.test_mode:
                self.result_saver.save(self.results)
                logger.info(f"Ending test run.")
                break
            elif (idx + 1) % 10 == 0:
                self.result_saver.save(self.results)
                logger.info(f"Intermediate results saved after {idx + 1} prompts")

        self.result_saver.save(self.results)
        logger.info(
            f"Benchmark completed. Results saved to {self.result_saver.output_file}"
        )


if __name__ == "__main__":
    OUTPUT_FILE = "benchmark_results.json"
    CHECKOV_OUTPUT_DIR = "checkov_output"

    llm_client = LLMClient(
        provider="mistralai",
        model="mistral-large-latest",
        temperature=0.7,
        # top_p=0.8,
    )

    benchmark = BenchmarkRunner(
        output_file=OUTPUT_FILE,
        llm_client=llm_client,
        checkov_output_dir=CHECKOV_OUTPUT_DIR,
        test_mode=True,
        prompting_strategy=PromptingStrategy.RAG,
    )
    benchmark.run()
