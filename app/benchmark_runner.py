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
                vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
                self.pipeline = RAGPipeline(
                    vector_store=vector_store,
                    llm_client=self.llm_client,
                )
                logger.info("Initialized RAGPipeline successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAGPipeline: {str(e)}")
                raise

    def process_prompt(self, prompt: str, prompt_id: int) -> Dict[str, Any]:
        """Process a single prompt based on the selected strategy."""
        logger.info(f"Processing prompt {prompt_id}: {prompt[:100]}...")
        try:
            generated_code = ""
            if self.prompting_strategy == PromptingStrategy.RAG:
                result = self.pipeline.run(prompt)
                generated_code = result.get("answer", "")
            else:
                generated_code = self.llm_client.run(prompt, self.full_prompt)
            val_result = self.validate(prompt, generated_code)
            return {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "generated_code": generated_code,
                "validation_summary": val_result.get("validation_summary", {}),
                "compliance_report": val_result.get("validation_results", {}).get(
                    "compliance_report", ""
                ),
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": self.prompting_strategy.value,
            }

        except Exception as e:
            logger.error(f"Error processing prompt {prompt_id}: {str(e)}")
            return {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "generated_code": "",
                "validation_summary": {"status": "error", "issues": [str(e)]},
                "compliance_report": "",
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": self.prompting_strategy.value,
            }

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

            self.results.append(
                {
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
            )

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
