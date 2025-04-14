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
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PromptingStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RAG = "rag"


class DatasetLoader:
    """Handles loading and validation of the dataset from Hugging Face."""

    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split

    def load(self) -> List[str]:
        """Load prompts from the Hugging Face dataset's 'Prompt' column."""
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(self.dataset_name, split="test")
            # Check for 'Prompt' column (case-sensitive, as per sample)
            if "Prompt" not in dataset.features:
                raise ValueError("Dataset must contain a 'Prompt' column")
            prompts = [
                item["Prompt"] for item in dataset if item["Prompt"]
            ]  # Skip empty prompts
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
    """Manages the benchmarking process using RAGPipeline."""

    def __init__(
        self,
        dataset_name: str,
        output_file: str,
        llm_model: str = "mistral-large-latest",
        llm_provider: str = "mistralai",
        checkov_output_dir: str = "checkov_output",
        dataset_split: str = "train",
        test_mode: bool = False,
    ):
        self.dataset_loader = DatasetLoader(dataset_name, dataset_split)
        self.result_saver = ResultSaver(output_file)
        self.checkov_output_dir = checkov_output_dir
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.pipeline = None
        self.results: List[Dict[str, Any]] = []
        self.test_mode = test_mode

    def initialize_pipeline(self) -> None:
        """Initialize the VectorStore and RAGPipeline."""
        try:
            vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
            self.pipeline = RAGPipeline(
                vector_store=vector_store,
                llm_model=self.llm_model,
                llm_provider=self.llm_provider,
                checkov_output_dir=self.checkov_output_dir,
            )
            logger.info("Initialized RAGPipeline successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAGPipeline: {str(e)}")
            raise

    def process_prompt(self, prompt: str, prompt_id: int) -> Dict[str, Any]:
        """Process a single prompt and return the result."""
        logger.info(f"Processing prompt {prompt_id}: {prompt[:100]}...")
        try:
            result = self.pipeline.run(prompt)
            return {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "generated_code": result.get("answer", ""),
                "validation_summary": result.get("validation_summary", {}),
                "compliance_report": result.get("validation_results", {}).get(
                    "compliance_report", ""
                ),
                "timestamp": datetime.utcnow().isoformat(),
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
            }

    def run(self) -> None:
        """Run the benchmark for all prompts."""
        # Load environment variables
        load_dotenv()

        # Initialize pipeline
        self.initialize_pipeline()

        # Ensure checkov output directory exists
        os.makedirs(self.checkov_output_dir, exist_ok=True)

        # Load prompts
        self.prompts = self.dataset_loader.load()

        # Verify expected number of prompts
        expected_prompts = 458
        if len(self.prompts) != expected_prompts:
            logger.warning(
                f"Expected {expected_prompts} prompts, but loaded {len(self.prompts)}"
            )

        # Process prompts with progress bar
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Processing prompts")):
            result = self.process_prompt(prompt, idx + 1)
            self.results.append(result)

            # Save intermediate results every 10 prompts
            if self.test_mode:
                self.result_saver.save(self.results)
                logger.info(f"Ending test run.")
                break
            elif (idx + 1) % 10 == 0:
                self.result_saver.save(self.results)
                logger.info(f"Intermediate results saved after {idx + 1} prompts")

        # Save final results
        self.result_saver.save(self.results)
        logger.info(
            f"Benchmark completed. Results saved to {self.result_saver.output_file}"
        )


if __name__ == "__main__":
    # Configuration
    DATASET_NAME = "autoiac-project/iac-eval"  # Hugging Face dataset name
    OUTPUT_FILE = "benchmark_results.json"
    CHECKOV_OUTPUT_DIR = "checkov_output"

    print(DATASET_NAME, OUTPUT_FILE)
    # # Run the benchmark
    benchmark = BenchmarkRunner(
        dataset_name=DATASET_NAME,
        output_file=OUTPUT_FILE,
        checkov_output_dir=CHECKOV_OUTPUT_DIR,
        test_mode=True,
    )
    benchmark.run()


""" 
TODO: 
1. add temperature and other variables to llms for determinism tests
2. add different prompting strategies, load the prompt txt files for these prompting strategies, rag_pipeline is used only for RAG strategy

"""
