import os
import json
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from benchmark_oop import BenchmarkRunner, PromptingStrategy
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("multi_llm_benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MultiLLMBenchmark:
    """Manages benchmarking multiple LLMs with different prompting strategies."""

    # Mapping of LLM providers to their expected API key environment variables
    API_KEY_MAP = {
        "mistralai": "MISTRALAI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        checkov_output_dir: str,
        llm_configs: List[Tuple[str, str]],
        prompting_strategies: List[str] = None,
        dataset_split: str = "train",
    ):
        """
        Initialize with dataset, LLM configurations, and prompting strategies.

        Args:
            dataset_name: Hugging Face dataset name
            output_dir: Directory for benchmark result files
            checkov_output_dir: Directory for Checkov validation outputs
            llm_configs: List of (llm_model, llm_provider) tuples
            prompting_strategies: List of prompting strategies to use
            dataset_split: Dataset split to use (default: 'train')
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.checkov_output_dir = checkov_output_dir
        self.llm_configs = llm_configs
        self.prompting_strategies = prompting_strategies or [
            PromptingStrategy.ZERO_SHOT.value,
            PromptingStrategy.FEW_SHOT.value,
            PromptingStrategy.CHAIN_OF_THOUGHT.value,
        ]
        self.dataset_split = dataset_split

        # Load environment variables
        load_dotenv()

        # Validate API keys
        self.validate_api_keys()

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkov_output_dir, exist_ok=True)

    def validate_api_keys(self) -> None:
        """Check that API keys for all LLM providers are available."""
        missing_keys = []
        providers = set(provider for _, provider in self.llm_configs)

        for provider in providers:
            api_key_var = self.API_KEY_MAP.get(provider)
            if not api_key_var:
                missing_keys.append(
                    f"Unknown provider '{provider}' (no API key mapping)"
                )
                continue
            if not os.getenv(api_key_var):
                missing_keys.append(f"Missing API key for {provider}: {api_key_var}")

        if missing_keys:
            error_msg = "API key validation failed:\n" + "\n".join(missing_keys)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("All required API keys are present")

    def compute_logistics(self, results: List[dict]) -> dict:
        """Compute logistics from benchmark results."""
        successes = 0
        failures = 0
        total_time = 0.0

        for result in results:
            validation_status = result.get("validation_summary", {}).get(
                "status", "error"
            )
            if validation_status == "passed":
                successes += 1
            else:
                failures += 1
            total_time += result.get("run_time", 0.0)

        return {
            "successful_prompts": successes,
            "failed_prompts": failures,
            "total_prompts": len(results),
            "total_run_time_seconds": round(total_time, 2),
            "average_run_time_per_prompt_seconds": (
                round(total_time / len(results), 2) if results else 0.0
            ),
        }

    def save_logistics(self, logistics: dict, output_file: str) -> None:
        """Save logistics to a JSON file."""
        try:
            with open(output_file, "w") as f:
                json.dump(logistics, f, indent=4)
            logger.info(f"Logistics saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving logistics to {output_file}: {str(e)}")

    def run(self) -> None:
        """Run benchmark for each LLM and prompting strategy."""
        for llm_model, llm_provider in self.llm_configs:
            for strategy in self.prompting_strategies:
                logger.info(
                    f"Starting benchmark for LLM: {llm_model} ({llm_provider}), Strategy: {strategy}"
                )

                # Generate unique output file names
                safe_model_name = llm_model.replace("/", "_").replace(":", "_")
                safe_strategy = strategy.replace("_", "-")
                result_file = os.path.join(
                    self.output_dir,
                    f"benchmark_results_{safe_model_name}_{safe_strategy}.json",
                )
                logistics_file = os.path.join(
                    self.output_dir, f"logistics_{safe_model_name}_{safe_strategy}.json"
                )

                # Create checkov output subdirectory
                llm_checkov_dir = os.path.join(
                    self.checkov_output_dir, f"{safe_model_name}_{safe_strategy}"
                )

                try:
                    # Initialize BenchmarkRunner
                    benchmark = BenchmarkRunner(
                        dataset_name=self.dataset_name,
                        output_file=result_file,
                        llm_model=llm_model,
                        llm_provider=llm_provider,
                        checkov_output_dir=llm_checkov_dir,
                        dataset_split=self.dataset_split,
                        prompting_strategy=strategy,
                    )

                    # Run the benchmark
                    benchmark.run()

                    # Compute and save logistics
                    logistics = self.compute_logistics(benchmark.results)
                    self.save_logistics(logistics, logistics_file)

                    logger.info(
                        f"Completed benchmark for {llm_model} ({strategy}). "
                        f"Results: {result_file}, Logistics: {logistics_file}"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed benchmark for {llm_model} ({llm_provider}, {strategy}): {str(e)}"
                    )
                    # Continue with next LLM/strategy


if __name__ == "__main__":
    # Configuration
    DATASET_NAME = "autoiac-project/iac-eval"
    OUTPUT_DIR = "benchmark_results"
    CHECKOV_OUTPUT_DIR = "checkov_output"

    # Define LLM configurations: (model, provider) pairs
    LLM_CONFIGS = [
        ("mistral-large-latest", "mistralai"),
        ("gpt-4", "openai"),
        ("claude-3-opus", "anthropic"),
        ("gemini", "gemini"),
    ]

    # Define prompting strategies (optional, defaults to all)
    PROMPTING_STRATEGIES = [
        PromptingStrategy.ZERO_SHOT.value,
        PromptingStrategy.FEW_SHOT.value,
        PromptingStrategy.CHAIN_OF_THOUGHT.value,
    ]

    try:
        # Initialize and run multi-LLM benchmark
        multi_benchmark = MultiLLMBenchmark(
            dataset_name=DATASET_NAME,
            output_dir=OUTPUT_DIR,
            checkov_output_dir=CHECKOV_OUTPUT_DIR,
            llm_configs=LLM_CONFIGS,
            prompting_strategies=PROMPTING_STRATEGIES,
        )
        multi_benchmark.run()
    except ValueError as e:
        logger.critical(f"Cannot proceed with benchmarking: {str(e)}")
        exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        exit(1)
