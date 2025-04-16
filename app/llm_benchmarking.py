import csv
import itertools
import subprocess
from datetime import datetime
from app.models import LLMClient
from app.prompt_loader import PromptingStrategy
from app.benchmark_runner import BenchmarkRunner

TEMPERATURES = [0.1, 0.7]
# TOP_P_LIST = [0.8, 1]
MODEL = "mistral-large-latest"
MODEL_PROVIDER = "mistralai"


def run_experiments():
    """
    Run benchmark experiments for different temperatures, max_tokens, and prompt strategies.
    Store results in a CSV file.
    """

    # Run experiments for all combinations
    for temp, strategy in itertools.product(TEMPERATURES, list(PromptingStrategy)):
        print(f"Running: temp={temp}, strategy={strategy}")
        OUTPUT_FILE = f"benchmarking_results/{MODEL}_{temp}_{strategy}.json"
        CHECKOV_OUTPUT_DIR = "checkov_output"

        llm_client = LLMClient(
            provider=MODEL_PROVIDER,
            model=MODEL,
            temperature=temp,
        )

        benchmark = BenchmarkRunner(
            output_file=OUTPUT_FILE,
            llm_client=llm_client,
            checkov_output_dir=CHECKOV_OUTPUT_DIR,
            test_mode=True,
            prompting_strategy=strategy,
        )
        benchmark.run()


if __name__ == "__main__":
    # Run the experiments
    run_experiments()
