import csv
import itertools
import subprocess
from datetime import datetime
from app.models import LLMClient
from app.prompt_loader import PromptingStrategy
from app.benchmark_runner import BenchmarkRunner

TEMPERATURES = [0.1, 0.7]
# TOP_P_LIST = [0.8, 1]
MISTRALAI_MODEL = "mistral-large-latest"
MISTRAL_MODEL_PROVIDER = "mistralai"

GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"
GEMINI_MODEL_PROVIDER = "gemini"


def run_experiments(model_provider, model, strategy, temperature, test_mode=False):
    """
    Run benchmark experiments for different temperatures, max_tokens, and prompt strategies.
    Store results in a CSV file.
    """

    # Run experiments for all combinations

    print(f"Running: temp={temperature}, strategy={strategy}, model={model}")
    OUTPUT_FILE = (
        f"benchmarking_results/{model_provider}/{model}_{temperature}_{strategy}.json"
    )
    CHECKOV_OUTPUT_DIR = "checkov_output"

    llm_client = LLMClient(
        provider=model_provider,
        model=model,
        temperature=temperature,
    )

    benchmark = BenchmarkRunner(
        output_file=OUTPUT_FILE,
        llm_client=llm_client,
        checkov_output_dir=CHECKOV_OUTPUT_DIR,
        test_mode=test_mode,
        prompting_strategy=strategy,
    )
    benchmark.run()


if __name__ == "__main__":
    # Run the experiments
    run_experiments(
        GEMINI_MODEL_PROVIDER,
        GEMINI_MODEL,
        PromptingStrategy.RAG,
        0.1,
        test_mode=False,
    )
