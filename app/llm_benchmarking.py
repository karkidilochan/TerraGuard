import csv
import itertools
import subprocess
import argparse
from datetime import datetime
from app.models import LLMClient
from app.prompt_loader import PromptingStrategy
from app.benchmark_runner import BenchmarkRunner
import os

TEMPERATURES = [0.1, 0.7]
# TOP_P_LIST = [0.8, 1]
MISTRALAI_MODEL = "mistral-large-latest"
MISTRAL_MODEL_PROVIDER = "mistralai"

GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"
GEMINI_MODEL_PROVIDER = "gemini"


def run_experiments(model_provider, model, strategy, temperature, 
                   test_mode=False, use_feedback_loop=False, max_iterations=3):
    """
    Run benchmark experiments with optional feedback loop for iterative improvement.
    
    Args:
        model_provider: The LLM provider (gemini, mistralai)
        model: The specific model to use
        strategy: Prompting strategy (PromptingStrategy enum)
        temperature: Temperature setting for generation
        test_mode: Run only a single test if True
        use_feedback_loop: Enable the feedback loop for iterative improvement
        max_iterations: Maximum number of feedback iterations
    """
    print(f"Running: temp={temperature}, strategy={strategy}, model={model}")
    print(f"Feedback loop: {'enabled' if use_feedback_loop else 'disabled'}, max iterations: {max_iterations}")
    
    # Create output dirs
    os.makedirs("benchmarking_results", exist_ok=True)
    os.makedirs(f"benchmarking_results/{model_provider}", exist_ok=True)
    
    # Add feedback suffix to filename if enabled
    feedback_suffix = "_with_feedback" if use_feedback_loop else ""
    OUTPUT_FILE = (
        f"benchmarking_results/{model_provider}/{model}_{temperature}_{strategy}{feedback_suffix}.json"
    )
    CHECKOV_OUTPUT_DIR = "checkov_output"
    os.makedirs(CHECKOV_OUTPUT_DIR, exist_ok=True)

    llm_client = LLMClient(
        provider=model_provider,
        model=model,
        temperature=temperature,
    )
    
    # Check if we need to set up the vector store first
    # Only needed for RAG strategy
    vector_store = None
    if strategy == PromptingStrategy.RAG:
        from app.vector_store import VectorStore, CHROMA_DB_NAME, CHROMA_COLLECTION_NAME
        
        # Create and initialize the vector store with CIS benchmark data
        vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
        
        # Try multiple potential locations for the benchmark file
        cis_benchmark_locations = [
            "scripts/cis_benchmark_enhanced.json",
            "../scripts/cis_benchmark_enhanced.json",
            os.path.join(os.getcwd(), "scripts/cis_benchmark_enhanced.json")
        ]
        
        cis_loaded = False
        for benchmark_file in cis_benchmark_locations:
            print(f"Attempting to load CIS benchmark data from: {benchmark_file}")
            if os.path.exists(benchmark_file):
                cis_loaded = vector_store.load_cis_benchmark_data(benchmark_file)
                if cis_loaded:
                    print(f"Successfully loaded CIS benchmark data from {benchmark_file}")
                    break
        
        if not cis_loaded:
            print("WARNING: Failed to load CIS benchmark data. Vector search for CIS controls may not work properly.")

    benchmark = BenchmarkRunner(
        output_file=OUTPUT_FILE,
        llm_client=llm_client,
        checkov_output_dir=CHECKOV_OUTPUT_DIR,
        test_mode=test_mode,
        prompting_strategy=strategy,
        vector_store=vector_store,
        use_feedback_loop=use_feedback_loop,
        max_iterations=max_iterations
    )
    benchmark.run()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run benchmarking experiments")
    parser.add_argument("--feedback-loop", action="store_true", 
                       help="Enable feedback loop for iterative improvement")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum number of feedback iterations")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode (single prompt)")
    args = parser.parse_args()
    
    # Run with command line arguments
    run_experiments(
        GEMINI_MODEL_PROVIDER,
        GEMINI_MODEL,
        PromptingStrategy.RAG,
        0.1,
        test_mode=args.test_mode,
        use_feedback_loop=args.feedback_loop,
        max_iterations=args.max_iterations
    )
