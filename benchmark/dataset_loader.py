from typing import List

from benchmark_logger import logger
from datasets import load_dataset


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
