import os
import logging
import enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


PROMPTS_DIR = "./app/prompts"


class PromptingStrategy(enum.Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RAG = "rag"


class PromptLoader:
    """Handles loading prompts from txt files."""

    def __init__(self):
        self.prompts_dir = PROMPTS_DIR
        self.system_prompt = None
        self.few_shot_prompt = None
        self.cot_prompt = None

        with open(f"{PROMPTS_DIR}/system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        with open(f"{PROMPTS_DIR}/cot.txt", "r") as f:
            self.cot_prompt = f.read()

        with open(f"{PROMPTS_DIR}/few-shot.txt", "r") as f:
            self.few_shot_prompt = f.read()
