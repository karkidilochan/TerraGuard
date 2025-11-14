import json
from typing import Any, Dict, List

from benchmark_logger import logger


class StoreResult:
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
