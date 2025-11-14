import logging
import time

from google import genai
from google.genai import types
from langchain.chat_models import init_chat_model
from settings import settings

from app.prompting.prompt_loader import PromptLoader

# from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Handles LLM interactions for multiple providers (simple version)."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        # top_p: int,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        # self.top_p = top_p
        self.prompt_loader = PromptLoader()

        # Initialize client during construction
        if self.provider == "mistralai":
            self.client = init_chat_model(
                self.model,
                model_provider=self.provider,
                temperature=self.temperature,
                # top_p=self.top_p,
            )
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def run(self, prompt: str, system_prompt: str) -> str:
        """Run the LLM with the given prompt and strategy."""
        try:

            if self.provider == "mistralai":
                max_retries = 3
                backoff_time = 2

                for attempt in range(max_retries):
                    try:
                        answer = self.client.invoke(
                            system_prompt + "\n Question:" + prompt
                        )

                        return answer.content
                    except Exception as e:
                        if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                            logger.warning(
                                f"Rate limit hit, retrying in {backoff_time} seconds (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(backoff_time)
                            backoff_time *= 2  # Exponential backoff
                        else:
                            logger.error(f"Error generating code: {str(e)}")
                            # Return an empty answer if we hit an error after all retries
                            return "Error generating code due to API limitations. Please try again later."

            elif self.provider == "gemini":
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        system_instruction=system_prompt,
                        # top_p=self.top_p,
                    ),
                    contents=prompt,
                )
                return response.text
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"Error running LLM: {str(e)}")
            return ""


if __name__ == "__main__":
    # Example usage
    llm_client = LLMClient(
        provider="gemini",
        model="gemini-2.0-flash",
        temperature=0.7,
        # top_p=0.8,
    )
    from app.prompting.prompt_loader import PromptLoader

    prompt_loader = PromptLoader()
    system_prompt = prompt_loader.system_prompt
    prompt = "How do I set up an AWS S3 bucket with versioning and encryption that complies with CIS benchmarks?"
    response = llm_client.run(prompt, system_prompt)
    print(response)
