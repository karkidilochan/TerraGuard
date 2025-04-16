import os
import logging
from app.prompt_loader import PromptLoader, PromptingStrategy
from langchain.chat_models import init_chat_model
from google import genai
from google.genai import types
from app.api_keys import GEMINI_API_KEY, MISTRAL_API_KEY, OPENAI_API_KEY
import time

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
            self.client = genai.Client(api_key=GEMINI_API_KEY)

        # elif self.provider == "openai":
        #     self.client = OpenAI(api_key=OPENAI_API_KEY)

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

            # elif self.provider == "openai":
            #     messages = [
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": prompt},
            #     ]

            #     response = self.client.ChatCompletion.create(
            #         model=self.model,  # Supports any OpenAI chat model
            #         messages=messages,
            #         temperature=self.temperature,
            #         max_tokens=self.max_tokens,
            #     )

            #     return response.choices[0]["message"]["content"]

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"Error running LLM: {str(e)}")
            return ""
