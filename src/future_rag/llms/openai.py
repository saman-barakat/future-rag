"""
OpenAI LLM configuration for FUTURE-RAG.
"""

import os

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from future_rag.llms.base import LLMProvider


class OpenAILLM(LLMProvider):
    """
    OpenAI LLM provider.

    Responsible for configuring and creating a
    LlamaIndex-compatible OpenAI LLM instance.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        """
        Initialize the OpenAI LLM configuration.

        Parameters
        ----------
        model:
            OpenAI model name.

        api_key:
            OpenAI API key. If not provided, the
            OPENAI_API_KEY environment variable is used.

        temperature:
            Controls randomness in the generated response.

        max_tokens:
            Maximum number of tokens generated in the response.
        """

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY or provide api_key."
            )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_llm(self) -> LLM:
        """
        Create and return a LlamaIndex-compatible OpenAI LLM.
        """

        return OpenAI(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )