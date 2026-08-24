"""
DeepSeek LLM configuration for FUTURE-RAG.
"""

import os

from llama_index.core.llms import LLM
from llama_index.llms.openai_like import OpenAILike

from future_rag.llms.base import LLMProvider


class DeepSeekLLM(LLMProvider):
    """
    DeepSeek LLM provider.

    Responsible for configuring and creating a
    LlamaIndex-compatible DeepSeek LLM instance.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        api_base: str = "https://api.deepseek.com",
        temperature: float = 0.2,
        context_window: int = 64000,
        max_tokens: int = 4096,
    ):
        """
        Initialize the DeepSeek LLM configuration.

        Parameters
        ----------
        model:
            DeepSeek model name.

        api_key:
            DeepSeek API key. If not provided, the
            DEEPSEEK_API_KEY environment variable is used.

        api_base:
            DeepSeek API endpoint.

        temperature:
            Controls randomness in the generated response.

        context_window:
            Maximum context window supported by the model.

        max_tokens:
            Maximum number of tokens generated in the response.
        """

        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. "
                "Set DEEPSEEK_API_KEY or provide api_key."
        )
        self.api_base = api_base
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens

    def get_llm(self) -> LLM:
        """
        Create and return a LlamaIndex-compatible DeepSeek LLM.
        """

        return OpenAILike(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            temperature=self.temperature,
            context_window=self.context_window,
            max_tokens=self.max_tokens,
            is_chat_model=True,
        )