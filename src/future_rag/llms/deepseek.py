"""
DeepSeek LLM configuration for FUTURE-RAG.
"""

import os

from llama_index.llms.openai_like import OpenAILike


class DeepSeekLLM:
    """
    DeepSeek LLM wrapper.

    Responsible for configuring and creating
    the LlamaIndex-compatible LLM instance.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        api_base: str = "https://api.deepseek.com",
        temperature: float = 0.1,
        context_window: int = 64000,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens


    def get_llm(self):
        """
        Create and return LlamaIndex LLM instance.
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