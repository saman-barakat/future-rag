"""
DeepSeek LLM implementation for FUTURE-RAG.
"""

import os
from typing import Any, Optional
from llama_index.llms.deepseek import DeepSeek
from typing import ClassVar

class DeepSeekLLM(DeepSeek):
    """
    DeepSeek Chat model wrapper for FUTURE-RAG.
    """

    DEFAULT_MODEL : ClassVar[str] = "deepseek-chat"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize DeepSeek model with environment checks.
        """
        # Resolve API key cleanly
        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not resolved_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is missing. "
                "Please set the DEEPSEEK_API_KEY environment variable or pass `api_key` directly."
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=resolved_api_key,
            **kwargs,
        )