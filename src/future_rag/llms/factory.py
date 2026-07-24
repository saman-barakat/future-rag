"""
Factory for creating Large Language Model (LLM) instances.
"""

import os
from typing import Any, Dict, Type
from llama_index.core import Settings
from llama_index.core.llms import LLM

from future_rag.llms.deepseek import DeepSeekLLM



class LLMFactory:
    """
    Factory for instantiating LLMs and updating global LlamaIndex Settings.
    """

    # Default to env var if available, otherwise 'deepseek'
    DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

    # Map provider string names to their respective classes
    _PROVIDERS: Dict[str, Type[LLM]] = {
        "deepseek": DeepSeekLLM
    }

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        set_as_global_default: bool = True,
        **kwargs: Any,
    ) -> LLM:
        """
        Create an LLM instance and register it in Settings.llm.
        """
        provider_name = (provider or cls.DEFAULT_PROVIDER).lower()

        if provider_name not in cls._PROVIDERS:
            supported = ", ".join(cls._PROVIDERS.keys())
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Supported providers: {supported}"
            )

        # Instantiate provider from registry
        llm_class = cls._PROVIDERS[provider_name]
        llm_instance = llm_class(**kwargs)

        # Set globally in LlamaIndex
        if set_as_global_default:
            Settings.llm = llm_instance

        return llm_instance