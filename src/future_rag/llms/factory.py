"""
Factory for creating LlamaIndex-compatible LLMs.
"""

from llama_index.core.llms import LLM

from future_rag.llms.base import LLMProvider
from future_rag.llms.deepseek import DeepSeekLLM
from future_rag.llms.openai import OpenAILLM


class LLMFactory:
    """
    Factory responsible for creating LLM providers.
    """

    _providers: dict[str, type[LLMProvider]] = {
        "deepseek": DeepSeekLLM,
        "openai": OpenAILLM,
    }

    @classmethod
    def create(
        cls,
        provider: str,
        **kwargs,
    ) -> LLM:
        """
        Create an LLM based on the selected provider.

        Parameters
        ----------
        provider:
            Name of the LLM provider.

        kwargs:
            Provider-specific configuration parameters.

        Returns
        -------
        LLM
            A LlamaIndex-compatible LLM instance.
        """

        provider = provider.lower()

        if provider not in cls._providers:
            supported = ", ".join(cls._providers.keys())

            raise ValueError(
                f"Unsupported LLM provider '{provider}'. "
                f"Supported providers: {supported}"
            )

        provider_class = cls._providers[provider]

        return provider_class(**kwargs).get_llm()