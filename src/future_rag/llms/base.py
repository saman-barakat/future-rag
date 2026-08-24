"""
Base interface for LLM providers.
"""

from abc import ABC, abstractmethod

from llama_index.core.llms import LLM


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    """

    @abstractmethod
    def get_llm(self) -> LLM:
        """
        Create and return a LlamaIndex-compatible LLM.
        """
        pass