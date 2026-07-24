"""
Custom exceptions for FUTURE-RAG.
"""


class FutureRAGError(Exception):
    """Base exception for all FUTURE-RAG errors."""

    pass


class ConfigurationError(FutureRAGError):
    """Raised when the project configuration is invalid."""

    pass


class DocumentError(FutureRAGError):
    """Raised when a document cannot be loaded or processed."""

    pass


class IndexError(FutureRAGError):
    """Raised when index creation or loading fails."""

    pass


class RetrievalError(FutureRAGError):
    """Raised when document retrieval fails."""

    pass


class LLMError(FutureRAGError):
    """Raised when an LLM request fails."""

    pass