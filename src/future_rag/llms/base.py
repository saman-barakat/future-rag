"""
Base interface for all Large Language Models in FUTURE-RAG.
"""

from llama_index.core.llms import LLM

# Use LlamaIndex's native LLM class as our base type.
# This ensures total compatibility with all LlamaIndex indices, retrievers, and query engines.
BaseLLM = LLM