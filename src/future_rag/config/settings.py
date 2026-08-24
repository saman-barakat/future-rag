from dataclasses import dataclass


@dataclass
class RAGConfig:
    """
    Main configuration for FUTURE-RAG pipeline.
    """

    # --------------------------------------------------
    # Document loading
    # --------------------------------------------------
    documents_path: str = "./data/knowledge/pdf"


    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------
    chunk_size: int = 1024
    chunk_overlap: int = 256


    # --------------------------------------------------
    # Embedding model
    # --------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"


    # --------------------------------------------------
    # Vector database
    # --------------------------------------------------
    vectorstore_path: str = "./data/vectorstores/chroma"
    collection_name: str = "future_rag"


    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    similarity_top_k: int = 5


    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-flash"
    temperature: float = 0.2
    max_tokens: int = 4096