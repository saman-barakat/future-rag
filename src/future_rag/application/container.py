from pathlib import Path

from future_rag.indexing.embeddings import EmbeddingFactory
from future_rag.indexing.indexer import IndexBuilder
from future_rag.indexing.loader import IndexLoader
from future_rag.indexing.vectorstore import VectorStore
from future_rag.knowledge.chunker import DocumentChunker
from future_rag.knowledge.loaders import DocumentLoader
from future_rag.llms.deepseek import DeepSeekLLM
from future_rag.llms.factory import LLMFactory
from future_rag.retrieval.retriever import Retriever


class ApplicationContainer:

    def __init__(
        self,
        config,
        project_root: Path,
    ):
        self.config = config
        self.project_root = project_root

        # -----------------------------------------------------
        # Infrastructure
        # -----------------------------------------------------

        self.vector_store_manager = None
        self.vector_store = None

        # -----------------------------------------------------
        # RAG dependencies
        # -----------------------------------------------------

        self.embedding_model = None
        self.index = None
        self.retriever = None
        self.llm = None

        # -----------------------------------------------------
        # Application state
        # -----------------------------------------------------

        self._initialized = False

    # =========================================================
    # Public API
    # =========================================================

    def initialize(self) -> None:
        """
        Initialize the application dependencies.

        If an existing vector index is available, it is loaded.
        Otherwise, the knowledge base is built from the source
        documents.

        This method should normally be called once when the
        application starts.
        """

        if self._initialized:
            return

        self._connect_vector_store()
        self._load_or_build_index()
        self._create_retriever()
        self._create_llm()

        self._initialized = True

    def build_index(self):
        """
        Build a new vector index from the source documents.

        Raises:
            RuntimeError:
                If a vector index already exists.
        """

        self._connect_vector_store()

        if self.has_index():
            raise RuntimeError(
                "A vector index already exists. "
                "Use rebuild_index() to rebuild it."
            )

        documents = self._load_documents()
        nodes = self._chunk_documents(documents)

        self._create_embedding_model()

        builder = IndexBuilder(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

        self.index = builder.build(
            nodes=nodes,
        )

        return self.index

    def rebuild_index(self):
        """
        Remove the existing vector index and rebuild it
        from the source documents.
        """

        self._connect_vector_store()
        self._clear_vector_store()

        documents = self._load_documents()
        nodes = self._chunk_documents(documents)

        self._create_embedding_model()

        builder = IndexBuilder(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

        self.index = builder.build(
            nodes=nodes,
        )

        # The old retriever is no longer valid because it
        # references the previous index.
        self.retriever = None

        if self._initialized:
            self._create_retriever()

        return self.index

    def has_index(self) -> bool:
        """
        Return True if the vector store contains vectors.
        """

        self._connect_vector_store()

        return self._get_vector_count() > 0

    @property
    def is_initialized(self) -> bool:
        """
        Return True if the application has been initialized.
        """

        return self._initialized

    # =========================================================
    # Vector store
    # =========================================================

    def _connect_vector_store(self) -> None:
        """
        Connect to the persistent vector store.
        """

        if self.vector_store is not None:
            return

        db_path = (
            self.project_root
            / self.config.vectorstore_path
        )

        self.vector_store_manager = VectorStore(
            db_path=db_path,
            collection_name=self.config.collection_name,
        )

        self.vector_store = (
            self.vector_store_manager.connect()
        )

    # =========================================================
    # Index
    # =========================================================

    def _load_or_build_index(self) -> None:
        """
        Load an existing index or build a new one.
        """

        if self.has_index():
            self._load_index()
        else:
            self.build_index()

    def _load_index(self) -> None:
        """
        Load an existing vector index.
        """

        self._create_embedding_model()

        loader = IndexLoader(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

        self.index = loader.load()

    # =========================================================
    # Embeddings
    # =========================================================

    def _create_embedding_model(self) -> None:
        """
        Initialize the embedding model once.
        """

        if self.embedding_model is not None:
            return

        self.embedding_model = EmbeddingFactory.create(
            self.config.embedding_model,
        )

    # =========================================================
    # Retrieval
    # =========================================================

    def _create_retriever(self) -> None:
        """
        Create the retriever from the initialized index.
        """

        if self.retriever is not None:
            return

        if self.index is None:
            raise RuntimeError(
                "Index must be initialized before "
                "creating the retriever."
            )

        self.retriever = Retriever(
            self.index,
            similarity_top_k=self.config.similarity_top_k,
        )

    # =========================================================
    # LLM
    # =========================================================

    def _create_llm(self) -> None:
        """
        Create the configured LLM.
        """

        if self.llm is not None:
            return

        self.llm = LLMFactory.create(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    # =========================================================
    # Knowledge base construction
    # =========================================================

    def _load_documents(self):
        """
        Load the source documents.
        """

        document_path = (
            self.project_root
            / self.config.documents_path
        )

        if not document_path.exists():
            raise FileNotFoundError(
                f"Document source not found: {document_path}"
            )

        return DocumentLoader.load(
            document_path,
        )

    def _chunk_documents(self, documents):
        """
        Split documents into retrieval nodes.
        """

        if not documents:
            raise RuntimeError(
                "No documents were loaded."
            )

        chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        return chunker.split(
            documents,
        )

    # =========================================================
    # Vector store management
    # =========================================================

    def _get_vector_count(self) -> int:
        """
        Return the number of vectors in the collection.
        """

        if self.vector_store is None:
            raise RuntimeError(
                "Vector store has not been initialized."
            )

        return self.vector_store._collection.count()

    def _clear_vector_store(self) -> None:
        """
        Clear the existing vector store.
        """

        if self.vector_store_manager is None:
            raise RuntimeError(
                "Vector store has not been initialized."
            )

        if not hasattr(
            self.vector_store_manager,
            "clear",
        ):
            raise NotImplementedError(
                "VectorStore.clear() must be implemented "
                "before rebuilding the index."
            )

        self.vector_store_manager.clear()

        self.vector_store = None
        self.index = None
        self.retriever = None

        # Reconnect to the newly cleared vector store.
        self._connect_vector_store()