"""
Unit tests for the ApplicationContainer class (FUTURE-RAG).

These tests isolate the ApplicationContainer by mocking all external
dependencies, including the vector store, index builder/loader,
embedding factory, document loader, chunker, retriever, and LLM factory.

No real database, embedding model, document, or LLM API is used.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from future_rag.application.container import ApplicationContainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Provides a mock application configuration."""
    config = MagicMock()

    config.vectorstore_path = "data/vectorstores/chroma"
    config.collection_name = "future_rag"

    config.embedding_model = "BAAI/bge-small-en-v1.5"

    config.similarity_top_k = 5

    config.chunk_size = 1024
    config.chunk_overlap = 128
    config.documents_path = "data/knowledge/pdf"

    config.llm_provider = "deepseek"
    config.llm_model = "deepseek-v4-flash"
    config.temperature = 0.2
    config.max_tokens = 4096

    return config


@pytest.fixture
def project_root(tmp_path):
    """Provides a temporary project root."""
    return tmp_path


@pytest.fixture
def container(mock_config, project_root):
    """Provides an ApplicationContainer instance."""
    return ApplicationContainer(
        config=mock_config,
        project_root=project_root,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestApplicationContainerInitialization:

    def test_init_sets_default_state(
        self,
        container,
        mock_config,
        project_root,
    ):
        """Verify that the container starts in an uninitialized state."""
        assert container.config == mock_config
        assert container.project_root == project_root

        assert container.vector_store_manager is None
        assert container.vector_store is None

        assert container.embedding_model is None
        assert container.index is None
        assert container.retriever is None
        assert container.llm is None

        assert container.is_initialized is False

    @patch("future_rag.application.container.LLMFactory")
    @patch("future_rag.application.container.Retriever")
    @patch.object(ApplicationContainer, "_load_or_build_index")
    @patch.object(ApplicationContainer, "_connect_vector_store")
    def test_initialize_creates_dependencies(
        self,
        mock_connect,
        mock_load_or_build,
        mock_retriever,
        mock_llm_factory,
        container,
    ):
        """Verify that initialize() creates the required dependencies."""
        # Arrange
        mock_index = MagicMock(name="mock_index")
        container.index = mock_index

        mock_retriever.return_value = MagicMock(
            name="mock_retriever"
        )

        mock_llm_factory.create.return_value = MagicMock(
            name="mock_llm"
        )

        # Act
        container.initialize()

        # Assert
        mock_connect.assert_called_once_with()
        mock_load_or_build.assert_called_once_with()

        mock_retriever.assert_called_once_with(
            mock_index,
            similarity_top_k=container.config.similarity_top_k,
        )

        mock_llm_factory.create.assert_called_once_with(
            provider=container.config.llm_provider,
            model=container.config.llm_model,
            temperature=container.config.temperature,
            max_tokens=container.config.max_tokens,
        )

        assert container.retriever == mock_retriever.return_value
        assert container.llm == mock_llm_factory.create.return_value
        assert container.is_initialized is True

    @patch.object(ApplicationContainer, "_connect_vector_store")
    @patch.object(ApplicationContainer, "_load_or_build_index")
    @patch.object(ApplicationContainer, "_create_retriever")
    @patch.object(ApplicationContainer, "_create_llm")
    def test_initialize_only_runs_once(
        self,
        mock_create_llm,
        mock_create_retriever,
        mock_load_or_build,
        mock_connect,
        container,
    ):
        """Verify that initialize() is idempotent."""
        # Act
        container.initialize()
        container.initialize()

        # Assert
        mock_connect.assert_called_once_with()
        mock_load_or_build.assert_called_once_with()
        mock_create_retriever.assert_called_once_with()
        mock_create_llm.assert_called_once_with()

        assert container.is_initialized is True


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------


class TestApplicationContainerVectorStore:

    @patch("future_rag.application.container.VectorStore")
    def test_connect_vector_store(
        self,
        mock_vector_store,
        container,
    ):
        """Verify that the vector store is created and connected."""
        # Arrange
        mock_manager = MagicMock(
            name="mock_vector_store_manager"
        )
        mock_store = MagicMock(name="mock_vector_store")

        mock_vector_store.return_value = mock_manager
        mock_manager.connect.return_value = mock_store

        expected_path = (
            container.project_root
            / container.config.vectorstore_path
        )

        # Act
        container._connect_vector_store()

        # Assert
        mock_vector_store.assert_called_once_with(
            db_path=expected_path,
            collection_name=container.config.collection_name,
        )

        mock_manager.connect.assert_called_once_with()

        assert container.vector_store_manager == mock_manager
        assert container.vector_store == mock_store

    @patch("future_rag.application.container.VectorStore")
    def test_connect_vector_store_only_once(
        self,
        mock_vector_store,
        container,
    ):
        """Verify that an existing vector store is not recreated."""
        # Arrange
        existing_store = MagicMock(
            name="existing_vector_store"
        )
        container.vector_store = existing_store

        # Act
        container._connect_vector_store()

        # Assert
        mock_vector_store.assert_not_called()
        assert container.vector_store == existing_store

    def test_get_vector_count(
        self,
        container,
    ):
        """Verify that _get_vector_count returns the collection count."""
        # Arrange
        mock_collection = MagicMock()
        mock_collection.count.return_value = 25

        mock_vector_store = MagicMock()
        mock_vector_store._collection = mock_collection

        container.vector_store = mock_vector_store

        # Act
        count = container._get_vector_count()

        # Assert
        mock_collection.count.assert_called_once_with()
        assert count == 25

    def test_get_vector_count_without_vector_store(
        self,
        container,
    ):
        """Verify that vector count requires an initialized vector store."""
        with pytest.raises(
            RuntimeError,
            match="Vector store has not been initialized",
        ):
            container._get_vector_count()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class TestApplicationContainerIndex:

    @patch.object(ApplicationContainer, "_load_index")
    @patch.object(ApplicationContainer, "has_index")
    def test_load_or_build_existing_index(
        self,
        mock_has_index,
        mock_load_index,
        container,
    ):
        """Verify that an existing index is loaded."""
        # Arrange
        mock_has_index.return_value = True

        # Act
        container._load_or_build_index()

        # Assert
        mock_has_index.assert_called_once_with()
        mock_load_index.assert_called_once_with()

    @patch.object(ApplicationContainer, "build_index")
    @patch.object(ApplicationContainer, "has_index")
    def test_load_or_build_creates_index_when_missing(
        self,
        mock_has_index,
        mock_build_index,
        container,
    ):
        """Verify that an index is built when none exists."""
        # Arrange
        mock_has_index.return_value = False

        # Act
        container._load_or_build_index()

        # Assert
        mock_has_index.assert_called_once_with()
        mock_build_index.assert_called_once_with()

    @patch("future_rag.application.container.IndexLoader")
    @patch("future_rag.application.container.EmbeddingFactory")
    def test_load_index(
        self,
        mock_embedding_factory,
        mock_index_loader,
        container,
    ):
        """Verify that _load_index creates an embedding model and loads the index."""
        # Arrange
        mock_embedding = MagicMock(
            name="mock_embedding"
        )
        mock_index = MagicMock(
            name="mock_index"
        )
        mock_loader = MagicMock(
            name="mock_index_loader"
        )

        mock_embedding_factory.create.return_value = (
            mock_embedding
        )

        mock_index_loader.return_value = mock_loader
        mock_loader.load.return_value = mock_index

        container.vector_store = MagicMock()

        # Act
        container._load_index()

        # Assert
        mock_embedding_factory.create.assert_called_once_with(
            container.config.embedding_model
        )

        mock_index_loader.assert_called_once_with(
            embedding_model=mock_embedding,
            vector_store=container.vector_store,
        )

        mock_loader.load.assert_called_once_with()

        assert container.embedding_model == mock_embedding
        assert container.index == mock_index


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestApplicationContainerEmbeddings:

    @patch("future_rag.application.container.EmbeddingFactory")
    def test_create_embedding_model(
        self,
        mock_embedding_factory,
        container,
    ):
        """Verify that the embedding model is created using the factory."""
        # Arrange
        mock_embedding = MagicMock(
            name="mock_embedding"
        )

        mock_embedding_factory.create.return_value = (
            mock_embedding
        )

        # Act
        container._create_embedding_model()

        # Assert
        mock_embedding_factory.create.assert_called_once_with(
            container.config.embedding_model
        )

        assert container.embedding_model == mock_embedding

    @patch("future_rag.application.container.EmbeddingFactory")
    def test_create_embedding_model_only_once(
        self,
        mock_embedding_factory,
        container,
    ):
        """Verify that an existing embedding model is not recreated."""
        # Arrange
        existing_embedding = MagicMock(
            name="existing_embedding"
        )

        container.embedding_model = existing_embedding

        # Act
        container._create_embedding_model()

        # Assert
        mock_embedding_factory.create.assert_not_called()
        assert container.embedding_model == existing_embedding


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TestApplicationContainerRetriever:

    @patch("future_rag.application.container.Retriever")
    def test_create_retriever(
        self,
        mock_retriever,
        container,
    ):
        """Verify that the retriever is created from the index."""
        # Arrange
        mock_index = MagicMock(name="mock_index")
        mock_retriever_instance = MagicMock(
            name="mock_retriever"
        )

        container.index = mock_index
        mock_retriever.return_value = mock_retriever_instance

        # Act
        container._create_retriever()

        # Assert
        mock_retriever.assert_called_once_with(
            mock_index,
            similarity_top_k=container.config.similarity_top_k,
        )

        assert container.retriever == mock_retriever_instance

    @patch("future_rag.application.container.Retriever")
    def test_create_retriever_only_once(
        self,
        mock_retriever,
        container,
    ):
        """Verify that an existing retriever is not recreated."""
        # Arrange
        existing_retriever = MagicMock(
            name="existing_retriever"
        )

        container.retriever = existing_retriever

        # Act
        container._create_retriever()

        # Assert
        mock_retriever.assert_not_called()
        assert container.retriever == existing_retriever

    def test_create_retriever_without_index(
        self,
        container,
    ):
        """Verify that retriever creation requires an initialized index."""
        with pytest.raises(
            RuntimeError,
            match="Index must be initialized",
        ):
            container._create_retriever()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class TestApplicationContainerLLM:

    @patch("future_rag.application.container.LLMFactory")
    def test_create_llm(
        self,
        mock_llm_factory,
        container,
    ):
        """Verify that the configured LLM is created through the factory."""
        # Arrange
        mock_llm = MagicMock(name="mock_llm")

        mock_llm_factory.create.return_value = mock_llm

        # Act
        container._create_llm()

        # Assert
        mock_llm_factory.create.assert_called_once_with(
            provider=container.config.llm_provider,
            model=container.config.llm_model,
            temperature=container.config.temperature,
            max_tokens=container.config.max_tokens,
        )

        assert container.llm == mock_llm

    @patch("future_rag.application.container.LLMFactory")
    def test_create_llm_only_once(
        self,
        mock_llm_factory,
        container,
    ):
        """Verify that an existing LLM is not recreated."""
        # Arrange
        existing_llm = MagicMock(
            name="existing_llm"
        )

        container.llm = existing_llm

        # Act
        container._create_llm()

        # Assert
        mock_llm_factory.create.assert_not_called()
        assert container.llm == existing_llm


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------


class TestApplicationContainerDocuments:

    def test_load_documents_missing_path(
        self,
        container,
    ):
        """Verify that a missing document path raises FileNotFoundError."""
        # The temporary project root does not contain the configured
        # documents directory.

        with pytest.raises(
            FileNotFoundError,
            match="Document source not found",
        ):
            container._load_documents()

    @patch("future_rag.application.container.DocumentLoader")
    def test_load_documents(
        self,
        mock_document_loader,
        container,
    ):
        """Verify that documents are loaded from the configured path."""
        # Arrange
        document_path = (
            container.project_root
            / container.config.documents_path
        )

        document_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        expected_documents = [
            MagicMock(name="document_1"),
            MagicMock(name="document_2"),
        ]

        mock_document_loader.load.return_value = (
            expected_documents
        )

        # Act
        documents = container._load_documents()

        # Assert
        mock_document_loader.load.assert_called_once_with(
            document_path,
        )

        assert documents == expected_documents

    @patch("future_rag.application.container.DocumentChunker")
    def test_chunk_documents(
        self,
        mock_chunker,
        container,
    ):
        """Verify that documents are chunked using the configured parameters."""
        # Arrange
        documents = [
            MagicMock(name="document_1"),
        ]

        expected_nodes = [
            MagicMock(name="node_1"),
            MagicMock(name="node_2"),
        ]

        mock_chunker_instance = MagicMock(
            name="mock_chunker"
        )

        mock_chunker.return_value = (
            mock_chunker_instance
        )

        mock_chunker_instance.split.return_value = (
            expected_nodes
        )

        # Act
        nodes = container._chunk_documents(
            documents
        )

        # Assert
        mock_chunker.assert_called_once_with(
            chunk_size=container.config.chunk_size,
            chunk_overlap=container.config.chunk_overlap,
        )

        mock_chunker_instance.split.assert_called_once_with(
            documents,
        )

        assert nodes == expected_nodes

    def test_chunk_documents_empty(
        self,
        container,
    ):
        """Verify that empty documents raise RuntimeError."""
        with pytest.raises(
            RuntimeError,
            match="No documents were loaded",
        ):
            container._chunk_documents([])


# ---------------------------------------------------------------------------
# Build Index
# ---------------------------------------------------------------------------


class TestApplicationContainerBuildIndex:

    @patch("future_rag.application.container.IndexBuilder")
    @patch.object(ApplicationContainer, "_create_embedding_model")
    @patch.object(ApplicationContainer, "_chunk_documents")
    @patch.object(ApplicationContainer, "_load_documents")
    @patch.object(ApplicationContainer, "has_index")
    @patch.object(ApplicationContainer, "_connect_vector_store")
    def test_build_index(
        self,
        mock_connect,
        mock_has_index,
        mock_load_documents,
        mock_chunk_documents,
        mock_create_embedding,
        mock_index_builder,
        container,
    ):
        """Verify that build_index constructs and stores a new index."""
        # Arrange
        mock_has_index.return_value = False

        documents = [
            MagicMock(name="document")
        ]

        nodes = [
            MagicMock(name="node")
        ]

        mock_index = MagicMock(name="mock_index")
        mock_builder = MagicMock(name="mock_builder")

        mock_load_documents.return_value = documents
        mock_chunk_documents.return_value = nodes
        mock_index_builder.return_value = mock_builder
        mock_builder.build.return_value = mock_index

        container.embedding_model = MagicMock(
            name="embedding_model"
        )

        container.vector_store = MagicMock(
            name="vector_store"
        )

        # Act
        result = container.build_index()

        # Assert
        mock_connect.assert_called_once_with()
        mock_has_index.assert_called_once_with()

        mock_load_documents.assert_called_once_with()
        mock_chunk_documents.assert_called_once_with(
            documents
        )

        mock_create_embedding.assert_called_once_with()

        mock_index_builder.assert_called_once_with(
            embedding_model=container.embedding_model,
            vector_store=container.vector_store,
        )

        mock_builder.build.assert_called_once_with(
            nodes=nodes
        )

        assert result == mock_index
        assert container.index == mock_index

    @patch.object(ApplicationContainer, "has_index")
    @patch.object(ApplicationContainer, "_connect_vector_store")
    def test_build_index_when_index_exists(
        self,
        mock_connect,
        mock_has_index,
        container,
    ):
        """Verify that build_index rejects an existing index."""
        # Arrange
        mock_has_index.return_value = True

        # Act & Assert
        with pytest.raises(
            RuntimeError,
            match="A vector index already exists",
        ):
            container.build_index()

        mock_connect.assert_called_once_with()
        mock_has_index.assert_called_once_with()


# ---------------------------------------------------------------------------
# Rebuild Index
# ---------------------------------------------------------------------------


class TestApplicationContainerRebuildIndex:

    @patch("future_rag.application.container.IndexBuilder")
    @patch.object(ApplicationContainer, "_create_embedding_model")
    @patch.object(ApplicationContainer, "_chunk_documents")
    @patch.object(ApplicationContainer, "_load_documents")
    @patch.object(ApplicationContainer, "_clear_vector_store")
    @patch.object(ApplicationContainer, "_connect_vector_store")
    def test_rebuild_index(
        self,
        mock_connect,
        mock_clear,
        mock_load_documents,
        mock_chunk_documents,
        mock_create_embedding,
        mock_index_builder,
        container,
    ):
        """Verify that rebuild_index clears and rebuilds the index."""
        # Arrange
        documents = [
            MagicMock(name="document")
        ]

        nodes = [
            MagicMock(name="node")
        ]

        mock_index = MagicMock(name="new_index")
        mock_builder = MagicMock(name="mock_builder")

        mock_load_documents.return_value = documents
        mock_chunk_documents.return_value = nodes

        mock_index_builder.return_value = mock_builder
        mock_builder.build.return_value = mock_index

        container.embedding_model = MagicMock(
            name="embedding_model"
        )

        container.vector_store = MagicMock(
            name="vector_store"
        )

        # Act
        result = container.rebuild_index()

        # Assert
        mock_connect.assert_called_once_with()
        mock_clear.assert_called_once_with()

        mock_load_documents.assert_called_once_with()
        mock_chunk_documents.assert_called_once_with(
            documents
        )

        mock_create_embedding.assert_called_once_with()

        mock_index_builder.assert_called_once_with(
            embedding_model=container.embedding_model,
            vector_store=container.vector_store,
        )

        mock_builder.build.assert_called_once_with(
            nodes=nodes
        )

        assert result == mock_index
        assert container.index == mock_index
        assert container.retriever is None


# ---------------------------------------------------------------------------
# Clear Vector Store
# ---------------------------------------------------------------------------


class TestApplicationContainerClearVectorStore:

    def test_clear_vector_store(
        self,
        container,
    ):
        """Verify that the vector store is cleared and state is reset."""
        # Arrange
        mock_manager = MagicMock(
            name="mock_vector_store_manager"
        )

        container.vector_store_manager = mock_manager
        container.vector_store = MagicMock(
            name="vector_store"
        )
        container.index = MagicMock(
            name="index"
        )
        container.retriever = MagicMock(
            name="retriever"
        )

        # Patch reconnect to avoid creating a real vector store.
        with patch.object(
            container,
            "_connect_vector_store",
        ) as mock_connect:

            # Act
            container._clear_vector_store()

        # Assert
        mock_manager.clear.assert_called_once_with()

        assert container.vector_store is None
        assert container.index is None
        assert container.retriever is None

        mock_connect.assert_called_once_with()

    def test_clear_vector_store_without_manager(
        self,
        container,
    ):
        """Verify that clearing requires an initialized vector store manager."""
        with pytest.raises(
            RuntimeError,
            match="Vector store has not been initialized",
        ):
            container._clear_vector_store()

    def test_clear_vector_store_without_clear_method(
        self,
        container,
    ):
        """Verify that a vector store manager must implement clear()."""
        manager = MagicMock(
            name="vector_store_manager"
        )

        # MagicMock normally reports arbitrary attributes as existing.
        # Explicitly remove clear to simulate an unsupported manager.
        del manager.clear

        container.vector_store_manager = manager

        with pytest.raises(
            NotImplementedError,
            match="VectorStore.clear\\(\\) must be implemented",
        ):
            container._clear_vector_store()


# ---------------------------------------------------------------------------
# Has Index
# ---------------------------------------------------------------------------


class TestApplicationContainerHasIndex:

    @patch.object(ApplicationContainer, "_get_vector_count")
    @patch.object(ApplicationContainer, "_connect_vector_store")
    def test_has_index_when_vectors_exist(
        self,
        mock_connect,
        mock_get_count,
        container,
    ):
        """Verify that has_index() returns True when vectors exist."""
        # Arrange
        mock_get_count.return_value = 10

        # Act
        result = container.has_index()

        # Assert
        mock_connect.assert_called_once_with()
        mock_get_count.assert_called_once_with()

        assert result is True

    @patch.object(ApplicationContainer, "_get_vector_count")
    @patch.object(ApplicationContainer, "_connect_vector_store")
    def test_has_index_when_no_vectors_exist(
        self,
        mock_connect,
        mock_get_count,
        container,
    ):
        """Verify that has_index() returns False when no vectors exist."""
        # Arrange
        mock_get_count.return_value = 0

        # Act
        result = container.has_index()

        # Assert
        mock_connect.assert_called_once_with()
        mock_get_count.assert_called_once_with()

        assert result is False