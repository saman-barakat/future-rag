from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from future_rag.indexing.vectorstore import VectorStore

# Target module path for unittest.mock patching
VECTORSTORE_MODULE = "future_rag.indexing.vectorstore"


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provides a temporary path for the ChromaDB directory."""
    return tmp_path / "chroma_test_db"


@pytest.fixture
def store(db_path: Path) -> VectorStore:
    """Provides an instance of VectorStore configured with the temporary path."""
    return VectorStore(db_path=db_path, collection_name="test_collection")


@pytest.fixture
def mock_chromadb():
    """Patch the `chromadb` module inside future_rag.indexing.vectorstore."""
    with patch(f"{VECTORSTORE_MODULE}.chromadb") as mock_module:
        mock_client = MagicMock(name="mock_chroma_client")
        mock_collection = MagicMock(name="mock_chroma_collection")

        mock_client.get_or_create_collection.return_value = mock_collection
        mock_module.PersistentClient.return_value = mock_client

        yield mock_module, mock_client, mock_collection


@pytest.fixture
def mock_chroma_vector_store():
    """Patch `ChromaVectorStore` class inside future_rag.indexing.vectorstore."""
    with patch(f"{VECTORSTORE_MODULE}.ChromaVectorStore") as mock_cls:
        mock_instance = MagicMock(name="mock_llamaindex_vector_store")
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_vector_store_default_initialization():
    """Test default parameters for VectorStore initialization."""
    store = VectorStore()
    assert store.db_path == Path("./data/vectorstores/chroma")
    assert store.collection_name == "future_rag"


def test_vector_store_custom_initialization(db_path: Path):
    """Test custom parameters for VectorStore initialization."""
    store = VectorStore(db_path=db_path, collection_name="custom_collection")
    assert store.db_path == db_path
    assert store.collection_name == "custom_collection"


def test_connect(store: VectorStore, mock_chromadb, mock_chroma_vector_store):
    """Test that connect() creates directory, initializes client/collection, and returns ChromaVectorStore."""
    mock_module, mock_client, mock_collection = mock_chromadb
    mock_cls, mock_instance = mock_chroma_vector_store

    vector_store = store.connect()

    # 1. Check directory creation
    assert store.db_path.exists()

    # 2. Check PersistentClient creation with path
    mock_module.PersistentClient.assert_called_once_with(path=str(store.db_path))

    # 3. Check collection fetch/creation with collection_name
    mock_client.get_or_create_collection.assert_called_once_with(
        name="test_collection"
    )

    # 4. Check ChromaVectorStore wrapping
    mock_cls.assert_called_once_with(chroma_collection=mock_collection)

    # 5. Check returned instance
    assert vector_store == mock_instance