from pathlib import Path

from future_rag.knowledge.loaders import DocumentLoader


TEST_DATA = Path(__file__).parent.parent / "data"


def test_load_pdf():
    """Test loading a PDF document."""

    documents = DocumentLoader.load(
        TEST_DATA / "future-ai.pdf"
    )

    assert len(documents) >= 1
    assert len(documents[0].text) > 0


def test_load_markdown():
    """Test loading a Markdown document."""

    documents = DocumentLoader.load(
        TEST_DATA / "sample.md"
    )

    assert len(documents) > 0
    #assert "Future-AI" in documents[0].text


def test_load_directory():
    """Test loading all supported documents in a directory."""

    documents = DocumentLoader.load(TEST_DATA)

    assert len(documents) >= 2


def test_invalid_path():
    """Loading a missing path should raise FileNotFoundError."""

    import pytest

    with pytest.raises(FileNotFoundError):
        DocumentLoader.load("missing_directory")