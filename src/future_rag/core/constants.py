"""
Global constants used throughout the FUTURE-RAG project.

This module centralizes common names, default values, and supported
options to avoid hardcoding strings across the codebase.
"""

from pathlib import Path

###############################################################################
# Project Information
###############################################################################

PROJECT_NAME = "FUTURE-RAG"
PROJECT_VERSION = "0.1.0"

###############################################################################
# Root Directories
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

###############################################################################
# Knowledge Directories
###############################################################################

KNOWLEDGE_DIR = DATA_DIR / "knowledge"

###############################################################################
# Vector Store Directories
###############################################################################

VECTORSTORE_DIR = DATA_DIR / "vectorstores"

CHROMA_DIR = VECTORSTORE_DIR / "chroma"
FAISS_DIR = VECTORSTORE_DIR / "faiss"
QDRANT_DIR = VECTORSTORE_DIR / "qdrant"
MILVUS_DIR = VECTORSTORE_DIR / "milvus"
NEO4J_DIR = VECTORSTORE_DIR / "neo4j"

###############################################################################
# Supported File Types
###############################################################################

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
    ".html",
}

###############################################################################
# Supported Vector Stores
###############################################################################

SUPPORTED_VECTOR_STORES = (
    "chroma",
    "faiss",
    "qdrant",
    "milvus",
    "neo4j",
)

###############################################################################
# Supported LLM Providers
###############################################################################

SUPPORTED_LLMS = (
    "openai",
    "gemini",
    "deepseek",
    "ollama",
)

###############################################################################
# Supported Embedding Providers
###############################################################################

SUPPORTED_EMBEDDINGS = (
    "openai",
    "huggingface",
    "ollama",
)

###############################################################################
# Default Values
###############################################################################

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5

###############################################################################
# Logging
###############################################################################

DEFAULT_LOG_LEVEL = "INFO"

###############################################################################
# Random Seed
###############################################################################

DEFAULT_RANDOM_SEED = 42