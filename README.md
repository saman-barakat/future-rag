# FUTURE-RAG

**Retrieval-Augmented Generation system for the FUTURE-AI guideline.**

FUTURE-RAG is a modular RAG system that uses the **FUTURE-AI guideline** as its knowledge base. It combines semantic retrieval with a large language model to generate answers grounded in the retrieved content.

The project is designed with a modular architecture so that components such as the embedding model, vector database, retriever, and LLM can be easily replaced or extended.

---

## Features

* 📄 Document ingestion and processing
* ✂️ Configurable document chunking
* 🔎 Semantic similarity retrieval
* 🧠 Hugging Face embeddings
* 🗄️ Persistent ChromaDB vector store
* 🤖 DeepSeek LLM integration
* 🔗 LlamaIndex RAG pipeline
* 🚀 FastAPI REST API
* 🧪 Unit testing with pytest
* ⚙️ Configuration-driven architecture
* 🧩 Modular RAG pipeline design

---

## Architecture

```text
                    ┌──────────────┐
                    │    Client    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   FastAPI    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  RAGService  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   BasicRAG   │
                    └──────┬───────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
           ┌─────────────┐   ┌─────────────┐
           │  Retriever  │   │     LLM     │
           └──────┬──────┘   └─────────────┘
                  │
                  ▼
           ┌─────────────┐
           │  ChromaDB   │
           └──────┬──────┘
                  │
                  ▼
           ┌─────────────┐
           │  Embeddings │
           └─────────────┘
```

### Knowledge Base Pipeline

```text
FUTURE-AI Document
       │
       ▼
Document Loader
       │
       ▼
Chunking
       │
       ▼
Embeddings
       │
       ▼
ChromaDB
```

---

## Project Structure

```text
future-rag/
│
├── data/
│   ├── knowledge/
│   ├── chunks/
│   └── vectorstores/
│
├── src/
│   └── future_rag/
│       ├── api/
│       ├── core/
│       ├── ingestion/
│       ├── indexing/
│       ├── knowledge/
│       ├── pipelines/
│       ├── services/
│       └── main.py
│
├── tests/
│   ├── pipelines/
│   ├── services/
│   └── ...
│
├── .env
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Component       | Technology             |
| --------------- | ---------------------- |
| Language        | Python                 |
| RAG Framework   | LlamaIndex             |
| Vector Database | ChromaDB               |
| Embeddings      | BAAI/bge-small-en-v1.5 |
| LLM             | DeepSeek               |
| API             | FastAPI                |
| Server          | Uvicorn                |
| Testing         | pytest                 |

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd future-rag
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the project as an editable package:

```bash
pip install -e .
```

---

## Configuration

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

The main RAG configuration includes parameters such as:

```python
RAGConfig(
    documents_path="./data/knowledge/pdf",
    chunk_size=1024,
    chunk_overlap=256,
    embedding_model="BAAI/bge-small-en-v1.5",
    vectorstore_path="./data/vectorstores/chroma",
    collection_name="future_rag",
    similarity_top_k=3,
    llm_provider="deepseek",
    llm_model="deepseek-chat",
    temperature=0.1,
    max_tokens=4096,
)
```

These parameters allow different RAG configurations to be tested without changing the pipeline implementation.

---

## Running the Application

Start the FastAPI server with:

```bash
uvicorn future_rag.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI's interactive documentation is available at:

```text
http://localhost:8000/docs
```

---

## API Usage

### Query the RAG system

**Endpoint**

```text
POST /rag/query
```

**Request**

```json
{
  "question": "What are the main principles of trustworthy AI?"
}
```

**Response**

```json
{
  "answer": "The FUTURE-AI guideline identifies several principles..."
}
```

The request is processed through:

```text
API
 ↓
RAGService
 ↓
BasicRAG
 ↓
Retriever
 ↓
ChromaDB
 ↓
Relevant Context
 ↓
DeepSeek
 ↓
Answer
```

---

## Testing

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/pipelines/test_basic_rag.py -v
```

For asynchronous tests, install:

```bash
pip install pytest-asyncio
```

and configure:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## RAG Configuration Experiments

The system supports experimenting with different chunking configurations.

For example:

```text
Chunk Size
├── 256
├── 512
├── 1024
└── 2048
```

Different configurations can use separate ChromaDB collections, making it possible to compare retrieval and answer quality.

Future experiments will include:

* Hybrid retrieval
* Reranking
* Query rewriting
* Advanced RAG pipelines
* RAGAS evaluation

---

## Roadmap

* [x] Document ingestion
* [x] Document chunking
* [x] Embedding generation
* [x] ChromaDB integration
* [x] Semantic retrieval
* [x] Basic RAG pipeline
* [x] DeepSeek integration
* [x] FastAPI API
* [x] Unit testing
* [ ] Advanced RAG
* [ ] Hybrid retrieval
* [ ] Reranking
* [ ] RAGAS evaluation
* [ ] Conversation history
* [ ] Streaming responses

---

## License

This project is currently intended for research and educational purposes.

---

## Project Status

**Active Development**

FUTURE-RAG is being developed as a modular RAG research and engineering project focused on improving retrieval and generation over the FUTURE-AI guideline.
