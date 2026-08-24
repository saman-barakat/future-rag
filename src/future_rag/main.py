from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI

from future_rag.api.routes.rag import router as rag_router
from future_rag.application.container import (
    ApplicationContainer,
)
from future_rag.config.settings import RAGConfig
from future_rag.pipelines.basic_rag import BasicRAG
from future_rag.services.rag_service import RAGService
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = RAGConfig()

    # ---------------------------------------------------------
    # Application container
    # ---------------------------------------------------------

    container = ApplicationContainer(
        config=config,
        project_root=PROJECT_ROOT,
    )

    container.initialize()

    # ---------------------------------------------------------
    # Basic RAG pipeline
    # ---------------------------------------------------------

    pipeline = BasicRAG(
        retriever=container.retriever,
        llm=container.llm,
    )

    # ---------------------------------------------------------
    # RAG service
    # ---------------------------------------------------------

    rag_service = RAGService(
        pipeline=pipeline,
    )

    # ---------------------------------------------------------
    # Application state
    # ---------------------------------------------------------

    app.state.container = container
    app.state.rag_service = rag_service

    yield

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------

    app.state.rag_service = None
    app.state.container = None


app = FastAPI(
    title="FUTURE-RAG API",
    description="Retrieval-Augmented Generation API",
    version="0.1.0",
    lifespan=lifespan,
)


app.include_router(
    rag_router,
)