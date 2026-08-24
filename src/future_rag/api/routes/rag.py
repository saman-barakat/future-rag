from fastapi import APIRouter, Request

from future_rag.api.schemas import (
    QueryRequest,
    QueryResponse,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: Request,
    data: QueryRequest,
) -> QueryResponse:

    rag_service = request.app.state.rag_service

    answer = rag_service.query(
        data.question,
    )

    return QueryResponse(
        answer=answer,
    )