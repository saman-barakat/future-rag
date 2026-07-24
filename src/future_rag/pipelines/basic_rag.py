"""
Basic Retrieval-Augmented Generation (RAG) pipeline.
"""

from typing import Optional

from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from future_rag.retrieval.retriever import Retriever


DEFAULT_QA_TEMPLATE = PromptTemplate(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer is not contained in the context, reply:
"I could not find the answer in the provided documents."

---------------------
Context
---------------------
{context_str}

---------------------
Question
---------------------
{query_str}

---------------------
Answer
---------------------
"""
)


class BasicRAG:
    """
    Basic Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        qa_template: Optional[PromptTemplate] = None,
    ):
        """
        Initialize the RAG pipeline.
        """
        self.retriever = retriever
        self.llm = llm

        synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=qa_template or DEFAULT_QA_TEMPLATE,
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=synthesizer,
        )

    def query(self, question: str):
        """
        Ask a question.
        """
        return self.query_engine.query(question)

    async def aquery(self, question: str):
        """
        Ask a question asynchronously.
        """
        return await self.query_engine.aquery(question)

    def retrieve(self, question: str):
        """
        Retrieve relevant nodes only.
        """
        return self.retriever.retrieve(question)