class RAGService:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline and return the generated answer.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        response = self.pipeline.query(question)

        return response.response