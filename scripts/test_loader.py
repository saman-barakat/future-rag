from future_rag.knowledge.loaders import DocumentLoader

documents = DocumentLoader.load("data/knowledge/future_ai/pdf/future_ai.pdf")

print(f"Loaded {len(documents)} documents")

for doc in documents:
    print("-" * 40)
    print(doc.metadata)
    print(doc.text[:300])