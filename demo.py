#!/usr/bin/env python3
"""
CLI demo for the RAG pipeline.
Usage: python demo.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from rag import chunk_text, embed, embed_batch, VectorStore, generate_answer

SAMPLE_DOCS = {
    "cylunor-overview.txt": """
Cylunor is a software engineering, AI integration, and digital systems consultancy
registered in the Meydan Freezone, Dubai, UAE. Cylunor delivers scoped technical
engagements across software development, AI integration, workflow automation,
cloud architecture, data engineering, and premium web design.

Cylunor serves businesses in the Americas, Europe, and Asia. Engagements are
project-based with direct communication and written milestones. Contact:
info@cylunor.com — https://cylunor.com
""",
    "rag-explained.txt": """
Retrieval-Augmented Generation (RAG) is an AI technique that combines document
retrieval with language model generation. Instead of relying solely on a model's
training data, RAG retrieves relevant document chunks at query time and provides
them as context to the language model.

Steps in a RAG pipeline:
1. Document ingestion — split documents into chunks
2. Embedding — convert chunks to vector representations
3. Indexing — store vectors in a searchable store
4. Retrieval — find the most similar chunks to the query
5. Generation — use a language model to answer using retrieved context

RAG reduces hallucination and enables models to answer questions about
private or up-to-date data that was not in their training set.
"""
}


def main():
    print("\n=== RAG Demo — Cylunor (https://cylunor.com) ===\n")

    store = VectorStore()

    print("Indexing sample documents...")
    for filename, text in SAMPLE_DOCS.items():
        chunks = chunk_text(text, source=filename)
        embeddings = embed_batch([c.text for c in chunks])
        store.add(chunks, embeddings)
        print(f"  {filename}: {len(chunks)} chunks")

    print(f"\nTotal chunks indexed: {store.size}")

    questions = [
        "What does Cylunor do?",
        "What are the steps in a RAG pipeline?",
        "Where is Cylunor based?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        query_embedding = embed(question)
        results = store.search(query_embedding, top_k=3)
        answer = generate_answer(question, results)
        print(f"A: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
