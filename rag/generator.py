from __future__ import annotations
import os
import anthropic
from rag.chunker import Chunk

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a precise document assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I could not find an answer in the provided documents."
Always cite the source document name when referencing information."""


def generate_answer(question: str, context_chunks: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[Source: {chunk.source}, chunk {chunk.index}]\n{chunk.text}"
        for chunk, _ in context_chunks
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.content[0].text
