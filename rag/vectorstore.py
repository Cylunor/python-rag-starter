import numpy as np
from dataclasses import dataclass, field
from rag.chunker import Chunk


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.embeddings:
            return []

        q = np.array(query_embedding)
        matrix = np.array(self.embeddings)

        q_norm = q / (np.linalg.norm(q) + 1e-10)
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        scores = m_norm @ q_norm

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings.clear()

    @property
    def size(self) -> int:
        return len(self.chunks)
