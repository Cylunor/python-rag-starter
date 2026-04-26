from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    index: int
    source: str


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    index = 0
    i = 0

    while i < len(words):
        slice_words = words[i : i + chunk_size]
        chunk_text = " ".join(slice_words).strip()
        if chunk_text:
            chunks.append(Chunk(id=f"{source}-{index}", text=chunk_text, index=index, source=source))
            index += 1
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap

    return chunks
