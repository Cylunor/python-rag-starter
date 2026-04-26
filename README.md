# nextjs-rag-starter

Production-ready **RAG (Retrieval-Augmented Generation)** implementation in Python.  
Upload documents, embed them with **OpenAI**, retrieve with cosine similarity, generate answers with **Anthropic Claude**.

Built by [Cylunor](https://cylunor.com) — Software, AI & Digital Systems Consultancy.

---

## Architecture

```
Document → Chunker → Embeddings (OpenAI) → VectorStore (NumPy)
                                                    ↓
Query → Embed → Cosine Search → Top-K Chunks → Claude → Answer
```

## Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | Anthropic Claude (`claude-sonnet-4-6`) |
| Vector search | NumPy cosine similarity |
| API | FastAPI |
| Runtime | Python 3.12+ |

---

## Quick start

```bash
git clone https://github.com/Cylunor/nextjs-rag-starter
cd nextjs-rag-starter
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add OPENAI_API_KEY and ANTHROPIC_API_KEY to .env
```

### Run CLI demo
```bash
python demo.py
```

### Run API server
```bash
uvicorn main:app --reload
# Open http://localhost:8000
```

---

## API Reference

### `POST /api/documents`
Upload a text file for indexing.
```bash
curl -X POST http://localhost:8000/api/documents \
  -F "file=@document.txt"
```

### `POST /api/query`
Ask a question against indexed documents.
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics?"}'
```

### `GET /api/status`
Check indexing status.
```bash
curl http://localhost:8000/api/status
```

### `DELETE /api/documents`
Clear all indexed documents.

---

## Project structure

```
rag/
├── __init__.py
├── chunker.py      # Split text into overlapping chunks
├── embeddings.py   # OpenAI embedding API
├── vectorstore.py  # In-memory cosine similarity search
└── generator.py    # Anthropic Claude answer generation

main.py             # FastAPI server
demo.py             # CLI demo script
static/index.html   # Simple web UI
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |

---

## License

MIT

---

Built by [Cylunor](https://cylunor.com) — a software engineering, AI integration, and digital systems consultancy based in Dubai.  
We help businesses build practical AI systems that integrate into real workflows.
