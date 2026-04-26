import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from rag import chunk_text, embed, embed_batch, VectorStore, generate_answer

store = VectorStore()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
TOP_K = int(os.getenv("TOP_K", 5))


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    store.clear()


app = FastAPI(
    title="RAG Demo",
    description="Retrieval-Augmented Generation demo built by Cylunor (https://cylunor.com)",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class StatusResponse(BaseModel):
    chunks_indexed: int
    ready: bool


@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return StatusResponse(chunks_indexed=store.size, ready=store.size > 0)


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or unreadable")

    chunks = chunk_text(text, source=file.filename, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    embeddings = embed_batch([c.text for c in chunks])
    store.add(chunks, embeddings)

    return {"filename": file.filename, "chunks_created": len(chunks), "total_chunks": store.size}


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if store.size == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload documents first.")

    query_embedding = embed(req.question)
    results = store.search(query_embedding, top_k=req.top_k)

    answer = generate_answer(req.question, results)
    sources = list({chunk.source for chunk, _ in results})

    return QueryResponse(answer=answer, sources=sources, chunks_used=len(results))


@app.delete("/api/documents")
async def clear_documents():
    store.clear()
    return {"message": "All documents cleared", "chunks_indexed": 0}
