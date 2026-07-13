"""
FastAPI Application - DocuChat Production RAG v3
Fixed: preloaded embeddings, incremental indexing, no slow restarts
"""
import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.insert(0, ".")

from src.ingestion.pipeline import DocumentIngestionPipeline
from src.retrieval.retriever import HybridRetriever
from src.guardrails.guards import InputGuardrails, OutputGuardrails, GuardrailStatus
from src.gateway.llm_gateway import LLMGateway
from src.gateway.tracer import RAGTracer
from src.memory.memory import SlidingWindowMemory

# Global state
retriever = HybridRetriever()
ingestion = DocumentIngestionPipeline()
input_guards = InputGuardrails()
output_guards = OutputGuardrails()
gateway = LLMGateway()
tracer = RAGTracer()
memory = SlidingWindowMemory(window_size=6)
all_docs = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload embedding model at startup so first upload is fast
    print("Preloading embedding model...")
    _ = retriever.embeddings.embed_query("warmup")
    print("DocuChat Production RAG v3 ready")
    yield
    print("Shutting down...")


app = FastAPI(
    title="DocuChat AI",
    description="Production RAG with hybrid retrieval, guardrails, caching, tracing",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    use_memory: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    provider: str
    cached: bool
    guardrail_status: str
    trace_id: str


class UploadResponse(BaseModel):
    filename: str
    num_chunks: int
    num_pages: int
    status: str


@app.get("/")
def root():
    return {
        "message": "DocuChat AI - Production RAG",
        "version": "3.0.0",
        "documents_loaded": len(all_docs),
        "retriever_ready": retriever.vectorstore is not None,
        "features": [
            "Hybrid BM25 + FAISS with RRF",
            "Input guardrails (jailbreak, PII, sensitive topics)",
            "Output grounding validation",
            "Redis caching",
            "LLM fallback",
            "Langfuse tracing",
            "Sliding window memory",
        ],
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingestion.ingest(tmp_path)
        new_chunks = result.chunks
        all_docs.extend(new_chunks)

        if retriever.vectorstore is None:
            # First upload — build full index
            retriever.build(all_docs)
        else:
            # Subsequent uploads — incremental add (fast, no full rebuild)
            retriever.vectorstore.add_documents(new_chunks)
            tokenized = [doc.page_content.lower().split() for doc in all_docs]
            retriever.bm25 = BM25Okapi(tokenized)
            retriever.documents = all_docs

        os.unlink(tmp_path)
        return UploadResponse(
            filename=result.filename,
            num_chunks=result.num_chunks,
            num_pages=result.num_pages,
            status="indexed",
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not all_docs:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Use /upload first."
        )

    # Input guardrail
    guard_result = input_guards.check(req.query)
    if guard_result.status == GuardrailStatus.BLOCKED:
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked: {guard_result.reason}"
        )

    effective_query = guard_result.modified_query or req.query
    guardrail_status = guard_result.status.value

    # Retrieve
    try:
        docs = retriever.retrieve(effective_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # Build context with optional memory
    context = "\n\n".join([d.page_content for d in docs[:5]])
    if req.use_memory and len(memory) > 0:
        context = (
            f"Previous conversation:\n{memory.get_context()}"
            f"\n\n---\nDocument context:\n{context}"
        )

    # Generate
    try:
        response = gateway.generate(effective_query, context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Output guardrail
    out_result = output_guards.validate(response.text, docs)
    final_answer = out_result.modified_query or response.text

    # Update memory
    if req.use_memory:
        memory.add("user", effective_query)
        memory.add("assistant", response.text)

    # Trace
    trace_id = tracer.trace(
        query=effective_query,
        retrieved_docs=docs,
        response=response.text,
        latency_ms=response.latency_ms,
        provider=response.provider,
        cached=response.cached,
        guardrail_status=guardrail_status,
        tokens_used=response.tokens_used,
    )

    sources = [
        {
            "content": d.page_content[:200],
            "page": d.metadata.get("page", "?"),
            "filename": d.metadata.get("filename", "unknown"),
        }
        for d in docs[:3]
    ]

    return QueryResponse(
        answer=final_answer,
        sources=sources,
        latency_ms=response.latency_ms,
        provider=response.provider,
        cached=response.cached,
        guardrail_status=guardrail_status,
        trace_id=trace_id,
    )


@app.delete("/memory")
def clear_memory():
    memory.clear()
    return {"status": "memory cleared", "messages": 0}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_loaded": len(all_docs),
        "memory_messages": len(memory),
        "retriever_ready": retriever.vectorstore is not None,
    }


@app.get("/stats")
def stats():
    return {
        "documents_loaded": len(all_docs),
        "memory_window": memory.window_size,
        "current_memory_size": len(memory),
        "retriever_ready": retriever.vectorstore is not None,
    }