"""
Langfuse Tracer
Full pipeline observability — query, retrieval, LLM call, latency
"""
import os
import uuid
from datetime import datetime

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


class RAGTracer:
    def __init__(self):
        self.enabled = False
        self.client = None

        if LANGFUSE_AVAILABLE:
            try:
                self.client = Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                self.enabled = True
                print("Langfuse tracing enabled")
            except Exception as e:
                print(f"Langfuse not available: {e}. Running without tracing.")

    def trace(
        self,
        query: str,
        retrieved_docs: list,
        response: str,
        latency_ms: float,
        provider: str,
        cached: bool,
        guardrail_status: str,
        tokens_used: int = 0,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]

        # Always log to console
        print(
            f"[TRACE {run_id}] "
            f"query_len={len(query)} | "
            f"docs={len(retrieved_docs)} | "
            f"latency={latency_ms:.0f}ms | "
            f"provider={provider} | "
            f"cached={cached} | "
            f"guardrail={guardrail_status} | "
            f"tokens={tokens_used}"
        )

        if not self.enabled or not self.client:
            return run_id

        try:
            trace = self.client.trace(
                name="rag-query",
                id=run_id,
                input={"query": query},
                output={"response": response},
                metadata={
                    "num_docs_retrieved": len(retrieved_docs),
                    "latency_ms": latency_ms,
                    "provider": provider,
                    "cached": cached,
                    "guardrail_status": guardrail_status,
                    "tokens_used": tokens_used,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Span for retrieval
            trace.span(
                name="retrieval",
                input={"query": query},
                output={"num_docs": len(retrieved_docs)},
                metadata={"doc_lengths": [len(d.page_content) for d in retrieved_docs[:5]]},
            )

            # Span for LLM call
            trace.span(
                name="llm-generation",
                input={"context_length": sum(len(d.page_content) for d in retrieved_docs)},
                output={"response_length": len(response)},
                metadata={"provider": provider, "latency_ms": latency_ms, "tokens": tokens_used},
            )

            self.client.flush()
        except Exception as e:
            print(f"Tracing error (non-fatal): {e}")

        return run_id