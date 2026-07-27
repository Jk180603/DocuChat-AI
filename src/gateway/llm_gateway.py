"""
LLM Gateway
Multi-provider with fallback, Redis caching, and latency tracking
"""
import hashlib
import time
import os
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class GatewayResponse:
    text: str
    latency_ms: float
    provider: str
    cached: bool
    tokens_used: int = 0


SYSTEM_PROMPT = """You are a helpful document assistant. 
Answer questions based ONLY on the provided context.
If the answer is not in the context, say so clearly.
Be concise, accurate, and cite relevant parts when helpful.
Never make up information not present in the context."""


class LLMGateway:
    def __init__(self):
        self.primary_model = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.cache_ttl = 3600  # 1 hour

        # Init primary LLM
        self.primary = ChatGroq(
            model=self.primary_model,
            groq_api_key=self.groq_key,
            temperature=0.1,
            max_tokens=1024,
        )

        # Init fallback (smaller model)
        self.fallback = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=self.groq_key,
            temperature=0.1,
            max_tokens=1024,
        )

        # Init Redis cache
        self.cache = None
        if REDIS_AVAILABLE:
            try:
                self.cache = redis.from_url(self.redis_url, decode_responses=True)
                self.cache.ping()
                print("Redis cache connected")
            except Exception:
                print("Redis not available, running without cache")
                self.cache = None

    def _make_cache_key(self, query: str, context: str) -> str: # cache glag as true for que i have to update it 
        content = f"{query}::{context[:500]}"
        return f"docuchat:{hashlib.md5(content.encode()).hexdigest()}"

    def _call_llm(self, llm, query: str, context: str, provider: str) -> tuple[str, int]: #what this fun revives str, int 
        messages = [
            SystemMessage(content=SYSTEM_PROMPT), # prompt flg or given to sys
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        response = llm.invoke(messages)
        text = response.content
        tokens = getattr(response, 'usage_metadata', {}) #token counting
        total_tokens = tokens.get('total_tokens', 0) if tokens else 0
        return text, total_tokens

    def generate(self, query: str, context: str) -> GatewayResponse:
        # Check cache first
        cache_key = self._make_cache_key(query, context)
        if self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    return GatewayResponse(
                        text=cached,
                        latency_ms=0.0,
                        provider="cache",
                        cached=True,
                    )
            except Exception:
                pass

        start = time.time()
        provider = "groq-primary"
        tokens = 0

        # Try primary
        try:
            text, tokens = self._call_llm(self.primary, query, context, provider)
        except Exception as e:
            print(f"Primary LLM failed: {e}. Trying fallback...")
            provider = "groq-fallback"
            try:
                text, tokens = self._call_llm(self.fallback, query, context, provider)
            except Exception as e2:
                raise RuntimeError(f"Both LLM providers failed. Primary: {e}. Fallback: {e2}")

        latency_ms = (time.time() - start) * 1000

        # Store in cache
        if self.cache:
            try:
                self.cache.setex(cache_key, self.cache_ttl, text)
            except Exception:
                pass

        return GatewayResponse(
            text=text,
            latency_ms=round(latency_ms, 2),
            provider=provider,
            cached=False,
            tokens_used=tokens,
        )
