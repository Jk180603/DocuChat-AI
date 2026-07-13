"""
Hybrid Retriever — BM25 + FAISS Dense Vector Search
Uses modern langchain packages, no deprecated community imports
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        k: int = 5,
    ):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None
        self.bm25 = None
        self.documents = []

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided")

        print(f"Building retriever from {len(documents)} chunks...")
        self.documents = documents

        # Build FAISS
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        # Build BM25
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Retriever ready — {len(documents)} chunks indexed")

    def _reciprocal_rank_fusion(
        self, dense_docs: list, bm25_docs: list, k: int = 60
    ) -> list[Document]:
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + self.dense_weight * (1 / (rank + k))
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + self.bm25_weight * (1 / (rank + k))
            doc_map[key] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[:self.k]]

    def retrieve(self, query: str) -> list[Document]:
        if not self.vectorstore or not self.bm25:
            raise RuntimeError("Retriever not built. Call build() first.")

        # Dense retrieval
        dense_results = self.vectorstore.similarity_search(query, k=self.k)

        # BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[::-1][:self.k]
        bm25_results = [self.documents[i] for i in top_indices]

        # Combine with RRF
        return self._reciprocal_rank_fusion(dense_results, bm25_results)

    def save(self, path: str) -> None:
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load(self, path: str, documents: list[Document]) -> None:
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.documents = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)