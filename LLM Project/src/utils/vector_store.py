from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config.config import settings
import os

class VectorStoreManager:
    """Manage FAISS vector store for document retrieval"""
    
    def __init__(self):
        # Use free HuggingFace embeddings (no API key needed!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store: Optional[FAISS] = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create new vector store from documents"""
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating vector store from {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("Vector store created successfully!")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """Add more documents to existing vector store"""
        if self.vector_store is None:
            self.create_vector_store(documents)
        else:
            print(f"Adding {len(documents)} documents to vector store...")
            self.vector_store.add_documents(documents)
    
    def save_vector_store(self, path: Optional[str] = None):
        """Save vector store to disk"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        save_path = path or settings.VECTOR_STORE_PATH
        self.vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
    
    def load_vector_store(self, path: Optional[str] = None) -> FAISS:
        """Load vector store from disk"""
        load_path = path or settings.VECTOR_STORE_PATH
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        print(f"Loading vector store from {load_path}...")
        self.vector_store = FAISS.load_local(
            load_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully!")
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def get_retriever(self, k: int = 4):
        """Get retriever for RAG chain"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )