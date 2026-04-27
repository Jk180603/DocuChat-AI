import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.config import settings

class DocumentProcessor:
    """Handle document loading and chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and chunk a single PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)
            
            return documents
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        return self.text_splitter.split_documents(documents)
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Complete pipeline: load and chunk PDF"""
        documents = self.load_pdf(file_path)
        if not documents:
            return []
        
        chunks = self.chunk_documents(documents)
        print(f"Processed {file_path}: {len(documents)} pages → {len(chunks)} chunks")
        return chunks
    
    def process_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """Process multiple PDF files"""
        all_chunks = []
        for file_path in file_paths:
            chunks = self.process_pdf(file_path)
            all_chunks.extend(chunks)
        
        print(f"Total chunks from all documents: {len(all_chunks)}")
        return all_chunks