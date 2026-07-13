"""
Document Ingestion Pipeline
Handles PDF loading, cleaning, and semantic chunking
"""
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass
class IngestionResult:
    chunks: list[Document]
    num_pages: int
    num_chunks: int
    file_hash: str
    filename: str


class DocumentIngestionPipeline:
    SUPPORTED_FORMATS = [".pdf"]
    MAX_FILE_SIZE_MB = 50

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def _validate(self, file_path: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: {self.SUPPORTED_FORMATS}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(f"File too large: {size_mb:.1f}MB. Max: {self.MAX_FILE_SIZE_MB}MB")

    def _hash_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    def _clean_text(self, text: str) -> str:
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        return text.strip()

    def ingest(self, file_path: str) -> IngestionResult:
        self._validate(file_path)

        file_hash = self._hash_file(file_path)
        filename = Path(file_path).name

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        if not pages:
            raise ValueError(f"No content extracted from {filename}")

        # Clean text in each page
        for page in pages:
            page.page_content = self._clean_text(page.page_content)
            page.metadata.update({
                "filename": filename,
                "file_hash": file_hash,
            })

        # Filter empty pages
        pages = [p for p in pages if len(p.page_content.strip()) > 50]

        # Chunk
        chunks = self.splitter.split_documents(pages)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return IngestionResult(
            chunks=chunks,
            num_pages=len(pages),
            num_chunks=len(chunks),
            file_hash=file_hash,
            filename=filename,
        )

    def ingest_multiple(self, file_paths: list[str]) -> list[Document]:
        all_chunks = []
        for path in file_paths:
            result = self.ingest(path)
            all_chunks.extend(result.chunks)
            print(f"Ingested {result.filename}: {result.num_chunks} chunks from {result.num_pages} pages")
        return all_chunks