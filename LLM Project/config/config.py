import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings"""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploaded_docs"
    VECTOR_STORE_DIR = DATA_DIR / "vectorstore"
    
    # Create directories if they don't exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Chunking parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Vector store settings
    VECTOR_STORE_PATH = str(VECTOR_STORE_DIR / "faiss_index")
    
    # Retrieval settings
    RETRIEVAL_K = 4  # Number of chunks to retrieve
    
settings = Settings()