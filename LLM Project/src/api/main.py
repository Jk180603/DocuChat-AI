from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
from pathlib import Path
import uuid
import os

from src.models.schemas import QueryRequest, QueryResponse, HealthCheck
from src.utils.document_loader import DocumentProcessor
from src.utils.vector_store import VectorStoreManager
from src.utils.llm_chain import RAGChain
from config.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="DocuChat AI API",
    description="Intelligent Document Q&A System with RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allows frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components
doc_processor = DocumentProcessor()
vector_store_manager = VectorStoreManager()
rag_chain = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global rag_chain
    
    print("🚀 Starting DocuChat AI API...")
    print(f"📁 Upload directory: {settings.UPLOAD_DIR}")
    print(f"💾 Vector store directory: {settings.VECTOR_STORE_DIR}")
    
    # Ensure directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to load existing vector store
    try:
        vector_store_manager.load_vector_store()
        rag_chain = RAGChain(vector_store_manager)
        print("✅ Loaded existing vector store successfully!")
        print(f"📊 Ready to answer questions!")
    except FileNotFoundError:
        print("⚠️  No existing vector store found.")
        print("📤 Please upload documents to create one.")
    except Exception as e:
        print(f"⚠️  Error loading vector store: {str(e)}")
        print("📤 Please upload documents to create a new one.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down DocuChat AI API...")

@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint - health check"""
    return HealthCheck(
        status="healthy",
        message="🚀 DocuChat AI is running! Visit /docs for API documentation."
    )

@app.get("/health")
async def health_check():
    """Detailed health check with system status"""
    vector_store_status = vector_store_manager.vector_store is not None
    rag_chain_status = rag_chain is not None
    
    # Count uploaded documents
    try:
        uploaded_files = list(settings.UPLOAD_DIR.glob("*.pdf"))
        num_documents = len(uploaded_files)
    except:
        num_documents = 0
    
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "vector_store_initialized": vector_store_status,
        "rag_chain_initialized": rag_chain_status,
        "documents_uploaded": num_documents,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "retrieval_k": settings.RETRIEVAL_K
    }

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload PDF documents and create/update vector store
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        Success message with processing details
    """
    global rag_chain
    
    if not files:
        raise HTTPException(
            status_code=400, 
            detail="No files provided. Please upload at least one PDF file."
        )
    
    uploaded_files = []
    skipped_files = []
    
    try:
        print(f"\n📤 Uploading {len(files)} file(s)...")
        
        # Save uploaded files
        for file in files:
            # Validate file type
            if not file.filename.endswith('.pdf'):
                skipped_files.append(f"{file.filename} (not a PDF)")
                continue
            
            # Save file
            file_path = settings.UPLOAD_DIR / file.filename
            
            print(f"💾 Saving: {file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(str(file_path))
        
        if not uploaded_files:
            raise HTTPException(
                status_code=400,
                detail=f"No valid PDF files provided. Skipped: {', '.join(skipped_files)}"
            )
        
        print(f"✅ Saved {len(uploaded_files)} PDF file(s)")
        
        # Process documents into chunks
        print("🔄 Processing documents into chunks...")
        chunks = doc_processor.process_multiple_pdfs(uploaded_files)
        
        if not chunks:
            raise HTTPException(
                status_code=500,
                detail="Failed to process PDF files. Please check file format."
            )
        
        print(f"✅ Created {len(chunks)} chunks from documents")
        
        # Create or update vector store
        print("🧠 Creating/updating vector store...")
        if vector_store_manager.vector_store is None:
            print("   Creating new vector store...")
            vector_store_manager.create_vector_store(chunks)
        else:
            print("   Adding to existing vector store...")
            vector_store_manager.add_documents(chunks)
        
        # Save vector store to disk
        print("💾 Saving vector store...")
        vector_store_manager.save_vector_store()
        
        # Initialize/update RAG chain
        print("🤖 Initializing RAG chain...")
        rag_chain = RAGChain(vector_store_manager)
        
        print("✅ Upload complete!\n")
        
        response = {
            "message": f"Successfully processed {len(uploaded_files)} document(s)",
            "files": [Path(f).name for f in uploaded_files],
            "total_chunks": len(chunks),
            "status": "success"
        }
        
        if skipped_files:
            response["skipped_files"] = skipped_files
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing documents: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents with a question
    
    Args:
        request: QueryRequest with question and optional conversation_id
        
    Returns:
        QueryResponse with answer and sources
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload PDF documents first using the /upload endpoint."
        )
    
    try:
        print(f"\n❓ Question: {request.question}")
        
        # Get answer from RAG chain
        result = rag_chain.ask_question(request.question)
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        print(f"✅ Answer generated (conversation: {conversation_id[:8]}...)")
        print(f"📎 Sources: {len(result['sources'])} chunks retrieved\n")
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=conversation_id
        )
    
    except Exception as e:
        print(f"❌ Error during query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating answer: {str(e)}"
        )

@app.delete("/reset")
async def reset_system():
    """
    Reset the entire system - clear vector store and uploaded documents
    
    WARNING: This will delete all uploaded documents and the vector store!
    """
    global rag_chain
    
    try:
        print("\n🔄 Resetting system...")
        
        # Clear in-memory components
        vector_store_manager.vector_store = None
        rag_chain = None
        
        # Delete uploaded documents
        if settings.UPLOAD_DIR.exists():
            for file in settings.UPLOAD_DIR.glob("*.pdf"):
                file.unlink()
            print("🗑️  Deleted uploaded documents")
        
        # Delete vector store
        if settings.VECTOR_STORE_DIR.exists():
            import shutil
            shutil.rmtree(settings.VECTOR_STORE_DIR)
            settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            print("🗑️  Deleted vector store")
        
        print("✅ System reset complete!\n")
        
        return {
            "message": "System reset successfully",
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error during reset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting system: {str(e)}"
        )

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    
    Returns:
        List of uploaded PDF files
    """
    try:
        pdf_files = list(settings.UPLOAD_DIR.glob("*.pdf"))
        
        documents = [
            {
                "filename": file.name,
                "size_mb": round(file.stat().st_size / (1024 * 1024), 2),
                "uploaded_at": file.stat().st_mtime
            }
            for file in pdf_files
        ]
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a specific document
    
    NOTE: This only deletes the file, not the vectors. 
    Use /reset to fully remove document data from vector store.
    """
    try:
        file_path = settings.UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found"
            )
        
        file_path.unlink()
        
        return {
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector embeddings still exist. Use /reset to clear vector store."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

# Development/testing endpoints
@app.get("/debug/config")
async def get_config():
    """Get current configuration (for debugging)"""
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "retrieval_k": settings.RETRIEVAL_K,
        "upload_dir": str(settings.UPLOAD_DIR),
        "vector_store_path": settings.VECTOR_STORE_PATH
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )