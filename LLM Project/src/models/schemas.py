from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentUpload(BaseModel):
    """Schema for document upload"""
    filename: str
    content: bytes

class QueryRequest(BaseModel):
    """Schema for question query"""
    question: str = Field(..., description="User's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")

class QueryResponse(BaseModel):
    """Schema for query response"""
    answer: str = Field(..., description="AI-generated answer")
    sources: List[dict] = Field(..., description="Source documents used")
    conversation_id: str = Field(..., description="Conversation ID")

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    message: str