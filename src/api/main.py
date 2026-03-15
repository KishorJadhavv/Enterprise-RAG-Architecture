from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time

from src.rag.hybrid_retriever import HybridRetriever
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise RAG Engine API",
    description="High-performance Retrieval Augmented Generation API",
    version="1.0.0"
)

# Mock documents for demonstration purposes
# In a real application, this would load from a vector database
MOCK_DOCS = [
    Document(page_content="Milvus is a distributed vector database.", metadata={"source": "manual"}),
    Document(page_content="BM25 is used for sparse retrieval.", metadata={"source": "whitepaper"}),
    Document(page_content="Hybrid search combines both for better results.", metadata={"source": "tutorial"}),
]

# Initialize Global Retriever
# In production, this would be an instance connected to a Milvus cluster
retriever = HybridRetriever(MOCK_DOCS)

class QueryRequest(BaseModel):
    """
    Standard query request model with validation.
    """
    query: str = Field(..., min_length=1, description="The search query.")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="The number of documents to retrieve.")

class DocumentResponse(BaseModel):
    """
    Response model for individual documents.
    """
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    """
    Comprehensive query response model.
    """
    query: str
    results: List[DocumentResponse]
    retrieval_time_ms: float

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to track request processing time.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Endpoint for hybrid retrieval queries.
    """
    try:
        logger.info(f"Received query: {request.query}")
        start_retrieval = time.time()
        
        # Perform Retrieval
        results = retriever.retrieve(request.query)
        
        retrieval_time = (time.time() - start_retrieval) * 1000
        
        formatted_results = [
            DocumentResponse(content=doc.page_content, metadata=doc.metadata) 
            for doc in results
        ]
        
        return QueryResponse(
            query=request.query,
            results=formatted_results,
            retrieval_time_ms=retrieval_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during retrieval.")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {"status": "healthy", "engine": "Enterprise-RAG"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
