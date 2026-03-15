import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Enterprise Hybrid Retriever implementation.
    Combines Dense (Vector) and Sparse (Keyword) retrieval strategies 
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self, 
        documents: List[Document], 
        model_name: str = 'all-MiniLM-L6-v2',
        k: int = 5
    ):
        self.documents = documents
        self.k = k
        self.model = SentenceTransformer(model_name)
        
        # Initialize Sparse Retriever (BM25)
        self.tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Initialize Dense Retriever (Vector Embeddings)
        # In production, this would be a Milvus or FAISS index.
        # Here we'll use local numpy-based similarity for demonstration.
        self.doc_embeddings = self.model.encode([doc.page_content for doc in documents])
        logger.info(f"HybridRetriever initialized with {len(documents)} documents.")

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[int], 
        sparse_results: List[int], 
        alpha: float = 60.0
    ) -> List[int]:
        """
        Applies Reciprocal Rank Fusion (RRF) to combine two sets of search results.
        
        Args:
            dense_results: Indices of documents ranked by dense retriever.
            sparse_results: Indices of documents ranked by sparse retriever.
            alpha: Smoothing parameter (default is 60).
        """
        fused_scores = {}
        
        for rank, doc_idx in enumerate(dense_results):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (rank + alpha)
            
        for rank, doc_idx in enumerate(sparse_results):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (rank + alpha)
            
        # Sort documents based on fused scores
        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return sorted_indices[:self.k]

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves top-k documents using hybrid search (BM25 + Dense).
        """
        logger.info(f"Retrieving for query: {query}")
        
        # 1. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_top_indices = np.argsort(sparse_scores)[::-1][:self.k*2] # Get more candidates for fusion
        
        # 2. Dense Search (Vector Similarity)
        query_embedding = self.model.encode([query])
        dense_scores = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        dense_top_indices = np.argsort(dense_scores)[::-1][:self.k*2]
        
        # 3. Reciprocal Rank Fusion
        fused_indices = self._reciprocal_rank_fusion(
            dense_results=dense_top_indices.tolist(), 
            sparse_results=sparse_top_indices.tolist()
        )
        
        # Final set of documents
        results = [self.documents[idx] for idx in fused_indices]
        logger.info(f"Retrieved {len(results)} relevant documents.")
        return results

if __name__ == "__main__":
    # Example usage
    sample_docs = [
        Document(page_content="Milvus is a vector database designed for AI.", metadata={"source": "milvus"}),
        Document(page_content="FAISS is a library for efficient similarity search.", metadata={"source": "faiss"}),
        Document(page_content="Hybrid search combines dense and sparse retrieval.", metadata={"source": "hybrid"}),
        Document(page_content="BM25 is a ranking function used by search engines.", metadata={"source": "bm25"}),
    ]
    retriever = HybridRetriever(sample_docs)
    results = retriever.retrieve("What is hybrid search and Milvus?")
    for doc in results:
        print(f"Content: {doc.page_content}")
