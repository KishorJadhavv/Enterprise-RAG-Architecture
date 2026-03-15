from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from ..utils.decorators import time_execution

class CrossEncoderReranker:
    """
    Reranking layer using Cross-Encoders for high-precision document relevance scoring.
    Cross-Encoders process (query, document) pairs together, yielding better results than bi-encoders.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the reranker with a pre-trained cross-encoder model.

        Args:
            model_name (str): The model name on HuggingFace Hub.
        """
        self.model = CrossEncoder(model_name)

    @time_execution
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Reranks a list of candidate documents based on their relevance to a query.

        Args:
            query (str): User search query.
            documents (List[str]): Candidate documents retrieved from the vector store or keyword index.
            top_k (int): Number of top documents to return after reranking.

        Returns:
            List[Tuple[str, float]]: List of (document_text, score) tuples, sorted by score.
        """
        if not documents:
            return []

        # Prepare pairs for the Cross-Encoder: (query, document)
        pairs = [[query, doc] for doc in documents]
        
        try:
            # Predict scores for each pair
            scores = self.model.predict(pairs)
            
            # Combine documents with their scores
            scored_docs = list(zip(documents, scores))
            
            # Sort by score in descending order
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return scored_docs[:top_k]
        except Exception as e:
            # TODO: Add structured logging
            print(f"Error during reranking: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    reranker = CrossEncoderReranker()
    sample_query = "How to optimize Milvus performance?"
    sample_docs = [
        "To optimize Milvus, use high-performance NVMe drives and tune the HNSW parameters like efConstruction.",
        "The sky is blue today because of Rayleigh scattering.",
        "Milvus supports diverse index types, including IVF_FLAT, HNSW, and ANNOY."
    ]
    
    # top_results = reranker.rerank(sample_query, sample_docs, top_k=2)
    # for i, (doc, score) in enumerate(top_results):
    #     print(f"[{i+1}] Score: {score:.4f} | Content: {doc}")
