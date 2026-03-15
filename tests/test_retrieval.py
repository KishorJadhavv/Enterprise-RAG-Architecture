import pytest
from langchain.schema import Document
from src.rag.hybrid_retriever import HybridRetriever

@pytest.fixture
def sample_documents():
    """
    Fixture for providing test documents.
    """
    return [
        Document(page_content="Python is a versatile programming language.", metadata={"id": 1}),
        Document(page_content="Data Science involves statistics and programming.", metadata={"id": 2}),
        Document(page_content="RAG systems use vector databases like Milvus.", metadata={"id": 3}),
        Document(page_content="Hybrid search combines BM25 and dense vectors.", metadata={"id": 4}),
    ]

def test_hybrid_retriever_initialization(sample_documents):
    """
    Test if the retriever initializes correctly with documents.
    """
    retriever = HybridRetriever(sample_documents)
    assert len(retriever.documents) == 4
    assert retriever.k == 5

def test_retrieval_logic(sample_documents):
    """
    Test if retrieval returns relevant documents.
    """
    retriever = HybridRetriever(sample_documents, k=2)
    query = "What is RAG and Milvus?"
    results = retriever.retrieve(query)
    
    assert len(results) == 2
    # Ensure the document with 'RAG' and 'Milvus' is highly ranked
    assert any("Milvus" in doc.page_content for doc in results)

def test_reciprocal_rank_fusion(sample_documents):
    """
    Test RRF logic specifically.
    """
    retriever = HybridRetriever(sample_documents)
    
    dense_results = [0, 1, 2] # Doc indices ranked by dense
    sparse_results = [2, 0, 1] # Doc indices ranked by sparse
    
    fused_indices = retriever._reciprocal_rank_fusion(dense_results, sparse_results)
    
    # Doc index 2 and 0 appear in both and should be ranked high
    assert 2 in fused_indices
    assert 0 in fused_indices
