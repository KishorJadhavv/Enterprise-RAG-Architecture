import os
import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Advanced Document Processor for handling multiple file formats and 
    applying sophisticated chunking strategies.
    """

    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Loads and splits PDF documents."""
        logger.info(f"Loading PDF from: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_markdown(self, file_path: str) -> List[Document]:
        """Loads Markdown files with header-aware splitting."""
        logger.info(f"Loading Markdown from: {file_path}")
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return markdown_splitter.split_text(content)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Applies Recursive Character Text Splitting to a list of documents."""
        logger.info(f"Processing {len(documents)} documents for chunking.")
        return self.text_splitter.split_documents(documents)

    def ingest_directory(self, directory_path: str) -> List[Document]:
        """Batch processes all supported files in a directory."""
        all_chunks = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith(".pdf"):
                docs = self.load_pdf(file_path)
                all_chunks.extend(self.process_documents(docs))
            elif filename.endswith(".md"):
                docs = self.load_markdown(file_path)
                # Header-split docs might already be appropriately sized, 
                # but we can further split them if necessary.
                all_chunks.extend(self.process_documents(docs))
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    # result = processor.ingest_directory("./data")
    print("DocumentProcessor initialized.")
