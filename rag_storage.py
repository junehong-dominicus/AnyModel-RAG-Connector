import logging
import os
import uuid
from typing import List, Optional, Dict

# Dependencies: pip install langchain-community faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import your custom modules
from rag_chunker import ContentChunker
from rag_embedder import LocalEmbedder

logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Manages storage and retrieval of text chunks using FAISS and local embeddings.
    """
    
    def __init__(self, index_folder: str = "faiss_index"):
        self.index_folder = index_folder
        
        # Initialize our custom components
        self.chunker = ContentChunker()
        # Use the specific model you requested
        self.embedder = LocalEmbedder(model_name="text-embedding-embeddinggemma-300m-qat")
        
        self.vector_store = None

    def ingest(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Process text: Chunk -> Embed -> Store.
        """
        # 1. Split text into chunks (strings)
        text_chunks = self.chunker.split_text(text)
        
        if not text_chunks:
            logger.warning("No chunks generated from input text.")
            return []

        # 2. Convert to LangChain Documents
        # FAISS requires Document objects with page_content and metadata
        documents = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in text_chunks
        ]
        
        # Generate unique IDs for the documents
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # 3. Create or Update Vector Store
        # We pass the underlying LangChain embeddings object (self.embedder.embeddings)
        if self.vector_store is None:
            logger.info("Creating new vector store...")
            self.vector_store = FAISS.from_documents(
                documents, 
                self.embedder.embeddings,
                ids=ids
            )
        else:
            logger.info(f"Adding {len(documents)} documents to existing store...")
            self.vector_store.add_documents(documents, ids=ids)
            
        logger.info(f"Ingested {len(documents)} chunks.")
        return ids

    def save(self):
        """Persist the vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.index_folder)
            logger.info(f"Saved vector store to '{self.index_folder}'")
            
    def load(self):
        """Load the vector store from disk."""
        if os.path.exists(self.index_folder):
            self.vector_store = FAISS.load_local(
                self.index_folder, 
                self.embedder.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from '{self.index_folder}'")
        else:
            logger.warning(f"No vector store found at '{self.index_folder}'")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve similar documents."""
        if not self.vector_store:
            logger.warning("Vector store not initialized. Load or ingest data first.")
            return []
            
        return self.vector_store.similarity_search(query, k=k)

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        if not self.vector_store:
            logger.warning("Vector store not initialized.")
            return False
            
        return self.vector_store.delete(ids)

    def delete_by_metadata(self, **kwargs) -> List[str]:
        """
        Delete documents where metadata matches the provided kwargs.
        Example: db.delete_by_metadata(source="manual_example")
        """
        if not self.vector_store:
            return []
            
        ids_to_delete = []
        # Iterate over docstore to find matches
        for doc_id, doc in self.vector_store.docstore._dict.items():
            if all(doc.metadata.get(key) == value for key, value in kwargs.items()):
                ids_to_delete.append(doc_id)
                
        if ids_to_delete:
            self.vector_store.delete(ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} documents matching {kwargs}")
            
        return ids_to_delete

    def clear(self):
        """Clear all documents from the vector store."""
        if not self.vector_store:
            return
            
        ids = list(self.vector_store.docstore._dict.keys())
        if ids:
            self.vector_store.delete(ids)
            logger.info(f"Cleared {len(ids)} documents.")
            self.save()

if __name__ == "__main__":
    # Quick verification
    logging.basicConfig(level=logging.INFO)
    
    db = VectorDatabase()
    
    # Example: Ingesting some data
    sample_content = """
    FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other. 
    It solves the problem of nearest neighbor search in high-dimensional vector sets.
    """
    
    print("--- Ingesting Data ---")
    db.ingest(sample_content, metadata={"source": "manual_example"})
    db.save()
    
    # Example: Searching
    print("\n--- Searching ---")
    results = db.search("What is FAISS used for?")
    for doc in results:
        print(f"Result: {doc.page_content}")
        