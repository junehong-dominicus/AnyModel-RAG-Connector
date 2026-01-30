import logging
import os
from typing import List

# Dependencies: pip install langchain-openai
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Generates embeddings using a local LLM server (e.g., LM Studio) via OpenAI-compatible API.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:1234/v1", 
        model_name: str = "text-embedding-embeddinggemma-300m-qat"
    ):
        """
        Initialize the embedder.
        
        Args:
            base_url: The local server URL (default: LM Studio default).
            model_name: The specific model identifier loaded in LM Studio.
        """
        self.base_url = base_url
        self.model_name = model_name
        
        # Initialize OpenAIEmbeddings pointing to local server
        # api_key is required by the library but ignored by most local servers like LM Studio
        self.embeddings = OpenAIEmbeddings(
            openai_api_key="lm-studio",
            openai_api_base=base_url,
            model=model_name,
            check_embedding_ctx_length=False
        )
        logger.info(f"Initialized LocalEmbedder with model '{model_name}' at '{base_url}'")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors (lists of floats).
        """
        if not texts:
            return []
            
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} vectors.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query string.
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

if __name__ == "__main__":
    # Quick verification
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing LocalEmbedder...")
    print("NOTE: Ensure LM Studio is running with the server started on port 1234")
    
    embedder = LocalEmbedder()
    
    test_text = ["This is a test chunk for embedding generation."]
    try:
        vectors = embedder.embed_documents(test_text)
        if vectors:
            print(f"Vector dimension: {len(vectors[0])}")
            print(f"First 5 values: {vectors[0][:5]}")
        else:
            print("No vectors returned.")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure LM Studio is running and the server is started.")