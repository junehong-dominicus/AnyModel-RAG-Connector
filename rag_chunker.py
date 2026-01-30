import logging
from typing import List

# Dependencies: pip install langchain-text-splitters tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

logger = logging.getLogger(__name__)

class ContentChunker:
    """
    Handles splitting of text content into chunks suitable for RAG (Retrieval Augmented Generation).
    Uses token-based length calculation for accurate context window management.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        model_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
            model_name: Encoding model name for tiktoken (default: cl100k_base for GPT-4/3.5).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer for {model_name}, falling back to cl100k_base. Error: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _token_len(self, text: str) -> int:
        """Calculate the number of tokens in a text string."""
        # disallowed_special=() ensures we don't crash on special tokens, just encode them
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The input text to split.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # RecursiveCharacterTextSplitter tries to split on the first separator, 
        # if the chunk is too large, it moves to the next separator.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
            strip_whitespace=True
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

if __name__ == "__main__":
    # Quick verification
    logging.basicConfig(level=logging.INFO)
    chunker = ContentChunker(chunk_size=50, chunk_overlap=10)
    print("Chunker initialized successfully.")