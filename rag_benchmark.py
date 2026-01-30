import time
import logging
import uuid
import shutil
import os
import sys
import statistics
from typing import List, Dict

# Ensure we can import modules from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_storage import VectorDatabase
except ImportError:
    print("Error: Could not import rag_storage. Make sure you are running this script from the examples directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RAG_Benchmark")

def generate_synthetic_data(num_docs: int = 20) -> List[Dict[str, str]]:
    """
    Generates synthetic documents with unique keywords for retrieval testing.
    Each document contains a unique UUID that acts as the 'needle' in the haystack.
    """
    data = []
    for i in range(num_docs):
        unique_id = str(uuid.uuid4())[:8]
        # The keyword is the target we want to retrieve
        keyword = f"CODE_{unique_id}"
        
        # Content mixes the keyword with filler text
        content = (
            f"Document ID: {i}. "
            f"This is a confidential report regarding project {keyword}. "
            f"The system must be able to retrieve this specific code when queried. "
            f"Filler text follows: {str(uuid.uuid4())} " * 5
        )
        
        data.append({
            "content": content,
            "keyword": keyword,
            "id": i
        })
    return data

def benchmark_rag(num_docs: int = 20, k: int = 3):
    """
    Runs a benchmark on the RAG system measuring ingestion speed, retrieval speed, and accuracy.
    
    Args:
        num_docs: Number of documents to ingest.
        k: Number of results to retrieve (Top-K).
    """
    # Use a temporary folder for benchmarking
    benchmark_index = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index_benchmark")
    
    # Clean up previous benchmark if exists
    if os.path.exists(benchmark_index):
        shutil.rmtree(benchmark_index)
        
    logger.info(f"Starting benchmark with {num_docs} documents...")
    logger.info("NOTE: Ensure LM Studio (or your embedding server) is running on port 1234.")
    
    try:
        db = VectorDatabase(index_folder=benchmark_index)
    except Exception as e:
        logger.error(f"Failed to initialize VectorDatabase: {e}")
        return

    # --- Phase 1: Ingestion ---
    data = generate_synthetic_data(num_docs)
    ingest_times = []
    
    logger.info("--- Phase 1: Ingestion ---")
    start_total_ingest = time.time()
    
    for i, item in enumerate(data):
        t0 = time.time()
        # We pass the ID in metadata to verify retrieval later
        # Note: ingest returns a list of IDs, but we track our own integer ID in metadata
        db.ingest(item["content"], metadata={"benchmark_id": item["id"]})
        t1 = time.time()
        duration = t1 - t0
        ingest_times.append(duration)
        if (i + 1) % 5 == 0:
            logger.info(f"Ingested {i + 1}/{num_docs} docs...")
        
    end_total_ingest = time.time()
    db.save()
    
    avg_ingest = statistics.mean(ingest_times) if ingest_times else 0
    total_ingest = end_total_ingest - start_total_ingest
    logger.info(f"Ingestion Complete. Total: {total_ingest:.2f}s, Avg per doc: {avg_ingest:.4f}s")

    # --- Phase 2: Retrieval ---
    logger.info(f"--- Phase 2: Retrieval (Top-k={k}) ---")
    
    retrieval_times = []
    hits = 0
    
    for item in data:
        # Query specifically for the unique keyword
        query = f"What is the report for project {item['keyword']}?"
        expected_id = item["id"]
        
        t0 = time.time()
        results = db.search(query, k=k)
        t1 = time.time()
        retrieval_times.append(t1 - t0)
        
        # Check accuracy (Recall@K)
        # We look for the document with metadata['benchmark_id'] matching our expected_id
        found = False
        for doc in results:
            if str(doc.metadata.get("benchmark_id")) == str(expected_id):
                found = True
                break
        
        if found:
            hits += 1
        else:
            logger.debug(f"Missed: Query '{query}' did not retrieve doc {expected_id}")
            
    avg_retrieval = statistics.mean(retrieval_times) if retrieval_times else 0
    accuracy = (hits / num_docs) * 100 if num_docs > 0 else 0
    
    # --- Results ---
    print("\n" + "="*40)
    print("       RAG SYSTEM BENCHMARK RESULTS       ")
    print("="*40)
    print(f"Documents Ingested: {num_docs}")
    print(f"Top-K Setting:      {k}")
    print("-" * 40)
    print(f"Ingestion Speed:    {avg_ingest:.4f} sec/doc")
    print(f"Retrieval Speed:    {avg_retrieval:.4f} sec/query")
    print(f"Accuracy (Recall):  {accuracy:.2f}%")
    print("="*40 + "\n")
    
    # Cleanup
    if os.path.exists(benchmark_index):
        shutil.rmtree(benchmark_index)
        logger.info("Benchmark index cleaned up.")

if __name__ == "__main__":
    # You can adjust num_docs to test scalability (e.g., 10, 100, 1000)
    benchmark_rag(num_docs=10, k=3)
