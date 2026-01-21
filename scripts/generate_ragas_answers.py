import os
import sys
import json
import pickle
import re
import time
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llmclass import LLM
from utils.search import HybridSearch

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(PROJECT_ROOT, "seamanuals")
BM25_PATH = os.path.join(DB_PATH, "bm25_retriever.pkl")
DATASET_PATH = os.path.join(PROJECT_ROOT, "rag_data", "ragas_qa_dataset.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "rag_data", "rag_generated_answers.json")
RATE_LIMIT_WAIT_SECONDS = 60
LIMIT_QUERIES = None  # Set to None to process all queries, or an integer to limit

def parse_llm_output(text):
    """
    Removes the <think>...</think> block from the LLM response.
    """
    pattern = r"<think>(.*?)</think>"
    # Use re.DOTALL to match across multiple lines
    final_answer = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return final_answer

def load_existing_results():
    """Loads existing results to resume progress."""
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Output file exists but is invalid. Starting fresh.")
    return []

def main():
    print("Initializing components...")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
    collection = client.get_or_create_collection(name="Sea-Database", embedding_function=embedding_func)
    
    # Load BM25 Retriever
    if not os.path.exists(BM25_PATH):
        print(f"Error: BM25 retriever not found at {BM25_PATH}")
        sys.exit(1)
        
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    
    # Initialize Hybrid Search
    search_fn = HybridSearch.get(collection, bm25_retriever)
    
    # Initialize LLM
    rag = LLM(model_name='qwen-3-32b')
    
    # Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)
        
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Apply query limit if specified
    if LIMIT_QUERIES is not None:
        print(f"Limiting processing to the first {LIMIT_QUERIES} queries.")
        data = data[:LIMIT_QUERIES]
    
    # Load existing results for checkpointing
    results = load_existing_results()
    completed_query_numbers = {entry.get("query_number") for entry in results}
    
    print(f"Processing {len(data)} queries. Already completed: {len(completed_query_numbers)}.")
    
    for entry in tqdm(data):
        query_number = entry.get('query_number')
        if query_number in completed_query_numbers:
            continue
            
        batch_index = entry.get('batch_index')
        question = entry.get('question')
        source_chunk_ids = entry.get('source_chunk_ids', [])
        
        # 1. Search
        docs = search_fn(question)
        
        context_strs = []
        retrieved_chunk_ids = []
        for doc in docs:
            source = doc.get('source', 'Unknown Source')
            content = doc.get('content', '')
            doc_id = doc.get('id', '')
            context_strs.append(f"{source} -> {content}")
            retrieved_chunk_ids.append(doc_id)
            
        # 2. Generate Answer with Rate Limit Handling
        success = False
        while not success:
            try:
                raw_answer = rag.generate_answer(question, context_strs)
                
                # Check for errors returned as strings from llmclass.py
                if raw_answer.startswith("Error during inference:"):
                    raise Exception(raw_answer)

                # 3. Clean Answer
                clean_answer = parse_llm_output(raw_answer)
                
                # 4. Rank 1 Chunk Content
                contexts = [d.get('content', '') for d in docs]
                
                # 5. Construct Result Entry
                result_entry = {
                    "query_number": query_number,
                    "batch_index": batch_index,
                    "question": question,
                    "source_chunk_ids": source_chunk_ids,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "contexts": contexts,
                    "answer": clean_answer
                }
                
                results.append(result_entry)
                
                # Save immediately to allow resuming on crash
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                
                success = True
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f"\nRate limit hit for query {query_number}. Waiting {RATE_LIMIT_WAIT_SECONDS} seconds...")
                    time.sleep(RATE_LIMIT_WAIT_SECONDS)
                else:
                    print(f"\nError processing query {query_number}: {e}. Skipping.")
                    break

    print(f"Done. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
