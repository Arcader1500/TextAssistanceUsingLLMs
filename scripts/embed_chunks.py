import json
import os
import pickle
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- Configuration ---
PROCESSED_CHUNKS_FILE = "../rag_data/processed_chunks.json"
# Database path relative to this script (scripts/ -> root/seamanuals)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seamanuals")
COLLECTION_NAME = "Sea-Database"
# Upgrading to a better open-source model as requested
# Note: BAAI/bge-base-en-v1.5 (768 dim) vs all-MiniLM-L6-v2 (384 dim)
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 

def main():
    print("--- Embed Chunks to ChromaDB ---")
    
    # 1. Check for processed chunks
    if not os.path.exists(PROCESSED_CHUNKS_FILE):
        print(f"Error: {PROCESSED_CHUNKS_FILE} not found. Run chunk processing first.")
        return

    # 2. Load Processed Chunks
    print(f"Loading chunks from {PROCESSED_CHUNKS_FILE}...")
    with open(PROCESSED_CHUNKS_FILE, "r") as f:
        processed_chunks_data = json.load(f)
    print(f"Loaded {len(processed_chunks_data)} chunks.")

    # 3. Initialize ChromaDB Client
    print(f"Connecting to ChromaDB at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # 4. Initialize Embedding Function
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    print("WARNING: You are changing the embedding model. Ensure 'app.py' and other consumers are updated to use this model, otherwise retrieval will fail due to dimension mismatch.")
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 5. Get or Create Collection
    # Note: If the collection exists with a different dimension (e.g. from MiniLM), 
    # Chroma might raise an error or behavior might be undefined if we append.
    # We proceed assuming the user wants to update/overwrite or is starting fresh.
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        print(f"Accessed collection '{COLLECTION_NAME}'. Existing count: {collection.count()}")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    # 6. Prepare Data for Upsert
    ids = []
    documents = []
    metadatas = []

    print("Preparing data for upsert...")
    for chunk in processed_chunks_data:
        # Use the ID provided in the processed data
        chunk_id = chunk.get("id")
        content = chunk.get("content")
        meta = chunk.get("metadata", {})
        
        # Ensure ID and content exist
        if chunk_id and content:
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(meta)

    # 7. Batch Upsert
    batch_size = 100
    total_chunks = len(ids)
    print(f"Upserting {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{total_chunks}...")
    
    print(f"Upsert complete. Total chunks in collection: {collection.count()}")

    # 8. Build and Save BM25 Retriever (Sync with Vectorize.py approach)
    print("Building BM25 index...")
    bm25_documents = [
        Document(page_content=doc, metadata={**meta, "id": cid})
        for cid, doc, meta in zip(ids, documents, metadatas)
    ]
    
    bm25_retriever = BM25Retriever.from_documents(bm25_documents)
    
    bm25_path = os.path.join(DB_PATH, "bm25_retriever.pkl")
    print(f"Saving BM25 retriever to {bm25_path}...")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    # 9. Verification
    print("\n--- Verification: Sample Retrieval ---")
    query = "What are the steps for pre-underway checks?"
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print(f"Query: '{query}'")
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            print(f"\nResult {i+1}:")
            print(f"Content: {doc[:200]}...")
            print(f"Metadata: {meta}")
    else:
        print("No results found.")

    print("\nProcess complete.")

if __name__ == "__main__":
    main()

