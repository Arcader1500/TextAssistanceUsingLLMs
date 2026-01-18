import sys
import io

# Fix encoding for Windows consoles/redirects
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import warnings
warnings.filterwarnings("ignore")

# 1. SETUP CLIENT
DB_PATH = "../arxiv"
client = chromadb.PersistentClient(path=DB_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="Arxiv-Database",
    embedding_function=embedding_func
)

# 2. LOAD CROSS-ENCODER (Re-ranker)
print("⏳ Loading Cross-Encoder Model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Cross-Encoder Loaded.")

def improved_search(query, top_k_retrieval=20, top_k_rerank=5):
    print(f"\n🔍 Query: {query}")
    
    # Stage 1: Semantic Retrieval (Bio-Encoder)
    results = collection.query(
        query_texts=[query],
        n_results=top_k_retrieval
    )
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    # Stage 2: Re-ranking (Cross-Encoder)
    # Prepare pairs: (Query, Document_Context)
    pairs = [[query, doc] for doc in documents]
    
    # Predict scores
    scores = cross_encoder.predict(pairs)
    
    # Sort by score (descending)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    print("-" * 50)
    print(f"Top {top_k_rerank} Results after Re-ranking:")
    print("-" * 50)
    
    for rank, idx in enumerate(ranked_indices[:top_k_rerank]):
        print(f"Rank {rank+1} (Score: {scores[idx]:.4f})")
        print(f"Source: {metadatas[idx]['source']} (Page {metadatas[idx]['page']})")
        print(f"Content: {documents[idx][:300]}...") # Show glimpse
        print("")

if __name__ == "__main__":
    # Test with the problematic query from user's results
    test_query = "What is the Assessment Misalignment problem in Large Language Models?"
    improved_search(test_query)
    
    # Test with another query
    test_query_2 = "What are the three steps in the Alignment Fine-Tuning paradigm?"
    improved_search(test_query_2)
