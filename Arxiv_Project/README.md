# Arxiv RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for querying and analyzing academic papers from Arxiv. It combines semantic search with a re-ranking step to ensure high relevance in retrieved contexts.

## Features

- **Automated Ingestion:** `download_pdfs.py` allows fetching papers directly from Arxiv based on search queries.
- **Vector Database:** Uses **ChromaDB** to store and retrieve document embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Advanced Retrieval:** Implements a two-stage retrieval process:
  1. **Bi-Encoder Retrieval:** Fast semantic search to get the top candidates.
  2. **Cross-Encoder Re-ranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score and re-rank passages for precise context alignment.
- **LLM Integration:** Connects to OpenAI-compatible endpoints (e.g., DeepSeek, GPT-4, or local models via Ollama) to generate grounded answers.

## File Structure

- `download_pdfs.py`: Script to search and download PDFs from Arxiv.
- `vectorize.py`: Processes PDFs, splits text, creates embeddings, and stores them in ChromaDB.
- `Pipeline.ipynb`: The main notebook for querying the system. It handles the full RAG workflow: Retrieval -> Re-ranking -> Generation.
- `rag_queries.json`: Contains sample queries and ground-truth answers for evaluation.

## Usage

1.  **Download Data:**
    ```bash
    python download_pdfs.py
    ```
    (Modify the query in the script as needed).

2.  **Build Vector Index:**
    ```bash
    python vectorize.py
    ```
    This will process PDFs in `rag_data/` and save the index to `arxiv/`.

3.  **Run Queries:**
    Open `Pipeline.ipynb` in Jupyter and execute the cells to query the system.
