# RAG Experiments & Prototypes

This repository houses a collection of Retrieval-Augmented Generation (RAG) projects and experiments designed to demonstrate advanced document processing, vectorization strategies, and LLM integration for specialized domains.

## Project Structure

### 1. [Arxiv_Project](./Arxiv_Project)
A specialized RAG pipeline focused on academic research papers.
- **Features:** Automated Arxiv PDF downloading, semantic search with ChromaDB, and re-ranking using Cross-Encoders.
- **Goal:** To accurately answer queries about specific scientific papers, handling complex academic language and citations.

### 2. [Sea_Project](./Sea_Project)
A RAG system tailored for Maritime Regulations and Training Manuals.
- **Features:** robust PDF processing pipeline capable of handling complex layouts, including table extraction and OCR (Optical Character Recognition) for scanned documents.
- **Goal:** To provide precise answers regarding maritime safety standards (STCW), regulations, and training requirements.

### 3. [Prototype](./Prototype)
A learning sandbox containing Jupyter notebooks that step through the fundamental components of LLM applications:
- Environment setup
- Basic LLM interaction
- Simple RAG implementations
- Fine-tuning (LoRA) experiments

### 4. [utils](./utils)
Shared utility library used across projects.
- **llmclass.py:** Wrapper for OpenAI-compatible LLM API interactions.
- **search.py:** Implements advanced search logic, including Hybrid Search (Vector + BM25) and Cross-Encoder Re-ranking.

## Getting Started

### Prerequisites
Ensure you have Python installed. The project dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Usage
Navigate to the specific project folders (`Arxiv_Project` or `Sea_Project`) and follow the instructions in their respective README files to run the pipelines.
