# Maritime Regulations RAG Pipeline (Sea_Project)

This project is a specialized RAG system designed to handle complex maritime documentation, such as STCW guides and training manuals. It features a robust ingestion pipeline capable of dealing with the formatting challenges common in technical manuals.

## Key Features

- **Hybrid PDF Extraction:**
  - **Text & Tables:** Uses `pdfplumber` to extract structured text and tables, converting tables into Markdown for better LLM comprehension.
  - **OCR Fallback:** Integrates `pytesseract` and `pdf2image` to handle scanned pages or images where text extraction fails.
- **Vector Storage:** Persists embeddings in **ChromaDB** for efficient retrieval.
- **Context-Aware Splitting:** Uses `RecursiveCharacterTextSplitter` to manage chunking while preserving document metadata.
- **Re-ranking:** Enhances retrieval accuracy using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).

## File Structure

- `vectorize.py`: The core ingestion script. It processes PDFs from `rag_data/`, applies OCR/Table extraction logic, and populates the vector database (`seamanuals/`).
- `Pipeline.ipynb`: The interactive notebook for querying the maritime knowledge base.
- `in_memory_pipe.ipynb`: A development notebook for testing the extraction and pipeline logic in memory before persistence.
- `rag_data/`: Directory for storing target PDF manuals (e.g., *STCW_guide_english.pdf*).

## Setup & Usage

### Dependencies
In addition to the python requirements, this project requires system-level tools for OCR and PDF processing:
- **Tesseract OCR:** Required for `pytesseract`.
- **Poppler:** Required for `pdf2image`.

### Running the Pipeline

1.  **Prepare Data:**
    Place your maritime PDF manuals in the `rag_data/` directory.

2.  **Ingest Data:**
    Run the vectorization script to process files and build the database:
    ```bash
    python vectorize.py
    ```

3.  **Query:**
    Use `Pipeline.ipynb` to search the database and generate answers to specific regulatory questions.
