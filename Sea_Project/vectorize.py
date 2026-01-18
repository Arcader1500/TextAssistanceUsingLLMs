import os
import re
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Tuple
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# 1. SETUP PERSISTENT STORAGE
# This folder will be created on your disk and house your data
DB_PATH = "./seamanuals"
client = chromadb.PersistentClient(path=DB_PATH)

# Use a standard embedding model (runs locally on CPU/GPU)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or load the collection
collection = client.get_or_create_collection(
    name="Sea-Database",
    embedding_function=embedding_func
)

# 2. PROCESSING PIPELINE
PDF_DIRECTORY = "./rag_data"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# --- Sub-function 1: Table Formatter ---
def _table_to_markdown(table: List[List[str]]) -> str:
    """Converts a raw list-of-lists table into a Markdown string."""
    if not table or len(table) < 2:
        return ""

    # Clean cell content (remove newlines within cells)
    clean_table = [[str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row] for row in table]

    # Construct Markdown parts
    header = "| " + " | ".join(clean_table[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"

    body_rows = []
    for row in clean_table[1:]:
        body_rows.append("| " + " | ".join(row) + " |")

    return f"\n{header}\n{separator}\n" + "\n".join(body_rows) + "\n"

# --- Sub-function 2: Hybrid Extractor (OCR + Digital) ---
def _extract_pdf_content(pdf_path: str) -> List[Dict]:
    """
    Extracts text and tables from a single PDF.
    Uses OCR fallback if page text is insufficient (< 100 chars).
    """
    extracted_pages = []
    print(f"Processing: {os.path.basename(pdf_path)}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_content = []

                # 1. Attempt Standard Extraction
                tables = page.extract_tables()
                text = page.extract_text() or ""
                clean_text = text.strip()

                # 2. Check if OCR is needed (Hybrid Approach)
                if len(clean_text) < 100:
                    # Render page to image
                    # Note: Linux/WSL requires poppler-utils installed
                    images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, dpi=200)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        clean_text = ocr_text.strip()

                if clean_text:
                    page_content.append(clean_text)

                # 3. Format Tables (if found digitally)
                if tables:
                    page_content.append("\n\n### Data Tables:\n")
                    for table in tables:
                        md_table = _table_to_markdown(table)
                        page_content.append(md_table)

                # 4. Compile Page Result
                final_content = "\n".join(page_content)

                # Only add pages that actually have content
                if len(final_content.strip()) > 0:
                    extracted_pages.append({
                        "content": final_content,
                        "metadata": {
                            "source": os.path.basename(pdf_path),
                            "page": i + 1
                        }
                    })

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return extracted_pages

# --- Sub-function 3: Cleaner and Splitter ---
def _clean_and_split(raw_docs: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[str], List[Dict]]:
    """
    Cleans the extracted text and splits it into chunks while preserving metadata.
    """
    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_splits = []
    all_metadatas = []

    for doc in raw_docs:
        text = doc['content'] # Note: key matches extraction output
        metadata = doc['metadata']

        # 1. Normalization: Replace 3+ newlines with 2 to fix spacing issues
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 2. Split
        chunks = text_splitter.split_text(text)

        # 3. Align Metadata
        for chunk in chunks:
            all_splits.append(chunk)
            all_metadatas.append(metadata.copy())

    return all_splits, all_metadatas

# --- Main Entry Point ---
def process_directory_for_rag(directory_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Main function to process all PDFs in a directory.

    Returns:
        tuple: (List of text chunks, List of corresponding metadata dicts)
    """
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {directory_path}")

    raw_documents = []
    ids = []

    # Step 1: Extraction
    for i, pdf_file in enumerate(pdf_files):
        pages = _extract_pdf_content(pdf_file)
        raw_documents.extend(pages)
        ids.append(f"{os.path.basename(pdf_file)}_{i}")

    print(f"Total pages extracted: {len(raw_documents)}")

    # Step 2: Cleaning & Splitting
    final_splits, final_metadatas = _clean_and_split(raw_documents)

    print(f"Total chunks created: {len(final_splits)}")

    # Generate unique IDs for every chunk
    ids = [f"id_{i}" for i in range(len(final_splits))]

    batch_size = 100
    for i in range(0, len(final_splits), batch_size):
        batch_end = min(i + batch_size, len(final_splits))
        collection.add(
            documents=final_splits[i:batch_end],
            metadatas=final_metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )

if __name__ == "__main__":
    process_directory_for_rag(PDF_DIRECTORY)
    print("✅ Vectorization complete. Database saved to disk.")