import os
import re
import glob
import hashlib
import pickle
import pdfplumber
import pytesseract
import chromadb
from typing import List, Dict, Tuple
from pdf2image import convert_from_path
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 1. SETUP PERSISTENT STORAGE
DB_PATH = "./seamanuals"
client = chromadb.PersistentClient(path=DB_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="Sea-Database",
    embedding_function=embedding_func
)

PDF_DIRECTORY = "./rag_data"


# --- Helper: Hashing for Deterministic IDs ---
def _generate_chunk_id(content: str, metadata: dict) -> str:
    """Creates a unique MD5 hash based on content and source metadata."""
    # We include source and page to distinguish same text on different pages if necessary
    unique_string = f"{content}_{metadata.get('source', '')}_{metadata.get('page', '')}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()


# --- Sub-function 1: Table Formatter (Unchanged) ---
def _table_to_markdown(table: List[List[str]]) -> str:
    if not table or len(table) < 2: return ""
    clean_table = [[str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row] for row in table]
    header = "| " + " | ".join(clean_table[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in clean_table[1:]]
    return f"\n{header}\n{separator}\n" + "\n".join(body_rows) + "\n"


# --- Sub-function 2: Hybrid Extractor (Unchanged) ---
def _extract_pdf_content(pdf_path: str) -> List[Dict]:
    extracted_pages = []
    print(f"Processing: {os.path.basename(pdf_path)}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_content = []
                tables = page.extract_tables()
                text = page.extract_text() or ""
                clean_text = text.strip()
                if len(clean_text) < 100:
                    images = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1, dpi=200)
                    if images:
                        clean_text = pytesseract.image_to_string(images[0]).strip()
                if clean_text: page_content.append(clean_text)
                if tables:
                    page_content.append("\n\n### Data Tables:\n")
                    for table in tables: page_content.append(_table_to_markdown(table))
                final_content = "\n".join(page_content)
                if len(final_content.strip()) > 0:
                    extracted_pages.append({
                        "content": final_content,
                        "metadata": {"source": os.path.basename(pdf_path), "page": i + 1}
                    })
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return extracted_pages


# --- Sub-function 3: Cleaner and Splitter ---
def _clean_and_split(raw_docs: List[Dict], chunk_size: int = 2000, chunk_overlap: int = 400) -> Tuple[
    List[str], List[Dict]]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    all_splits, all_metadatas = [], []
    for doc in raw_docs:
        text = re.sub(r'\n{3,}', '\n\n', doc['content'])
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_splits.append(chunk)
            all_metadatas.append(doc['metadata'].copy())
    return all_splits, all_metadatas


# --- Main Entry Point ---
def process_directory_for_rag(directory_path: str):
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    raw_documents = []
    for pdf_file in pdf_files:
        raw_documents.extend(_extract_pdf_content(pdf_file))

    # Step 2: Cleaning & Splitting
    final_splits, final_metadatas = _clean_and_split(raw_documents)

    # --- NEW: IN-MEMORY DEDUPLICATION ---
    unique_chunks = {}  # Key: hash, Value: (content, metadata)

    for content, meta in zip(final_splits, final_metadatas):
        chunk_id = _generate_chunk_id(content, meta)
        # If the hash exists, this simply overwrites with the same data,
        # effectively deduplicating the batch before it hits Chroma.
        unique_chunks[chunk_id] = (content, meta)

    # Extract back into lists for Chroma
    deduped_ids = list(unique_chunks.keys())
    deduped_docs = [v[0] for v in unique_chunks.values()]
    deduped_metas = [v[1] for v in unique_chunks.values()]

    print(f"Total chunks created: {len(final_splits)}")
    print(f"Unique chunks after deduplication: {len(deduped_ids)}")

    # Batching the Upsert (using the deduped lists)
    batch_size = 100
    for i in range(0, len(deduped_ids), batch_size):
        end = min(i + batch_size, len(deduped_ids))
        collection.upsert(
            documents=deduped_docs[i:end],
            metadatas=deduped_metas[i:end],
            ids=deduped_ids[i:end]
        )

    # Step 3: BM25 Indexing (Using the deduped data)
    print("Building BM25 index...")
    bm25_documents = [
        Document(page_content=chunk, metadata={**meta, "id": chunk_id})
        for chunk_id, chunk, meta in zip(deduped_ids, deduped_docs, deduped_metas)
    ]

    bm25_retriever = BM25Retriever.from_documents(bm25_documents)

    BM25_PATH = os.path.join(DB_PATH, "bm25_retriever.pkl")
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"✅ Hybrid Database Synchronized. Total unique chunks: {collection.count()}")


if __name__ == "__main__":
    # Optional: Wipe collection before first run to clear old duplicates
    process_directory_for_rag(PDF_DIRECTORY)