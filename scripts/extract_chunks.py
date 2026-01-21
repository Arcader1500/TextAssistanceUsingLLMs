import hashlib
import json
import os
import glob
import re
from typing import List, Dict

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_data_dir = os.path.join(os.path.dirname(current_dir), 'rag_data')
output_filename = os.path.join(current_dir, '../rag_data/processed_chunks.json')

# --- Helper Functions for Enhanced Extraction ---

def _table_to_markdown(table: List[List[str]]) -> str:
    """Converts a parsed table into a Markdown table string."""
    if not table or len(table) < 2: return ""
    clean_table = [[str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row] for row in table]
    header = "| " + " | ".join(clean_table[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(clean_table[0])) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in clean_table[1:]]
    return f"\n{header}\n{separator}\n" + "\n".join(body_rows) + "\n"

def _extract_pdf_content_to_documents(pdf_path: str) -> List[Document]:
    """Extracts text and tables from PDF using pdfplumber, returning LangChain Documents."""
    extracted_docs = []
    print(f"Extracting content from: {os.path.basename(pdf_path)}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_content = []
                tables = page.extract_tables()
                text = page.extract_text() or ""
                clean_text = text.strip()
                
                # OCR fallback for pages with very little text (e.g., scanned images)
                if len(clean_text) < 100:
                    try:
                        images = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1, dpi=200)
                        if images:
                            clean_text = pytesseract.image_to_string(images[0]).strip()
                    except Exception as ocr_e:
                        # Fail silently or log if OCR tools aren't installed/configured
                        print(f"OCR warning for page {i+1} of {os.path.basename(pdf_path)}: {ocr_e}")

                if clean_text: 
                    page_content.append(clean_text)
                
                if tables:
                    page_content.append("\n\n### Data Tables:\n")
                    for table in tables: 
                        page_content.append(_table_to_markdown(table))
                
                final_content = "\n".join(page_content)
                
                # Clean up excessive newlines
                final_content = re.sub(r'\n{3,}', '\n\n', final_content)

                if len(final_content.strip()) > 0:
                    extracted_docs.append(Document(
                        page_content=final_content,
                        metadata={"source": os.path.basename(pdf_path), "page": i + 1}
                    ))
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return extracted_docs

# --- Main Logic ---

# Load existing processed chunks if available
processed_chunks_list = []
existing_ids = set()

if os.path.exists(output_filename):
    try:
        with open(output_filename, 'r', encoding='utf-8') as f:
            processed_chunks_list = json.load(f)
            # Create a set of existing IDs to avoid duplicates
            for chunk in processed_chunks_list:
                existing_ids.add(chunk.get('id'))
        print(f"Loaded {len(processed_chunks_list)} existing chunks.")
    except Exception as e:
        print(f"Could not load existing file: {e}. Starting fresh.")

# Get all PDF files in rag_data
pdf_files = glob.glob(os.path.join(rag_data_dir, '*.pdf'))

if not pdf_files:
    print(f"No PDF files found in {rag_data_dir}")
else:
    print(f"Found {len(pdf_files)} PDF files to process.")

# Initialize RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=850, # Updated to 850 for consistency
    chunk_overlap=120 # Updated to 120 for consistency
)

# Iterate through each PDF file
for pdf_file_path in pdf_files:
    
    try:
        # Use the custom extraction function instead of PyPDFLoader
        docs = _extract_pdf_content_to_documents(pdf_file_path)
        
        # Split the loaded documents into chunks
        chunks = splitter.split_documents(docs)
        
        new_chunks_count = 0
        for i, chunk in enumerate(chunks):
            # Generate a unique hash ID for the chunk.page_content
            hash_id = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
            
            # Avoid duplicates based on hash_id
            if hash_id not in existing_ids:
                # Create a dictionary for each processed chunk
                processed_chunk = {
                    "content": chunk.page_content,
                    "id": hash_id,
                    "metadata": chunk.metadata
                }
                
                # Append to list and update ID set
                processed_chunks_list.append(processed_chunk)
                existing_ids.add(hash_id)
                new_chunks_count += 1
        
        print(f"Added {new_chunks_count} new chunks from {os.path.basename(pdf_file_path)}")
        
    except Exception as e:
        print(f"Error processing {pdf_file_path}: {e}")

# Save the updated list of processed chunks to a JSON file
with open(output_filename, "w", encoding='utf-8') as f:
    json.dump(processed_chunks_list, f, indent=2)

print(f"Total processed chunks: {len(processed_chunks_list)} saved to {output_filename}")