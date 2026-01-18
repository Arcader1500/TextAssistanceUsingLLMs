import os
import glob
import pymupdf4llm  # pip install pymupdf4llm


def extract_markdown_from_pdfs(directory):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    processed_docs = []

    print(f"Found {len(pdf_files)} PDF files in {directory}")

    for pdf_file in pdf_files:
        print(f"Processing: {os.path.basename(pdf_file)}")
        try:
            # Returns a list of dictionaries [{'text': '...', 'metadata': {...}}, ...]
            # separating content by page automatically
            doc_data = pymupdf4llm.to_markdown(pdf_file, page_chunks=True)

            for page in doc_data:
                # Add source filename to metadata for citation
                page['metadata']['source'] = os.path.basename(pdf_file)
                processed_docs.append(page)

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    return processed_docs


if __name__ == "__main__":
    rag_data_dir = os.path.join(os.path.dirname(__file__), "rag_data")
    extracted_text = extract_markdown_from_pdfs(rag_data_dir)

    extracted_text = "\n".join(extracted_text)
    # Save to file to avoid console buffer limits if large, and also print a summary
    output_file = os.path.join(rag_data_dir, "extracted_content.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for output in extracted_text:
            f.write(output)

    print(f"Extracted dictionary length: {len(extracted_text)}")
    print(f"Saved extracted text to: {output_file}")