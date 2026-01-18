import os
import glob
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdfs(directory):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    all_text = ""
    
    print(f"Found {len(pdf_files)} PDF files in {directory}")
    
    for pdf_file in pdf_files:
        print(f"Processing: {os.path.basename(pdf_file)}")
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            text = "".join([page.page_content for page in pages])
            all_text += f"\n\n--- START OF {os.path.basename(pdf_file)} ---\n\n"
            all_text += text
            all_text += f"\n\n--- END OF {os.path.basename(pdf_file)} ---\n\n"
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            
    return all_text

if __name__ == "__main__":
    rag_data_dir = os.path.join(os.path.dirname(__file__), "rag_data")
    extracted_text = extract_text_from_pdfs(rag_data_dir)
    
    # Save to file to avoid console buffer limits if large, and also print a summary
    output_file = os.path.join(rag_data_dir, "extracted_content.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)
        
    print(f"Extracted dictionary length: {len(extracted_text)}")
    print(f"Saved extracted text to: {output_file}")
