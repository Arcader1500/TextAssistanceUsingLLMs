import arxiv
import os
import requests

def download_arxiv_pdfs(query, max_results=5, download_path="./rag_data"):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Search for papers
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    print(f"🔍 Searching for '{query}'...")

    for result in search.results():
        # Clean filename: remove special characters
        clean_title = "".join(x for x in result.title if x.isalnum() or x in " -_").strip()
        filename = f"{result.entry_id.split('/')[-1]}_{clean_title[:50]}.pdf"
        file_path = os.path.join(download_path, filename)

        print(f"📥 Downloading: {result.title}")
        result.download_pdf(dirpath=download_path, filename=filename)


if __name__ == "__main__":
    # Example: Download 5 papers about "Large Language Models"
    download_arxiv_pdfs("Large Language Models", max_results=5)