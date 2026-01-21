import streamlit as st
import os
import sys
import pickle
import re
import chromadb
from chromadb.utils import embedding_functions

# Add parent directory to path to allow importing utils
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.llmclass import LLM
from utils.search import HybridSearch

# Page Configuration
st.set_page_config(page_title="Maritime RAG Assistant", page_icon="⚓", layout="centered")

# Title and Caption
st.title("⚓ Maritime AI Assistant")
st.caption("Ask questions about maritime regulations and procedures.")

# Helper: Parse LLM Output for Reasoning
def parse_llm_output(text):
    pattern = r"<think>(.*?)</think>"
    # Use re.DOTALL to match across multiple lines
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        reasoning = match.group(1).strip()
        final_answer = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return reasoning, final_answer

    # # Fallback: if <think> exists but </think> is missing (incomplete response)
    # if "<think>" in text.lower():
    #     parts = re.split(r"<think>", text, flags=re.IGNORECASE)
    #     # parts[0] is before <think>, parts[1] is inside/after
    #     return parts[1], parts[0]

    return None, text


# Initialize Resources (Cached)
@st.cache_resource
def load_resources():
    # Paths based on current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "seamanuals")
    bm25_path = os.path.join(db_path, "bm25_retriever.pkl")
    
    client = chromadb.PersistentClient(path=db_path)
    # Using the updated model as per previous instructions
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
    collection = client.get_or_create_collection(name="Sea-Database", embedding_function=embedding_func)
    
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    search_engine = HybridSearch.get(collection, bm25_retriever)
    rag_generator = LLM(model_name='qwen-3-32b')
    
    return search_engine, rag_generator

try:
    search, rag = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Parse stored content for history display
            reasoning, final_answer = parse_llm_output(message["content"])
            if reasoning:
                with st.expander("Reasoning Process"):
                    st.markdown(reasoning)
            
            st.markdown(final_answer)
            
            # Display context if available
            if "context_docs" in message:
                with st.expander("View Retrieved Context Sources"):
                    for i, doc in enumerate(message["context_docs"]):
                        rank_label = f"Rank {doc.get('rank', i+1)}"
                        with st.expander(rank_label):
                            st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                            st.text(doc.get('content', ''))
        else:
            st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How do I battle fires aboard other boats?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching manuals..."):
            # 1. Retrieve Context
            docs = search(prompt)
            context_list = ["-> ".join([str(doc.get('source', 'Unknown')), str(doc.get('content', ''))]) for doc in docs]
            
            # 2. Generate Answer
            raw_answer = rag.generate_answer(prompt, context_list)
            
            # 3. Parse Answer for Reasoning
            reasoning, final_answer = parse_llm_output(raw_answer)
            
            # Display Reasoning Dropdown (if present)
            if reasoning:
                with st.expander("Reasoning Process"):
                    st.markdown(reasoning)
            
            # Display Final Answer
            st.markdown(final_answer)
            
            # Display Context Dropdown
            with st.expander("View Retrieved Context Sources"):
                for i, doc in enumerate(docs):
                    rank_label = f"Rank {doc.get('rank', i+1)}"
                    with st.expander(rank_label):
                        st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                        st.text(doc.get('content', ''))
            
            # Add assistant response to chat history (storing raw content to preserve reasoning)
            st.session_state.messages.append({
                "role": "assistant",
                "content": raw_answer,  # Store the full text containing <think> tags
                "context_docs": docs
            })
