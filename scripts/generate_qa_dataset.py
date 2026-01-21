import json
import os
import random
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables first
load_dotenv()

# --- Configuration ---
PROCESSED_CHUNKS_FILE = "../rag_data/processed_chunks.json"
OUTPUT_FILE = "../rag_data/ragas_qa_dataset.json"
PLAN_FILE = "../rag_data/qa_generation_plan.json"
BATCH_SIZE = 5
NUM_BATCHES_TO_SELECT = 100
LLM_MODEL = "gpt-oss-120b"
RATE_LIMIT_WAIT_SECONDS = 60 # Time to wait if a rate limit error occurs

# --- Setup LLM & Chain ---
def setup_chain():
    """Initializes the LLM, prompt, and output parser."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    prompt_template = """Your task is to generate one factual question and its ground truth answer STRICTLY from the provided context. Do NOT hallucinate. Ensure the question and answer are verbose, detailed, and directly contextual to the provided text. Ignore any non-factual filler like '----------------------------------'. Return the output as a JSON object with the following keys: 'question' and 'ground_truth'.
- 'question' should be a well-formulated, verbose question based on the context.
- 'ground_truth' should be a comprehensive, detailed ground truth answer, directly extracted or summarized from the context.

Context:
{context}

Return JSON:

{{
  "question": "...",
  "ground_truth": "..."
}}"""

    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template
    )

    return prompt | llm | JsonOutputParser()

# --- Helper Functions ---
def load_processed_chunks(file_path: str) -> List[Dict[str, Any]]:
    """Loads processed chunks from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found. Please run the chunk processing script first.")

    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

def get_or_create_plan(chunks: List[Dict], batch_size: int, num_batches: int) -> List[List[Dict]]:
    """Loads existing plan or creates a new one with random batches."""
    if os.path.exists(PLAN_FILE):
        print(f"Loading existing generation plan from {PLAN_FILE}")
        with open(PLAN_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    
    print("Creating new generation plan...")
    # Group chunks into batches
    all_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    # Select random batches
    if len(all_batches) < num_batches:
        print(f"Warning: Only {len(all_batches)} batches available. Using all of them.")
        selected_batches = all_batches
    else:
        selected_batches = random.sample(all_batches, num_batches)
    
    # Save the plan
    with open(PLAN_FILE, "w", encoding='utf-8') as f:
        json.dump(selected_batches, f, indent=2)
    
    return selected_batches

def load_existing_dataset() -> List[Dict]:
    """Loads existing Q&A dataset to resume progress."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Output file exists but is empty or invalid. Starting fresh.")
    return []

def generate_dataset(chain, batches: List[List[Dict]]):
    """Generates Q&A pairs for each batch using the LLM chain, resuming from last saved state."""
    dataset = load_existing_dataset()
    
    # Determine where to resume
    completed_indices = {entry.get("batch_index") for entry in dataset if "batch_index" in entry}
    last_query_number = 0
    if dataset:
        last_query_number = max([entry.get("query_number", 0) for entry in dataset])

    total_batches = len(batches)
    print(f"Total batches to process: {total_batches}. Already completed: {len(completed_indices)}.")

    for i, batch in enumerate(batches):
        if i in completed_indices:
            continue
            
        current_query_number = last_query_number + 1
        print(f"Processing batch {i} (Query #{current_query_number})...")

        # Combine content
        combined_context = "\n---\n".join([chunk["content"] for chunk in batch])
        batch_ids = [chunk["id"] for chunk in batch]

        success = False
        while not success:
            try:
                qa_pair = chain.invoke({"context": combined_context})

                qa_entry = {
                    "query_number": current_query_number,
                    "batch_index": i,
                    "question": qa_pair.get("question"),
                    "source_chunk_ids": batch_ids,
                    "reference_contexts": [chunk["content"] for chunk in batch],
                    "ground_truth": qa_pair.get("ground_truth")
                }
                dataset.append(qa_entry)
                
                # Save immediately to allow resuming on crash
                with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
                    json.dump(dataset, f, indent=2)
                
                print(f"Successfully generated Q&A for batch {i}")
                last_query_number = current_query_number
                success = True

            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    print(f"Rate limit hit. Waiting {RATE_LIMIT_WAIT_SECONDS} seconds before retrying...")
                    time.sleep(RATE_LIMIT_WAIT_SECONDS)
                else:
                    print(f"Error generating Q&A for batch {i}: {e}. Skipping this batch.")
                    # Mark as failed/skipped by not adding to dataset but continuing loop
                    # Note: In a strict resume logic, you might want to log failures to a separate file
                    # to avoid retrying them forever. For now, we just break to skip.
                    break

def main():
    try:
        # 1. Setup
        chain = setup_chain()

        # 2. Load Data
        chunks = load_processed_chunks(PROCESSED_CHUNKS_FILE)

        # 3. Get or Create Plan (Resumable batches)
        selected_batches = get_or_create_plan(chunks, BATCH_SIZE, NUM_BATCHES_TO_SELECT)

        # 4. Generate Data (Resumable execution)
        generate_dataset(chain, selected_batches)

        print(f"Process completed. Data saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
