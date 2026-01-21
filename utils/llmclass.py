import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

load_dotenv()

class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = "gpt-5-nano:free"):
        """
        Initialize the OpenAI-compatible client.

        Args:
            api_key: Your API Key (use "dummy" for local models like Ollama)
            base_url: The API endpoint (e.g., "http://localhost:11434/v1" for Ollama)
            model_name: The specific model to target (e.g., "llama3", "gpt-4o")
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model_name = model_name

    def construct_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Builds the prompt by combining the user query with retrieved context.
        """
        # Join chunks with a clear separator
        context_str = "\n\n---\n\n".join(context_chunks)

        prompt = f"""You are a helpful assistant for maritime regulations.
Answer the user's question based ONLY on the following context.
If the answer is not in the context, say "I don't know."

The context may contain Markdown tables. Please interpret the rows and columns accurately.

### CONTEXT:
{context_str}

### USER QUESTION:
{query}

### ANSWER:
"""
        return prompt

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Sends the prompt to the LLM and returns the response.
        """
        prompt = self.construct_prompt(query, context_chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise technical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Keep strict for RAG to avoid hallucinations
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during inference: {e}"
