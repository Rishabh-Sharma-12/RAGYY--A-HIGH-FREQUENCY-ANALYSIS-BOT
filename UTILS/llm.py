# llm.py
import os
from groq import Groq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate

load_dotenv(dotenv_path="/Users/ssris/Desktop/RIMSAB/AI-MANTRA/RAG_TENDOR/.env")


def init_groq_client():
    """
    Initializes and returns a Groq client using the API key from environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables")
    return Groq(api_key=api_key)


def get_groq_response(prompt, model="llama3-70b-8192", temperature=0.2):
    """
    Get response from Groq's LLM using the official client.
    """
    try:
        client = init_groq_client()
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes and answers based only on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[❌] Error getting LLM response: {str(e)}"

    
def build_prompt(query, chunks, system_message=None, max_words=4000):
    """
    Builds a detailed prompt by injecting retrieved chunks into a comprehensive instruction format.
    The prompt includes explicit instructions, context formatting, and answer guidelines.
    """
    context_texts = []
    total_words = 0

    for idx, chunk in enumerate(chunks):
        # Extract text and metadata
        if hasattr(chunk, "payload"):
            text = chunk.payload.get("text", "")
            metadata = chunk.payload.get("metadata", {})
        elif isinstance(chunk, dict):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
        else:
            text, metadata = str(chunk), {}

        # Get chunk index and page number, defaulting if not present
        chunk_id = metadata.get("chunk_index", idx)
        page_num = metadata.get("page_number", "N/A")

        words = text.split()
        if total_words + len(words) > max_words:
            break

        # Include chunk metadata inline
        context_texts.append(f"[Chunk {chunk_id}, Page {page_num}]\n{text.strip()}")
        total_words += len(words)

    context_block = "\n\n".join(context_texts)
    

    # Default system instructions with citation guidance
    system_message = system_message or (
        "You are a highly knowledgeable assistant. "
        "Use only the provided context to answer the user's question. "
        "When referencing information, always cite it using the chunk metadata, e.g., [Chunk X, Page Y]. "
        "If the answer cannot be found in the context, reply with: 'I don't know based on the provided information.' "
        "Provide detailed and enriched paragraph-style answers."
    )


    prompt = f"""
{system_message}

-------------------- CONTEXT START --------------------
Below are excerpts from the document for reference.

{context_block}
--------------------- CONTEXT END ---------------------

Instructions:
- Read the context above thoroughly.
- Answer the question in a detailed paragraph format.
- Cite chunks explicitly using the [Chunk X, Page Y] labels provided in the context.
- Use only the information from the context.
- If the answer is not present, say: "I don't know based on the provided information."
- At the end of your response, if applicable, include a relevance or confidence score.

Question:
{query}

Answer:
"""
    return prompt