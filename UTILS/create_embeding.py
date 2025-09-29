from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableLambda

# 🔁 Keep the model loaded globally to avoid reloading every time
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(chunks, model=None):
    """Generates embeddings for a list of text chunks."""
    model = model or _model
    texts = [chunk["text"] for chunk in chunks]
    
    try:
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        print(f"[❌] Error in embedding text: {e}")
        return None

def embed_text_chain_fn(inputs):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunks = inputs.get("chunks")
    if not chunks:
        raise ValueError("[embed_text_chain_fn] ❌ No 'chunks' found in inputs")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Create embedding entries with metadata
    embed_data = [
        {
            "text": text,
            "embedding": embedding.tolist(),  # convert numpy to list
            "metadata": chunk.get("metadata", {})
        }
        for text, embedding, chunk in zip(texts, embeddings, chunks)
    ]

    return {
        **inputs,
        "embed_data": embed_data,  # This should be used by save_to_json
    }

def embed_query_chain_fn(inputs):
    query = inputs.get("query", "")
    if not query:
        raise ValueError("[embed_query_chain_fn] ❌ No 'query' found in inputs")

    try:
        embedding = _model.encode(query, convert_to_numpy=True)
        inputs["query_vector"] = embedding.tolist()
        return inputs
    except Exception as e:
        raise RuntimeError(f"[❌] Error in embedding query: {e}")

def embed_text_runnable():
    
    return RunnableLambda(embed_text_chain_fn)

def embed_query_runnable():
    return RunnableLambda(embed_query_chain_fn)

# from langchain_core.runnables import RunnableLambda
# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# load_dotenv("/Users/ssris/Desktop/RIMSAB/AI-MANTRA/RAG_TENDOR/.env")

# # ------------------------------
# # NEW: Initialize OpenAI v1 client
# # ------------------------------
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # UPDATED: Use OpenAI() instead of old openai.Embedding.create

# # ------------------------------
# # Text embedding
# # ------------------------------
# def embed_text_chain_fn(inputs):
#     chunks = inputs.get("chunks", [])
#     if not chunks:
#         return {**inputs, "embed_data": []}

#     embed_data = []
#     for chunk in chunks:
#         text = chunk["text"]
#         try:
#             # UPDATED: Use client.embeddings.create(...) instead of openai.Embedding.create
#             response = client.embeddings.create(
#                 input=text,
#                 model="text-embedding-3-small"  # smallest & cheapest model
#             )
#             embedding = response.data[0].embedding  # UPDATED: new v1 response structure
#         except Exception as e:
#             raise RuntimeError(f"[❌] Error embedding chunk: {e}")

#         embed_data.append({
#             "text": text,
#             "embedding": embedding,
#             "metadata": chunk.get("metadata", {})
#         })

#     return {**inputs, "embed_data": embed_data}

# # ------------------------------
# # Query embedding
# # ------------------------------
# def embed_query_chain_fn(inputs):
#     query = inputs.get("query", "")
#     if not query:
#         raise ValueError("[embed_query_chain_fn] ❌ No 'query' found in inputs")
    
#     try:
#         # UPDATED: Use client.embeddings.create(...) for query
#         response = client.embeddings.create(
#             input=query,
#             model="text-embedding-3-small"
#         )
#         inputs["query_vector"] = response.data[0].embedding  # UPDATED: new response structure
#         return inputs
#     except Exception as e:
#         raise RuntimeError(f"[❌] Error embedding query: {e}")

# # ------------------------------
# # Runnables
# # ------------------------------
# def embed_text_runnable():
#     return RunnableLambda(embed_text_chain_fn)

# def embed_query_runnable():
#     return RunnableLambda(embed_query_chain_fn)
