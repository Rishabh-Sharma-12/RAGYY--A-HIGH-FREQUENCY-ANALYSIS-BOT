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
