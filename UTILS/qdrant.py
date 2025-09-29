import json
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import Runnable
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableLambda


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.log import setup_logger
from utils.llm import get_groq_response, build_prompt

# Setup
load_dotenv()

logger = setup_logger("qdrant_logger")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def upload_in_batches(client, collection_name, points, batch_size=50):
    """Uploads points to Qdrant in batches."""
    total_batches = (len(points) + batch_size - 1) // batch_size  # Ceiling division

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(collection_name=collection_name, points=batch)
        except Exception as e:
            logger.error(f"Error uploading batch {i // batch_size + 1}: {e}")

    logger.info(f"✅ Uploaded all {total_batches} batches to collection '{collection_name}'")


def upload_embed_to_qdrant(json_path, collection_name, qdrant_url, qdrant_api_key=None, vector_size=384):
    """
    Uploads embedded chunks from JSON to a Qdrant collection.
    Only uploads if collection doesn't exist or has different vector parameters.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} chunks from {json_path}")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Check if collection exists and has the same configuration
    if client.collection_exists(collection_name):
        collection_info = client.get_collection(collection_name)
        existing_config = collection_info.config.params.vectors
        
        # Check if vector size and distance match
        if (hasattr(existing_config, 'size') and 
            existing_config.size == vector_size and 
            existing_config.distance == Distance.COSINE):
            
            existing_count = client.count(collection_name=collection_name, exact=True).count
            if existing_count > 0:
                logger.info(f"✅ Collection '{collection_name}' already exists with {existing_count} points and matching configuration. Skipping upload.")
                return client
            else:
                logger.info(f"📝 Collection '{collection_name}' exists with matching configuration but is empty. Proceeding to upload.")
        else:
            logger.warning(f"⚠️ Collection '{collection_name}' exists but with different vector configuration. Skipping upload.")
            logger.warning(f"   Existing: size={existing_config.size}, distance={existing_config.distance}")
            logger.warning(f"   Requested: size={vector_size}, distance={Distance.COSINE}")
            return client
    else:
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"🆕 Created collection '{collection_name}' with vector size {vector_size}")
    
    # Upload points
    points = [
        PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={"text": chunk["text"], "metadata": chunk["metadata"]}
        )
        for i, chunk in enumerate(data)
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.info(f"✅ Uploaded {len(points)} points to collection '{collection_name}'")
    
    return client
 
def search_qdrant(query_text, client: QdrantClient, collection_name: str, top_k: int = 5):
    """
    Searches Qdrant for the most similar chunks to the query text.
    """
    query_vector = model.encode([query_text], show_progress_bar=False, convert_to_numpy=True)
    result = client.search(
        collection_name=collection_name,
        query_vector=query_vector[0].tolist(),
        limit=top_k
    )

    logger.info(f"Retrieved {len(result)} results for query: '{query_text}'")
    # Log matched chunk and page numbers
    for i, res in enumerate(result, 1):
        metadata = res.payload.get("metadata", {})
        chunk_id = metadata.get("chunk_index", "N/A")
        page_num = metadata.get("page_number", "N/A")
        score = res.score
        logger.info(f"Match {i}: Chunk #{chunk_id}, Page #{page_num}, Score: {score:.4f}")
        
    return result


def rag_query(client, collection_name, query_text, top_k=5):
    """
    Full RAG workflow: Qdrant search -> Prompt build -> LLM response.
    """
    logger.info(f"Running RAG query for: {query_text}")
    try:
        chunks = search_qdrant(query_text, client, collection_name, top_k=top_k)
        logger.debug(f"Chunks received for prompt: {[res.payload['metadata'] for res in chunks]}")
        if not chunks:
            logger.warning("No relevant chunks found.")
            return "No relevant information found in the document."

        prompt = build_prompt(query_text, chunks)
        logger.debug(f"Generated prompt:\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}")
        response = get_groq_response(prompt)
        context_texts = [chunk.payload["text"] for chunk in chunks]
        logger.info("LLM response generated successfully.")
        return {
            "response":response,
            "contexts":context_texts
        }

    except Exception as e:
        logger.exception(f"Error in RAG query: {str(e)}")
        return f"Error generating response: {str(e)}"

#-------------#-------------#-------------#-------------#-------------#-------------

# Runnables 

#-------------#-------------#-------------#-------------#-------------#-------------


def upload_qdrant_runnable():
    return RunnableLambda(lambda inputs: _upload_qdrant_runnable_impl(inputs))

def _upload_qdrant_runnable_impl(inputs):
    try:
        client = upload_embed_to_qdrant(
            json_path=inputs["embed_json_path"],
            collection_name=inputs["collection_name"],
            qdrant_url=inputs.get("qdrant_url") or os.getenv("QDRANT_URL"),
            qdrant_api_key=inputs.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY"),
            vector_size=inputs.get("vector_size", 384)
        )
        return {
            **inputs,
            "qdrant_client": client
        }
    except Exception as e:
        logger.exception(f"❌ Error in upload_qdrant_runnable: {e}")
        return {
            **inputs,
            "qdrant_client": None,
            "error": str(e)
        }

def rag_query_runnable():
    return RunnableLambda(lambda inputs: _rag_query_runnable_impl(inputs))

def _rag_query_runnable_impl(inputs):
    try:
        response = rag_query(
            client=inputs["qdrant_client"],
            collection_name=inputs["collection_name"],
            query_text=inputs["query"],
            top_k=inputs.get("top_k", 5)
        )
        return {
            **inputs,
            "response": response["response"],
            "contexts": response["contexts"]
        }
    except Exception as e:
        logger.exception(f"❌ Error in rag_query_runnable: {e}")
        return {
            **inputs,
            "response": f"[❌] RAG failed: {str(e)}",
            "contexts": []
        }
        
        
if __name__ == "__main__":
    if len(sys.argv) < 4:
        logger.error("Usage: python qdrant.py <json_path> <collection_name> <qdrant_url> [qdrant_api_key]")
        sys.exit(1)

    json_path = sys.argv[1]
    collection_name = sys.argv[2]
    qdrant_url = sys.argv[3]
    qdrant_api_key = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        logger.info("Starting Qdrant upload workflow...")
        client = upload_embed_to_qdrant(json_path, collection_name, qdrant_url, qdrant_api_key)

        test_query = "What are the main requirements?"
        logger.info(f"Testing query: '{test_query}'")

        search_results = search_qdrant(test_query, client, collection_name)
        answer = rag_query(client, collection_name, test_query)

        logger.info(f"Final Answer:\n{answer}")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)