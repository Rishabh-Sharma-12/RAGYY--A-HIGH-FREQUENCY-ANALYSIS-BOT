from concurrent.futures import process
import os 
from datetime import date, datetime
from turtle import title
from fastapi import FastAPI ,UploadFile,File,Form
import collections
from fastapi.responses import JSONResponse

from config import input_dict, processing_time, rag_pipeline, query_pipe, logger, result

app=FastAPI(title="tendor-Bot RAG API")

STATE = {
    "qdrant_client": None,
    "collection_name": None,
    "document_processed": False,
    "processing_stats": {},
    "chat_history": collections.deque(maxlen=20),
}

@app.post("/process_document/")
async def process_document(file:UploadFile = File(...)):
    try:
        os.makedirs("temp_upload",exist_ok=True)
        file_path=os.path.join("temp_uploads",file.filename)
        
        with open(file_path,"wb") as f:
            f.write(await file.read())
        file_name=os.path.splittext(file.filename)[0]
        start_time=datetime.now()
        
        input_dict = {
            "pdf_path": file_path,
            "raw_json_path": f"temp_uploads/{file_name}.json",
            "embed_json_path": f"temp_uploads/{file_name}_chunks.json",
            "index_json_path": f"temp_uploads/{file_name}_index.json",
            "source_name": file_name,
            "doc_date": datetime.now().strftime("%B %Y"),
            "title": file_name.replace("_", " ").title(),
            "chunk_size": 800,
            "chunk_overlap": 160,
            "collection_name": f"{file_name}_collection"
        }
        result=rag_pipeline.invoke(input_dict)
        processing_time=(datetime.now()-start_time).total_seconds()
        
        STATE["qdrant_client"] = result["qdrant_client"]
        STATE["collection_name"] = result["collection_name"]
        STATE["document_processed"] = True
        STATE["processing_stats"] = {
            "processing_time": processing_time,
            "chunks_created": result.get("chunks_count", "N/A"),
            "embeddings_generated": result.get("embeddings_count", "N/A"),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return {"message":"Document processed sucessfully", "stats": STATE["processing_stats"]}
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    if not STATE["document_processed"]:
        return JSONResponse(status_code=400, content={"error": "No document processed yet"})
    try:
        query_inputs={
            "qdrant_client": STATE["qdrant_client"],
            "collection_name": STATE["collection_name"],
            "query": query,
            "history": [(q, a) for q, a in STATE["chat_history"]]
        }
        start_time = datetime.now()
        response = query_pipe.invoke(query_inputs)
        response_time = (datetime.now() - start_time).total_seconds()

        answer = response.get("response", "")
        contexts = response.get("contexts", [])
        STATE["chat_history"].append((query,answer))
        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "trace": {
                "response_time": response_time,
                "contexts_found": len(contexts),
                "query_length": len(query),
                "response_length": len(answer),
                "collection": STATE["collection_name"]
            }
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status/")
async def status():
    return {
        "document_processed": STATE["document_processed"],
        "stats": STATE["processing_stats"],
        "messages": len(STATE["chat_history"])
    }