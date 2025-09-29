import os
import json
from langchain_core.runnables import RunnableLambda
from utils.log import setup_logger
def save_json(data, path):
    """Saves data as JSON to the specified path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding="utf-8") as out_file:
            json.dump(data, out_file, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved {len(data)} chunks to {path}")
    except Exception as e:
        print(f"[❌] Error saving JSON to {path}: {e}")
        

# LangChain-compatible runnable
def save_json_runnable():
    return RunnableLambda(lambda inputs: _save_json_runnable_impl(inputs))


def _save_json_runnable_impl(inputs):
    try:
        data = inputs.get("embed_data") or inputs.get("data") or inputs.get("chunks")
        path = inputs.get("embed_json_path") or inputs.get("path")

        if not data or not path:
            raise ValueError("[save_to_json] No data or path provided.")

        save_json(data, path)

        return {
            **inputs,
            "embed_json_path": path
        }

    except Exception as e:
        from utils.log import setup_logger
        logger = setup_logger("save_json_logger")
        logger.exception(f"Error in save_json_runnable: {e}")
        return {
            **inputs,
            "saved_path": None,
            "error": str(e)
        }