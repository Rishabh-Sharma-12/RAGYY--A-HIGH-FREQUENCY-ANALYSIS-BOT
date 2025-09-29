import os
import json
import fitz
from langchain_core.runnables.base import Runnable  # PyMuPDF
from utils.log import setup_logger  # or your correct logger import
from langchain_core.runnables import RunnableLambda

TEMP_OUTPUT_DIR = "/Users/ssris/Desktop/RIMSAB/AI-MANTRA/RAG_TENDOR/temp_uploads"
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True) 

logger = setup_logger()

def pdf_to_basic_json(pdf_path, output_json_path):
    """Extracts text from each page of a PDF and saves as a JSON list."""
    import tempfile

    # Handle UploadedFile from Streamlits
    if hasattr(pdf_path, "read"):  # Likely a Streamlit UploadedFile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_path.read())
            tmp_path = tmp.name
    elif isinstance(pdf_path, str) and os.path.exists(pdf_path):
        tmp_path = pdf_path
    else:
        logger.error("❌ Invalid PDF path or file not found.")
        return []

    logger.info(f"📄 Loading PDF: {tmp_path}")
    try:
        doc = fitz.open(tmp_path)
    except Exception as e:
        logger.exception(f"Error opening PDF: {e}")
        return []

    logger.info(f"📄 Total Pages: {len(doc)}")
    
    json_data = []
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if page_num % 100 == 0:
                logger.info(f"✅ Extracted Page {page_num}")
            json_data.append({
                "page_number": page_num,
                "text": text
            })
        except Exception as e:
            logger.warning(f"⚠️ Error extracting page {page_num + 1}: {e}")
            continue
        
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Successfully written JSON to: {output_json_path}")
    except Exception as e:
        logger.exception(f"Error writing JSON: {e}")
    
    return json_data


def pdf_to_basic_json_runnable():
    return RunnableLambda(lambda inputs: _pdf_to_basic_json_runnable_impl(inputs))

def _pdf_to_basic_json_runnable_impl(inputs):
    try:
        # Read input
        pdf_path = inputs["pdf_path"]
        raw_json_path = inputs["raw_json_path"]

        logger.info(f"📥 Converting PDF to JSON: {pdf_path} -> {raw_json_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)
        
        # Run conversion
        pages = pdf_to_basic_json(pdf_path, raw_json_path)

        if not pages:
            raise ValueError("❌ PDF to JSON returned no pages!")

        logger.info(f"✅ Extracted {len(pages)} pages from {pdf_path}")

        return {
            **inputs,
            "pages": pages,
            "raw_json_path": raw_json_path
        }

    except Exception as e:
        logger.exception(f"❌ Error in pdf_to_basic_json_runnable: {e}")
        return {
            "pages": [],
            "raw_json_path": inputs.get("raw_json_path", "N/A")
        }

if __name__ == "__main__":
    input_path = input("📥 Enter the input PDF path: ").strip()
    if not os.path.exists(input_path):
        logger.error("❌ File not found.")
        exit(1)

    TEMP_OUTPUT_DIR = "/Users/ssris/Desktop/RIMSAB/AI-MANTRA/RAG_TENDOR/temp_uploads"
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(TEMP_OUTPUT_DIR, f"{file_name}.json")

    # Manual test
    result = pdf_to_basic_json(input_path, output_path)
    logger.info(f"📄 Extracted {len(result)} pages manually.")

    # Runnable test
    logger.info("🔁 Running via RunnableLambda...")
    runnable = pdf_to_basic_json_runnable()
    result = runnable.invoke({
        "pdf_path": input_path,
        "raw_json_path": output_path
    })
    logger.info(f"✅ Runnable Output: Extracted {len(result['pages'])} pages")
    logger.info(f"📄 Saved to: {result['raw_json_path']}")