import json
import re
import os


TEMP_OUTPUT_DIR = "/temp_uploads"
from langchain_core.runnables import Runnable, RunnableLambda
from utils.log import setup_logger

logger = setup_logger("index_extractor")

def extract_des_via_index(json_path, index_output_path):
    try:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            pdf_pages = json.load(f)
        logger.info(f"Loaded {len(pdf_pages)} pages from {json_path}")
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return []

    index_text = ""
    # Check first 15 pages for index-like content
    for page in pdf_pages[:15]:
        page_text = page.get("text", "")
        if re.search(r'\b(contents?|index|table\s+of\s+contents?)\b', page_text, re.IGNORECASE):
            index_text = page_text
            logger.info(f"Found index on page {page['page_number'] + 1}")
            break

    if not index_text:
        logger.warning("No index/content found in the first 15 pages")
        return []

    # Normalize spaces and lines
    cleaned_index = re.sub(r'[ \t]+', ' ', index_text)
    cleaned_index = re.sub(r'\n+', '\n', cleaned_index)

    # Extract index-like lines using regex
    pattern = re.compile(
        r'(?:\d+\s+)?([A-Z][^\n\d]{5,}?)\s+(\d{1,3})(?:\s*-\s*(\d{1,3}))?', 
        re.MULTILINE
    )
    matches = pattern.findall(cleaned_index)

    if not matches:
        logger.error("No valid index entries found")
        logger.debug(f"Sample cleaned index text:\n{cleaned_index[:500]}")
        return []

    logger.info(f"Found {len(matches)} index entries")

    index_data = []
    for i, match in enumerate(matches):
        desc, start_pg, end_pg = match
        try:
            start_pg = int(start_pg)
            end_pg = int(end_pg) if end_pg else start_pg

            if not (1 <= start_pg <= 1000):
                continue

            cleaned_desc = re.sub(r'\s+', ' ', desc.strip())
            cleaned_desc = re.sub(
                r'^(Annex(?:ure)?|Appendix|Enclosure|Supplement)\s*[-:]?\s*([IVX\d]*)\s*[-:]?\s*',
                r'\1 \2: ',
                cleaned_desc
            )
            if len(cleaned_desc) < 3:
                continue

            index_data.append({
                "description": cleaned_desc,
                "start": start_pg,
                "end": end_pg
            })
        except Exception as e:
            logger.warning(f"Error parsing index entry {match}: {e}")
            continue

    index_data.sort(key=lambda x: x["start"])

    try:
        with open(index_output_path, "w", encoding="utf-8") as out:
            json.dump(index_data, out, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(index_data)} entries to {index_output_path}")
    except Exception as e:
        logger.error(f"Error writing output JSON: {e}")

    return index_data

def extract_index_runnable():
    return RunnableLambda(
        lambda inputs: _extract_index_runnable_impl(inputs)
    )

def _extract_index_runnable_impl(inputs):
    try:
        input_json_path = inputs.get("raw_json_path") 
        
        if not input_json_path:
            raise ValueError("Missing input_json_path")
        
        filename = os.path.splitext(os.path.basename(input_json_path))[0]
        
        output_index_path = os.path.join(TEMP_OUTPUT_DIR, f"{filename}_index.json")

        logger.info(f"🔎 Extracting index from {input_json_path} -> {output_index_path}")

        os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
        
        index_data = extract_des_via_index(input_json_path, output_index_path)

        if not index_data:
            raise ValueError("❌ No index entries extracted.")

        logger.info(f"✅ Extracted {len(index_data)} index entries")

        return {
            **inputs,
            "index_entries": index_data,
            "index_json_path": output_index_path
        }

    except Exception as e:
        logger.exception(f"❌ Error in extract_index_runnable: {e}")
        return {
            "index_entries": [],
            "index_json_path": os.path.join(TEMP_OUTPUT_DIR, f"{filename}_index.json")
        }

# Optional CLI test
if __name__ == "__main__":
    input_path = input("📥 Enter the input JSON path: ").strip()
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(TEMP_OUTPUT_DIR, f"{filename}_index.json")

    # Direct test
    result = extract_des_via_index(input_path, output_path)
    logger.info(f"📑 Extracted {len(result)} index entries manually")

    # Runnable test
    logger.info("🔁 Running via RunnableLambda...")
    runnable = extract_index_runnable()
    output = runnable.invoke({"input_json_path": input_path})
    logger.info(f"✅ Runnable Output: Extracted {len(output['index_entries'])} entries")
    logger.info(f"📄 Saved to: {output['index_json_path']}")
