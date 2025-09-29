import re
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.log import setup_logger
from langchain_core.runnables import RunnableLambda
logger = setup_logger()
TEMP_OUTPUT_DIR = "RAG_TENDOR/temp_uploads"


def chunk_pages_to_embedding_ready_format(
    pages, 
    source_name="Unknown", 
    doc_date="Unknown", 
    title="Untitled", 
    index_data=None,
    chunk_size=2500,
    chunk_overlap=400
):
    """
    ENHANCED: More accurate chunk-to-page mapping and section assignment
    """
    # Validate input
    if not pages:
        logger.error("No pages provided")
        return []
    
    first_page_text = pages[0]["text"] if pages else ""
    title_match = re.search(r"(?i)(request for proposal.*|tender.*|rfp.*|bid document.*)", first_page_text)
    if title_match:
        dynamic_title = title_match.group(0).strip()
    else:
        lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
        dynamic_title = lines[0] if lines else title

    # Build page mapping with actual character positions
    page_start_offsets = []
    page_end_offsets = []
    page_texts = []
    
    cumulative_length = 0
    for i, page in enumerate(pages):
        page_text = page["text"]
        page_texts.append(page_text)
        
        page_start_offsets.append(cumulative_length)
        cumulative_length += len(page_text)
        page_end_offsets.append(cumulative_length)
        
        # Add separator length (2 chars for "\n\n")
        if i < len(pages) - 1:  # Don't add separator after last page
            cumulative_length += 2

    all_text = "\n\n".join(page_texts)
    
    # Enhanced text splitter with better separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    # Split text and get documents with metadata
    text_chunks = splitter.split_text(all_text)
    
    # Prepare index sections
    index_sections = []
    if index_data:
        index_sections = sorted(index_data, key=lambda x: x["start"])
        logger.info(f"Using {len(index_sections)} index sections for mapping")

    # Estimate visible page offset (keep for debugging, but no log)
    visible_page_offset = 0
    if index_sections:
        visible_start = index_sections[0]["start"]
        estimated_internal_start = None
        for i, page in enumerate(pages):
            if visible_start in range(page_start_offsets[i], page_end_offsets[i]):
                estimated_internal_start = i + 1
                break
        if estimated_internal_start:
            visible_page_offset = visible_start - estimated_internal_start
        else:
            visible_page_offset = visible_start - 1

    chunk_doc = []
    current_position = 0
    
    for i, chunk_text in enumerate(text_chunks):
        # Find the actual position of this chunk in the full text
        chunk_start_pos = all_text.find(chunk_text, current_position)
        if chunk_start_pos == -1:
            # If exact match fails, use approximate position
            chunk_start_pos = current_position
        
        # Update current position for next iteration
        current_position = chunk_start_pos + len(chunk_text)
        
        # Find which page this chunk belongs to
        chunk_page_num = 1  # Default to page 1
        for page_idx in range(len(page_start_offsets)):
            if (page_start_offsets[page_idx] <= chunk_start_pos < 
                (page_end_offsets[page_idx] + (2 if page_idx < len(pages) - 1 else 0))):
                chunk_page_num = page_idx + 1  # Convert to 1-based
                break
        
        # Alternative method: if chunk spans multiple pages, find the dominant page
        chunk_end_pos = chunk_start_pos + len(chunk_text)
        pages_spanned = []
        for page_idx in range(len(page_start_offsets)):
            page_start = page_start_offsets[page_idx]
            page_end = page_end_offsets[page_idx] + (2 if page_idx < len(pages) - 1 else 0)
            
            # Check if chunk overlaps with this page
            if not (chunk_end_pos <= page_start or chunk_start_pos >= page_end):
                overlap_start = max(chunk_start_pos, page_start)
                overlap_end = min(chunk_end_pos, page_end)
                overlap_length = overlap_end - overlap_start
                pages_spanned.append((page_idx + 1, overlap_length))
        
        # Use the page with the most overlap
        if pages_spanned:
            chunk_page_num = max(pages_spanned, key=lambda x: x[1])[0]

        # Find appropriate section description
        section_desc = ""
        if index_sections:
            # Method 1: Direct range match
            for section in index_sections:
                if section["start"] <= chunk_page_num <= section["end"]:
                    section_desc = section["description"]
                    break
            
            # Method 2: Find the most recent applicable section
            if not section_desc:
                applicable = [s for s in index_sections if s["start"] <= chunk_page_num]
                if applicable:
                    # If there are multiple applicable sections, prefer the one with closer start page
                    section_desc = max(applicable, key=lambda x: x["start"])["description"]
            
            # Method 3: If page is before all sections, use first section
            if not section_desc and index_sections and chunk_page_num < index_sections[0]["start"]:
                section_desc = index_sections[0]["description"]
            
            # Method 4: If page is after all sections, use last section  
            if not section_desc and index_sections and chunk_page_num > index_sections[-1]["end"]:
                section_desc = index_sections[-1]["description"]

        base_metadata = {
            "source": source_name,
            "doc_date": doc_date,
            "title": dynamic_title,
            "chunk_index": i,
            "page_number": chunk_page_num,
            "char_start": chunk_start_pos,
            "char_end": chunk_start_pos + len(chunk_text),
            "description": section_desc
        }
        
        chunk_doc.append({
            "id": f"chunk_{i}",
            "text": chunk_text,
            "metadata": base_metadata
        })
        
        # Only log the first chunk for sanity check
        if i == 0:
            logger.info(f"First chunk: Page {chunk_page_num}, Chars {chunk_start_pos}-{chunk_start_pos + len(chunk_text)}, Section: '{section_desc[:50]}{'...' if len(section_desc) > 50 else ''}'")
    
    # Summary statistics (log only total chunks and number of sections)
    logger.info(f"Total chunks created: {len(chunk_doc)}")
    if index_sections:
        logger.info(f"Number of index sections: {len(index_sections)}")
    
    return chunk_doc

# -------------------------------
# Runnable Wrapper
# -------------------------------
def chunking_runnable():
    return RunnableLambda(lambda inputs: _chunking_runnable_impl(inputs))

def _chunking_runnable_impl(inputs):
    try:
        raw_json_path = inputs.get("raw_json_path")
        index_json_path = inputs.get("index_json_path")

        print(f"[DEBUG] Input keys in chunking: {list(inputs.keys())}")

        # Ensure both required paths are present
        if not raw_json_path or not index_json_path:
            raise ValueError(f"[chunking.py] ❌ Missing 'raw_json_path' or 'index_json_path' in inputs: {list(inputs.keys())}")

        # Load the full document JSON
        with open(raw_json_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        # Load index if it exists
        index_data = []
        if os.path.exists(index_json_path):
            with open(index_json_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            logger.warning(f"[chunking.py] ⚠️ Index file not found at {index_json_path}. Proceeding without index.")

        # Perform chunking
        chunks = chunk_pages_to_embedding_ready_format(
            pages,
            source_name=inputs.get("source_name", "Unknown"),
            doc_date=inputs.get("doc_date", "Unknown"),
            title=inputs.get("title", "Untitled"),
            index_data=index_data,
            chunk_size=inputs.get("chunk_size", 2500),
            chunk_overlap=inputs.get("chunk_overlap", 400)
        )

        logger.info(f"✅ Chunking complete. Total chunks: {len(chunks)}")

        return {
            **inputs,  # ✅ Carry forward everything to next stage
            "chunks": chunks
        }

    except Exception as e:
        logger.exception("❌ Failed in chunking runnable")
        return {
            **inputs,  # Still pass all context for downstream logging/debugging
            "chunks": []
        }


# -------------------------------
# CLI Entrypoint
# -------------------------------
# -------------------------------
# CLI Entrypoint (with hardcore input)
# -------------------------------
if __name__ == "__main__":
    print("🔧 Chunker CLI - Ready to slice and dice documents!")

    basic_json_path = input("📄 Enter path to BASIC JSON (required): ").strip()
    if not os.path.exists(basic_json_path):
        logger.error(f"❌ File not found: {basic_json_path}")
        exit(1)

    index_json_path = input("🧾 Enter path to INDEX JSON (optional): ").strip()
    if index_json_path and not os.path.exists(index_json_path):
        logger.warning("⚠️ Index file not found. Proceeding without index.")
        index_json_path = None

    source_name = input("🏷️  Enter Source Name [default: TenderPDF]: ").strip() or "TenderPDF"
    doc_date = input("📅 Enter Document Date [default: today]: ").strip() or "2025-07-15"
    title = input("📚 Enter Title [default: Parsed Tender Document]: ").strip() or "Parsed Tender Document"

    try:
        chunk_size = int(input("🔢 Enter Chunk Size [default: 2500]: ").strip() or 2500)
        chunk_overlap = int(input("🔁 Enter Chunk Overlap [default: 400]: ").strip() or 400)
    except ValueError:
        logger.warning("⚠️ Invalid numbers, using defaults.")
        chunk_size = 2500
        chunk_overlap = 400

    inputs = {
        "basic_json_path": basic_json_path,
        "index_json_path": index_json_path,
        "source_name": source_name,
        "doc_date": doc_date,
        "title": title,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

    runnable = chunking_runnable()
    output = runnable.invoke(inputs)

    out_file = os.path.splitext(os.path.basename(basic_json_path))[0] + "_chunks.json"
    out_path = os.path.join(TEMP_OUTPUT_DIR, out_file)
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(output["chunks"], out, indent=2, ensure_ascii=False)

    logger.info(f"💾 Chunks saved to: {out_path}")
    print(f"✅ All done! Output written to: {out_path}")