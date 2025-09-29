import os
import streamlit as st
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
import json

# Your custom imports
from utils.pdf_to_json import pdf_to_basic_json_runnable
from utils.extract_index import extract_index_runnable
from utils.chunking import chunking_runnable
from utils.create_embeding import embed_text_runnable
from utils.save_to_json import save_json_runnable
from utils.qdrant import upload_qdrant_runnable, rag_query_runnable
from utils.log import setup_logger
import pprint



# Setup
load_dotenv()
logger = setup_logger("streamlit_app")
MAX_TURNS = 20

# Page config
st.set_page_config(
    page_title="Bot AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=MAX_TURNS)
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if 'latest_contexts' not in st.session_state:
    st.session_state.latest_contexts = [] 
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {}
    


# Pipelines
rag_pipeline = (
    pdf_to_basic_json_runnable()
    | extract_index_runnable()
    | chunking_runnable()
    | embed_text_runnable()
    | save_json_runnable()
    | upload_qdrant_runnable()
)
query_pipe = rag_query_runnable()

with open('/utils/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🤖Bot AI Assistant</h1>
    <p>Intelligent Document Analysis & Query System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    st.markdown('<div class="sidebar-header">📂 Document Management</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a Tender PDF", 
        type=["pdf"],
        help="Select a PDF file to analyze and chat with"
    )

    if uploaded_file:
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("✅ File uploaded successfully!")
        
        # File info
        file_size = len(uploaded_file.getbuffer()) / 1024 / 1024  # MB
        st.info(f"📄 **{uploaded_file.name}**\n\n💾 Size: {file_size:.2f} MB")

        # Process Document
        if st.button("🔄 Process Document", use_container_width=True):
            with st.spinner("🔍 Analyzing document..."):
                try:
                    start_time = datetime.now()
                    file_name = os.path.splitext(uploaded_file.name)[0]
                    
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
                    
                    result = rag_pipeline.invoke(input_dict)
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Store results
                    st.session_state.qdrant_client = result["qdrant_client"]
                    st.session_state.collection_name = result["collection_name"]
                    st.session_state.document_processed = True
                    st.session_state.processing_stats = {
                        "processing_time": processing_time,
                        "chunks_created": result.get("chunks_count", "N/A"),
                        "embeddings_generated": result.get("embeddings_count", "N/A"),
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.success("🎉 Document processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    logger.error(f"Document processing failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Status Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">📊 System Status</div>', unsafe_allow_html=True)
    
    if st.session_state.document_processed:
        st.markdown('''
        <div class="status-indicator status-ready">
            🟢 System Ready
        </div>
        ''', unsafe_allow_html=True)
        
        # Processing Stats
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">{stats.get('processing_time', 0):.1f}s</div>
                    <div class="stat-label">Process Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats.get('chunks_created', 'N/A')}</div>
                    <div class="stat-label">Text Chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown('''
        <div class="status-indicator status-processing">
            🟡 Awaiting Document
        </div>
        ''', unsafe_allow_html=True)
    
    # Chat Statistics
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-number">{len(st.session_state.chat_history)}</div>
            <div class="stat-label">Messages</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{MAX_TURNS}</div>
            <div class="stat-label">Max Turns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main chat interface ---
col1, col2 = st.columns([3, 1])


with col1:
    # Chat input
    if st.session_state.document_processed:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input(
                "", 
                placeholder="💬 Ask me anything about your document...",
                key="chat_input"
            )
            
            col_send, col_clear = st.columns([1, 1])
            with col_send:
                submitted = st.form_submit_button("🚀 Send", use_container_width=True)
            with col_clear:
                clear_chat = st.form_submit_button("🧹 Clear Chat", use_container_width=True)
            
#           # Clear chat
            if clear_chat:
                st.session_state.chat_history.clear()
                st.session_state.latest_contexts = []
                st.success("✅ Chat history cleared!")
                st.rerun()
            
            if submitted and query.strip():
                try:
                    with st.spinner("🤔 Thinking..."):
                        start_time = datetime.now()
                        
                        # Build full conversation history for LLM
                        conversation_context = ""
                        for chat_item in st.session_state.chat_history:
                            q_prev, a_prev, *_ = chat_item
                            conversation_context += f"User: {q_prev}\nAssistant: {a_prev}\n"
                        
                        # Append new user query
                        prompt_for_llm = conversation_context + f"User: {query.strip()}\nAssistant:"

                        # Call RAG query pipeline with full history
                        query_inputs = {
                            "qdrant_client": st.session_state.qdrant_client,
                            "collection_name": st.session_state.collection_name,
                            "query": prompt_for_llm,
                            "history": [(q, a) for q, a, *rest in st.session_state.chat_history]  # optional, can be passed if pipeline needs
                        }
                        
                        response = query_pipe.invoke(query_inputs)
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        answer = response.get("response", "")
                        raw_contexts = response.get("contexts", [])
                        contexts = []
                        for r in raw_contexts:
                            if isinstance(r, str):
                                contexts.append({"text": r, "metadata": {}, "score": None})
                            else:
                                contexts.append({
                                    "text": getattr(r, "text", str(r)),
                                    "metadata": getattr(r, "metadata", {}),
                                    "score": getattr(r, "score", None)
                                })
                        
                        # Save latest contexts for citations
                        st.session_state.latest_contexts = contexts
                        
                        trace_info = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "response_time": response_time,
                            "contexts_found": len(contexts),
                            "query_length": len(query.strip()),
                            "response_length": len(answer),
                            "collection": st.session_state.collection_name
                        }
                        
                        # Add to chat history
                        st.session_state.chat_history.append([
                            query.strip(),
                            answer,
                            contexts,
                            trace_info
                        ])
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")
                    logger.error(f"Query processing failed: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('''
        <div class="input-container">
            <div style="text-align: center; padding: 2rem; color: #64748b;">
                <h3>📋 Getting Started</h3>
                <p>Please upload and process a PDF document first to start chatting.</p>
                <p>Once processed, you'll be able to ask questions about the document content.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Chat History Display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("## 💬 Conversation")
    
    if not st.session_state.chat_history:
        st.markdown('''
        <div style="text-align: center; padding: 3rem; color: #94a3b8;">
            <h4>🌟 Start Your Conversation</h4>
            <p>No messages yet. Ask me anything about your document!</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        # Display messages with enhanced formatting
        for i, chat_item in enumerate(st.session_state.chat_history):
            if len(chat_item) >= 4:
                q, a, contexts, trace_info = chat_item
            else:
                q, a, contexts = chat_item[:3]
                trace_info = {}
            
            st.markdown('<div class="message-container">', unsafe_allow_html=True)
            
            # User message
            st.markdown(f'''
            <div class="user-message">
                <strong>You:</strong> {q}
            </div>
            ''', unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f'''
            <div class="bot-message">
                <strong>Assistant:</strong> {a}
            </div>
            ''', unsafe_allow_html=True)
            
            
            # Add separator between conversations
            if i < len(st.session_state.chat_history) - 1:
                st.markdown('<hr style="margin: 2rem 0; border: 1px solid #e2e8f0;">', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    # Quick Actions Panel
    st.markdown('''
    <div class="sidebar-section">
        <div class="sidebar-header">⚡ Quick Actions</div>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("📊 Export Chat", use_container_width=True):
        if st.session_state.chat_history:
            # Create export data
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_messages": len(st.session_state.chat_history),
                "collection_name": st.session_state.collection_name,
                "conversations": []
            }
            
            for i, chat_item in enumerate(st.session_state.chat_history):
                if len(chat_item) >= 4:
                    q, a, contexts, trace_info = chat_item
                else:
                    q, a, contexts = chat_item[:3]
                    trace_info = {}
                    
                export_data["conversations"].append({
                    "id": i + 1,
                    "query": q,
                    "response": a,
                    "contexts": contexts,
                    "trace_info": trace_info
                })
            
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("No chat history to export")
    
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()
    
    # 👉 New Trace Button
    if st.button("🔍 View Latest Trace", use_container_width=True):
        if st.session_state.chat_history:
            last_chat = st.session_state.chat_history[-1]
            if len(last_chat) >= 4:
                _, _, _, trace_info = last_chat
                st.markdown("### 🔎 Latest Query Trace")
                st.json(trace_info)  # shows full trace in JSON format
            else:
                st.warning("No trace info available for the last message.")
        else:
            st.info("No chat history available to show trace.")
            
            
    ###citation
    if st.button("🧐 Citations", use_container_width=True):
        if st.session_state.latest_contexts:
            with st.expander(f"📚 Sources & Citations ({len(st.session_state.latest_contexts)} found)", expanded=False):
                for j, ctx in enumerate(st.session_state.latest_contexts[:5]):
                    text = ctx.get("text", "")
                    metadata = ctx.get("metadata", {})
                    chunk_id = metadata.get("chunk_index", "N/A")
                    page_num = metadata.get("page_number", "N/A")
                    score = ctx.get("score", None)
                    
                    preview = text[:150] + "..." if len(text) > 150 else text
                    
                    st.markdown(f'''
                    <div class="citation-content">
                        <strong>Source {j+1}:</strong>
                        <div>Chunk #{chunk_id}, Page #{page_num}, Score: {score}</div>
                        <div style="font-style: italic; margin-top: 0.25rem;">"{preview}"</div>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            st.info("No citations available. Send a message first!")
            
    # Help Section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        **How to use:**
        1. Upload a PDF document
        2. Click "Process Document"
        3. Start asking questions!
        
        **Tips:**
        - Be specific in your queries
        - Ask about document sections
        - Request summaries or analyses
        - Use **'View Latest Trace'** to debug performance & sources
        """)
