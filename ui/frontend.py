import sys
from pathlib import Path
import streamlit as st
from datetime import datetime

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from documents.processor import DocumentProcessor
from graph.graph import create_due_diligence_graph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from documents.registry import get_processor

load_dotenv()

# ====================== INITIALIZE LANGGRAPH ======================
if "due_diligence_agent" not in st.session_state:
    st.session_state.due_diligence_agent = create_due_diligence_graph(processor_provider=get_processor)

# ====================== SESSION MANAGEMENT ======================
if "session_manager" not in st.session_state:
    st.session_state.session_manager = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# NEW: Store chat history PER session
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}   # {session_id: [messages]}

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📁 Due Diligence Sessions")
    
    # Create New Session
    st.subheader("New Session")
    company_name = st.text_input("Company Name", placeholder="AetherLabs AI")
    
    if st.button("Create New Session", type="primary"):
        if company_name.strip():
            session_id = f"{company_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            st.session_state.session_manager[session_id] = {
                "session_id": session_id,
                "company_name": company_name.strip(),
                "created_at": datetime.now().isoformat(),
            }
            
            # Initialize empty chat history for new session
            st.session_state.chat_histories[session_id] = []
            st.session_state.current_session_id = session_id
            
            st.success(f"✅ New session created: **{company_name}**")
            st.rerun()
        else:
            st.error("Please enter a company name")

    # List Previous Sessions
    st.subheader("Previous Sessions")
    if st.session_state.session_manager:
        for sid, session in st.session_state.session_manager.items():
            label = f"🔹 {session['company_name']}" if sid == st.session_state.current_session_id else session['company_name']
            if st.button(label, key=sid, use_container_width=True):
                st.session_state.current_session_id = sid
                st.rerun()
    else:
        st.info("No sessions created yet.")

# ====================== MAIN AREA ======================
st.title("🔍 VC Due Diligence Research Agent")

if not st.session_state.current_session_id:
    st.info("👈 Please create or select a session from the sidebar.")
    st.stop()

current = st.session_state.session_manager[st.session_state.current_session_id]
st.caption(f"**Active Session:** {current['company_name']} | ID: `{current['session_id']}`")

# ====================== DOCUMENT UPLOAD ======================
with st.expander("📄 Upload Documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload Pitch Deck, Financial Model, Reports, etc.",
        type=["pdf", "xlsx", "xls", "md"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # processor = DocumentProcessor(session_id=st.session_state.current_session_id)
            processor = get_processor(st.session_state.current_session_id)
            
            file_paths = []
            for file in uploaded_files:
                temp_path = Path(f"temp_uploads/{file.name}")
                temp_path.parent.mkdir(exist_ok=True)
                temp_path.write_bytes(file.getvalue())
                file_paths.append(str(temp_path))

            result = processor.ingest_multiple_documents(file_paths)
            st.success(f"✅ Stored {result['total_chunks']} chunks")

# ====================== CHAT INTERFACE ======================
st.divider()
st.subheader("💬 Chat with Documents")

# Get current session's chat history
current_chat = st.session_state.chat_histories.get(st.session_state.current_session_id, [])

# Display Chat History
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_query := st.chat_input("Ask about team, market, traction, risks, valuation..."):
    
    # Add user message
    current_chat.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Performing due diligence analysis..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.current_session_id}}
                
                inputs = {
                    "messages": [HumanMessage(content=user_query)],
                    "session_id": st.session_state.current_session_id,
                    "company_name": current["company_name"]
                }
                
                output = st.session_state.due_diligence_agent.invoke(inputs, config)
                final_response = output["messages"][-1].content
                
                st.markdown(final_response)
                
                # Save assistant response
                current_chat.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                current_chat.append({"role": "assistant", "content": error_msg})

# Optional: Clear current session chat
if st.button("Clear Current Session Chat"):
    st.session_state.chat_histories[st.session_state.current_session_id] = []
    st.rerun()