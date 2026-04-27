import streamlit as st
import requests
from pathlib import Path
import time

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .source-box {
        background-color: #e8f4f8;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

# Header
st.markdown('<h1 class="main-header">📚 DocuChat AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Intelligent Document Q&A with RAG</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to ask questions about"
    )
    
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Prepare files for upload
                    files = [
                        ("files", (file.name, file.getvalue(), "application/pdf"))
                        for file in uploaded_files
                    ]
                    
                    # Upload to API
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Processed {len(uploaded_files)} documents!")
                        st.info(f"📊 Created {result['total_chunks']} chunks")
                        st.session_state.documents_uploaded = True
                    else:
                        st.error(f"❌ Error: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error uploading documents: {str(e)}")
        else:
            st.warning("⚠️ Please upload at least one PDF file")
    
    st.divider()
    
    # System info
    st.header("ℹ️ System Info")
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.metric("Vector Store", "✅ Ready" if health["vector_store_initialized"] else "❌ Not Ready")
        st.metric("RAG Chain", "✅ Ready" if health["rag_chain_initialized"] else "❌ Not Ready")
    except:
        st.error("❌ API not running")
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset System", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/reset")
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.documents_uploaded = False
            st.success("✅ System reset!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Main chat interface
st.divider()

# Check if documents are uploaded
if not st.session_state.documents_uploaded:
    st.info("👈 Please upload PDF documents using the sidebar to start chatting!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📎 View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {idx}: {source['source']} (Page {source['page']})</strong><br>
                            {source['content']}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query API
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": prompt,
                            "conversation_id": st.session_state.conversation_id
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        sources = result["sources"]
                        st.session_state.conversation_id = result["conversation_id"]
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display sources
                        if sources:
                            with st.expander("📎 View Sources"):
                                for idx, source in enumerate(sources, 1):
                                    st.markdown(f"""
                                    <div class="source-box">
                                        <strong>Source {idx}: {source['source']} (Page {source['page']})</strong><br>
                                        {source['content']}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"❌ Error: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")

# Footer
st.divider()
st.markdown("""
<p style='text-align: center; color: #888; font-size: 0.9rem;'>
    Built with ❤️ using LangChain, FAISS, and Streamlit | 
    <a href='https://github.com/Jk180603/docuchat-ai'>GitHub</a>
</p>
""", unsafe_allow_html=True)