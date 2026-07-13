"""
Streamlit Dashboard for DocuChat Production RAG
"""
import streamlit as st
import requests
import json

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📚",
    layout="wide"
)

API_URL = "http://localhost:8000"

st.title("📚 DocuChat AI")
st.caption("Production RAG System — Hybrid Retrieval, Guardrails, Caching, Tracing")

# Sidebar
with st.sidebar:
    st.header("📤 Upload Document")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        with st.spinner("Ingesting and indexing..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            )
        if response.status_code == 200:
            data = response.json()
            st.success(f"Indexed {data['num_chunks']} chunks from {data['num_pages']} pages")
        else:
            st.error(f"Upload failed: {response.text}")

    st.divider()

    # Stats
    try:
        stats = requests.get(f"{API_URL}/stats").json()
        st.metric("Documents Chunks", stats.get("documents_loaded", 0))
        st.metric("Memory Messages", stats.get("current_memory_size", 0))
        retriever_ready = stats.get("retriever_ready", False)
        if retriever_ready:
            st.success("Retriever Ready")
        else:
            st.warning("No documents yet")
    except:
        st.warning("API not connected")

    if st.button("Clear Memory"):
        requests.delete(f"{API_URL}/memory")
        st.session_state.messages = []
        st.rerun()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("metadata"):
            with st.expander("Response details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", f"{msg['metadata'].get('latency_ms', 0):.0f}ms")
                col2.metric("Provider", msg['metadata'].get('provider', 'unknown'))
                col3.metric("Cached", str(msg['metadata'].get('cached', False)))
                if msg['metadata'].get('sources'):
                    st.write("**Sources:**")
                    for s in msg['metadata']['sources']:
                        st.caption(f"Page {s.get('page', '?')} — {s.get('content', '')[:150]}...")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": prompt, "use_memory": True},
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.write(answer)

                    with st.expander("Response details"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Latency", f"{data['latency_ms']:.0f}ms")
                        col2.metric("Provider", data['provider'])
                        col3.metric("Cached", str(data['cached']))
                        st.caption(f"Trace ID: {data['trace_id']} | Guardrail: {data['guardrail_status']}")
                        if data.get("sources"):
                            st.write("**Sources:**")
                            for s in data["sources"]:
                                st.caption(f"Page {s.get('page', '?')} — {s.get('content', '')[:150]}...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": data,
                    })
                elif response.status_code == 400:
                    err = response.json().get("detail", "Query blocked")
                    st.warning(f"Query blocked: {err}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Query blocked: {err}",
                    })
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running.")