import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("🤖 RAG Chat with FAISS")

# ----------- Session Memory ----------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------- Sidebar (Upload) ----------- #
with st.sidebar:
    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])

    if uploaded_file and st.button("Upload"):
        with st.spinner("Uploading and processing..."):
            files = {
                "file": (uploaded_file.name, uploaded_file, "application/pdf")
            }

            res = requests.post(f"{API_URL}/upload_document", files=files)

            if res.status_code == 200:
                st.success("Document uploaded & indexed ✅")
            else:
                st.error("Upload failed ❌")

    if st.button("🧹 Reset Data"):
        requests.post(f"{API_URL}/reset_all_data")
        st.warning("All data cleared")

# ----------- Chat UI ----------- #
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask about your document...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_URL}/query_document",
                data={"query": query}
            )

            if res.status_code == 200:
                answer = res.json()["answer"]
            else:
                answer = "Error from backend"

            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})