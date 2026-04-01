from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
import requests
import faiss
import numpy as np
import uuid

app = FastAPI()

# ---------------- FAISS Setup ---------------- #
dimension = 768  # embedding size (nomic model)
index = faiss.IndexFlatL2(dimension)

documents = []  # store text chunks
doc_ids = []    # store ids


# ---------------- Embedding ---------------- #
def get_embedding(text):
    res = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return res.json()["embedding"]


# ---------------- LLM ---------------- #
def generate(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]


# ---------------- Upload PDF ---------------- #
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    global documents, doc_ids

    reader = PdfReader(file.file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # Split into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    vectors = []

    for chunk in chunks:
        emb = get_embedding(chunk)
        vectors.append(emb)
        documents.append(chunk)
        doc_ids.append(str(uuid.uuid4()))

    if vectors:
        vectors_np = np.array(vectors).astype("float32")
        index.add(vectors_np)

    return {"message": "Stored in FAISS", "chunks": len(chunks)}


# ---------------- Query ---------------- #
@app.post("/query_document")
def query_document(query: str = Form(...)):

    if len(documents) == 0:
        return {"answer": "No documents uploaded"}

    # 1. Embed query
    query_embedding = np.array([get_embedding(query)]).astype("float32")

    # 2. Search FAISS
    k = 3
    distances, indices = index.search(query_embedding, k)

    retrieved_docs = []
    for idx in indices[0]:
        if idx < len(documents):
            retrieved_docs.append(documents[idx])

    if not retrieved_docs:
        return {"answer": "No relevant info found"}

    # 3. Build context
    context = "\n".join(retrieved_docs)

    # 4. Prompt
    prompt = f"""
You are a helpful assistant. Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

    # 5. Generate answer
    answer = generate(prompt)

    return {"answer": answer}


# ---------------- Reset ---------------- #
@app.post("/reset_all_data")
def reset():
    global index, documents, doc_ids
    index = faiss.IndexFlatL2(dimension)
    documents = []
    doc_ids = []
    return {"message": "Reset successful"}