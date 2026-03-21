import os
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pypdf import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama


# ---------------- CONFIG ---------------- #
DATA_DIR = os.environ.get("RAG_DATA_DIR", "rag_data")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index")

os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=4)

app.state.vectorstore = None
app.state.conversation = None


# ---------------- PDF TO TEXT ---------------- #
def pdf_to_text(pdf_bytes: bytes) -> str:
    text = ""
    reader = PdfReader(io.BytesIO(pdf_bytes))

    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    return text


# ---------------- TEXT SPLIT ---------------- #
def split_text(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    return splitter.split_text(text)


# ---------------- BUILD VECTORSTORE ---------------- #
def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434",  # change if needed
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    serialized = FAISS.serialize_to_bytes(vectorstore)

    return vectorstore, serialized


# ---------------- LOAD VECTORSTORE ---------------- #
async def load_vectorstore():
    if not os.path.exists(FAISS_PATH):
        return None

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434",
    )

    with open(FAISS_PATH, "rb") as f:
        serialized = f.read()

    vectorstore = FAISS.deserialize_from_bytes(
        serialized,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


# ---------------- CREATE CONVERSATION CHAIN ---------------- #
async def create_conversation_chain(vstore):
    llm = Ollama(
        model="gemma:4b",
        base_url="http://localhost:11434",
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(),
        memory=memory,
    )


# ---------------- UPLOAD DOCUMENT ---------------- #
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()

    text = await asyncio.get_event_loop().run_in_executor(
        executor, pdf_to_text, data
    )

    chunks = await asyncio.get_event_loop().run_in_executor(
        executor, split_text, text
    )

    vectorstore, serialized = await asyncio.get_event_loop().run_in_executor(
        executor, build_vectorstore, chunks
    )

    with open(FAISS_PATH, "wb") as f:
        f.write(serialized)

    app.state.vectorstore = vectorstore
    app.state.conversation = await create_conversation_chain(vectorstore)

    return {"message": "Document uploaded and indexed successfully"}


# ---------------- QUERY DOCUMENT ---------------- #
@app.post("/query_document")
async def query_document(query: str = Form(...)):
    q = query.lower().strip()

    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "hey!",
        "hi!",
        "good morning",
        "good evening",
        "good afternoon",
    ]

    for pat in greeting_patterns:
        if pat in q:
            return {"answer": "Hi!! How can I assist you today?"}

    if app.state.vectorstore is None:
        vstore = await load_vectorstore()

        if vstore is None:
            raise HTTPException(
                status_code=404,
                detail="No document indexed. Upload a document first.",
            )

        app.state.vectorstore = vstore
        app.state.conversation = await create_conversation_chain(vstore)

    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        executor,
        lambda: app.state.conversation({"question": query}),
    )

    answer = response["answer"]

    return {"answer": answer}


# ---------------- RESET ---------------- #
@app.post("/reset_all_data")
async def reset_all_data():
    import shutil

    app.state.vectorstore = None
    app.state.conversation = None

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)

    return {"message": "All FAISS data cleaned"}