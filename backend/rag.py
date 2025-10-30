import os
import streamlit as st
import json
import tempfile
import time
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss


HF_TOKEN = st.secrets["HF_TOKEN"]
hf_client = InferenceClient(token=HF_TOKEN)

embedder = SentenceTransformer("all-MiniLM-L6-v2")    
DB_PATH = "vector_store.index"         # stores the FAISS index
META_PATH = "documents_meta.json"      # store metadata in json format
dimension = embedder.get_sentence_embedding_dimension()   # produces vectors of a fixed length (384)
index = faiss.IndexFlatL2(dimension)  # Uses the L2 distance metric to measure similarity.

documents = []     # holds the text chunks
conversation_history = []  
metrics = {        # tracks performance data
    "uploaded_docs": 0,
    "total_embeddings": 0,
    "queries_made": 0,
    "avg_latency": 0.0,
}
# 1.Extraction
def extract_text_from_file(uploaded_file):
    filetype = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filetype)[-1]) as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name
    try:
        if filetype.endswith(".txt"):
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif filetype.endswith(".pdf"):
            text = ""
            reader = PdfReader(temp_path)
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        elif filetype.endswith(".docx"):
            doc = Document(temp_path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif filetype.endswith((".xlsx", ".xls")):
            df = pd.read_excel(temp_path)
            return df.to_string(index=False)
        elif filetype.endswith((".png", ".jpg", ".jpeg")):
            return pytesseract.image_to_string(Image.open(temp_path))
        else:
            raise ValueError("Unsupported file type")
    finally:
        os.remove(temp_path)

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    st.write(f"Number of chunks created: {len(chunks)}")
    # Removing duplicates
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)
    return unique_chunks

def add_document(name, text):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    index.add(np.array(embeddings).astype("float32"))
    for c in chunks:
        documents.append({"name": name, "content": c})    # file name, chunk text
    metrics["uploaded_docs"] += 1
    metrics["total_embeddings"] += len(chunks)
    save_index()
    return True

def search(query, top_k=5):
    if not documents:
        return []
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")  # converts query string into a numeric vector
    scores, indices = index.search(query_vec, top_k)
    results = []
    seen = set()
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(documents):
            doc = documents[idx]
            key = (doc["name"], doc["content"])
            if key not in seen:
                seen.add(key)
                results.append((doc["name"], float(score), doc["content"]))    # Appends a tuple for each result
    return results  # Returns a list of tuples with the most relevant chunks.

def list_documents():
    """ Returns a list of all document names currently in the system. """
    return list(set(doc["name"] for doc in documents))

def answer_with_context(query, retrieved_docs):
    start_time = time.time()
    context = "\n\n".join([doc[2] for doc in retrieved_docs[:5]])
    memory = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history[-3:]])
    prompt = f"""
You are a medical assistant. Use the following context and chat history to answer the question.
Chat history: {memory}
Context: \"\"\"{context}\"\"\"
Question: \"\"\"{query}\"\"\"
Answer concisely and accurately. If not in context, say "Not enough information."
"""
    try:
        completion = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a helpful healthcare assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        answer = completion.choices[0].message["content"].strip()
        
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": answer})
        metrics["queries_made"] += 1
        latency = time.time() - start_time
        # Update average latency
        total_q = metrics["queries_made"]
        metrics["avg_latency"] = ((metrics["avg_latency"] * (total_q - 1)) + latency) / total_q
        return answer
    except Exception as e:
        return f"⚠️ Failed to generate answer: {str(e)}"

def get_dashboard_data():
    return {
        "Uploaded Documents": metrics["uploaded_docs"],
        "Total Embeddings": metrics["total_embeddings"],
        "Queries Made": metrics["queries_made"],
        "Avg Model Latency (s)": round(metrics["avg_latency"], 2),
    }
def save_index():    # saves the FAISS index and metadata to disk
    if not os.path.exists("db"):
        os.makedirs("db")
    faiss.write_index(index, os.path.join("db", DB_PATH))
    with open(os.path.join("db", META_PATH), "w", encoding="utf-8") as f:
        json.dump(documents, f)

def load_index():   # loads them back into memory when the app starts
    global index, documents
    try:
        index = faiss.read_index(os.path.join("db", DB_PATH))
        with open(os.path.join("db", META_PATH), "r", encoding="utf-8") as f:
            documents = json.load(f)
        metrics["uploaded_docs"] = len(set(d["name"] for d in documents))
        metrics["total_embeddings"] = len(documents)
    except Exception:
        documents = []
        index.reset()

load_index()
