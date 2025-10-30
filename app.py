import streamlit as st
import pandas as pd
from backend import rag, ocr, agents
from backend.voice_utils import render_speech_buttons

st.set_page_config(page_title="Health GenAI Demo", layout="wide")
st.title("🩺 Health GenAI — Demo")

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Chat", "Upload & RAG", "OCR"])

if "history" not in st.session_state:
    st.session_state.history = []

# Chat Mode
if mode == "Chat":
    st.subheader("💬 Health Assistant Chat")

    for role, text in st.session_state.history:    # Render previous conversations if any
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)
            render_speech_buttons(text)

    user_input = st.chat_input("Ask a health-related question")
    if user_input:
        st.session_state.history.append(("user", user_input))

        with st.spinner("💬 Thinking..."):
            response = agents.smart_chat(user_input)

        st.session_state.history.append(("assistant", response))

        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(response)
        render_speech_buttons(response)

# RAG Mode
elif mode == "Upload & RAG":
    st.subheader("📄 Upload Documents & Query Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload document (txt, pdf, docx, xlsx, png, jpg, jpeg)",
        type=["txt", "pdf", "docx", "xlsx", "xls", "png", "jpg", "jpeg"]
    )
    if uploaded_file:
        with st.spinner("📥 Processing document..."):
            try:
                content = rag.extract_text_from_file(uploaded_file)
                rag.add_document(uploaded_file.name, content)
                st.success(f"✅ Added document: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to process file: {e}")

    query = st.text_input("🔍 Enter a query to search documents")
    if st.button("Search & Get Answer"):
        if not query.strip():
            st.warning("Please enter a valid search query.")
        else:
            with st.spinner("🧠 Generating answer using RAG..."):
                retrieved_docs = rag.search(query, top_k=5)
                answer = rag.answer_with_context(query, retrieved_docs)

            st.markdown("### 🧠 Answer from Knowledge Base")
            st.success(answer)
            render_speech_buttons(answer)

    if st.button("📋 Show Uploaded Documents"):
        docs = rag.list_documents()
        if not docs:
            st.info("No documents added yet.")
            render_speech_buttons("No documents added yet.")
        else:
            st.write("### 📑 Uploaded Documents")
            for name in docs:
                st.markdown(f"- {name}")
            render_speech_buttons("Here are your uploaded documents.")

# OCR 
elif mode == "OCR":
    st.subheader("🖼 OCR — Extract & Structure Prescription")

    uploaded_img = st.file_uploader("Upload an image for OCR", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        with st.spinner("🔍 Running OCR and structuring..."):
            result = ocr.ocr_stub(uploaded_img)    # Holds dictionary with raw_text, parsed, summary

        st.markdown("### 📝 Extracted Text")
        st.text_area("OCR Text", value=result.get("raw_text", ""), height=200)

        st.markdown("### 🧾 Structured Prescription Data")
        st.json(result.get("parsed", {}))

        parsed = result.get("parsed", {})
        if isinstance(parsed, dict) and parsed.get("medicines"):
            df = pd.DataFrame(parsed["medicines"])   # Converts into pandas Dataframe 
            st.markdown("#### 💊 Medicines Table")
            st.dataframe(df, use_container_width=True)

        st.markdown("### 🗣️ Prescription Summary")
        summary_text = result.get("summary", "⚠️ Summary could not be generated.")
        st.success(summary_text)
        render_speech_buttons(summary_text)
