from huggingface_hub import InferenceClient
import streamlit as st
import spacy

HF_TOKEN = st.secrets["HF_TOKEN"]
hf_client = InferenceClient(token=HF_TOKEN)

nlp = spacy.load("en_core_web_sm")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

def sanitize_text(text: str) -> str:
    """
    Removes sensitive information like person names using Named Entity Recognition (NER).
    Example: "John Doe is here." → "[REPLACED] is here."
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = text.replace(ent.text, "[REPLACED]")
    return text


def smart_chat(prompt: str) -> str:
    query = prompt.strip()
    if not query:
        return "Please type a question."

    history_prompt = ""
    for turn in st.session_state.chat_memory:
        history_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and polite health assistant. "
                "You must NEVER generate or mention any patient names, lab data, or medical records. "
                "If the user greets you, respond politely in one short line (e.g., 'Hello! How can I assist you today?'). "
                "If the user asks about general health or home remedies, provide simple, accurate, and general guidance. "
                "If the question is unrelated or unclear, respond naturally and conversationally. "
                "Under NO circumstances should you invent patient data or refer to fake individuals."
            )
        }
    ]

    if history_prompt:
        messages.append({"role": "user", "content": history_prompt})  # Past conversations are read as single user message
    messages.append({"role": "user", "content": query})

    # Generates Response
    try:  
        completion = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=300,
            temperature=0.4
        )

        response_text = completion.choices[0].message["content"].strip()
        response_text = sanitize_text(response_text)

        st.session_state.chat_memory.append({"user": query, "assistant": response_text})
        return response_text

    except Exception as e:
        return f"⚠️ Failed to generate a response: {str(e)}"
