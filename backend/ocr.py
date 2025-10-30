import pytesseract
from PIL import Image
import tempfile  # to create temporary files (for uploading images)
import json
import streamlit as st
from huggingface_hub import InferenceClient

HF_TOKEN = st.secrets["HF_TOKEN"]
hf_client = InferenceClient(token=HF_TOKEN)

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

def structure_with_inference(ocr_text):
    """Use Mistral model to structure prescription text into JSON."""
    try:
        prompt = f"""
You are a medical assistant. Read the following OCR text extracted from a doctor's prescription and convert it into a structured JSON format.

Required fields:
- doctor_name
- doctor_registration
- clinic
- clinic_address
- patient_name
- patient_dob
- patient_age
- patient_gender
- diagnosis
- notes
- medicines: list of objects with
  - name
  - form
  - strength
  - dosage
  - frequency
  - duration
  - route
  - notes

OCR Text:
\"\"\"{ocr_text}\"\"\"

Return ONLY valid JSON. Do not include explanations or text outside JSON.
"""

        completion = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.3,
        )

        text = completion.choices[0].message["content"].strip()

        # Extracting only the JSON portion
        start, end = text.find("{"), text.rfind("}") + 1
        json_text = text[start:end]

        if not json_text.strip().startswith("{"):
            raise ValueError("No valid JSON found in model response.")

        return json.loads(json_text)

    except Exception as e:
        return {"error": "Transformers failed", "details": str(e)}

def summarize_prescription(structured_data):
    """Generate a humanized summary of the structured prescription JSON."""
    try:
        summary_prompt = f"""
            You are a medical assistant. Summarize the following structured prescription JSON
            into a short, friendly, human-readable description.

            Prescription JSON:
            {json.dumps(structured_data, indent=2)}

            Write 3–5 sentences summarizing the prescription.
            Mention the doctor, patient, diagnosis, and medicines with their usage.
            Be conversational and clear. Output only the summary text.
            """

        completion = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a friendly and concise medical assistant."},
                {"role": "user", "content": summary_prompt},
            ],
            max_tokens=300,
            temperature=0.6,
        )

        summary = completion.choices[0].message["content"].strip()
        return summary

    except Exception as e:
        return f"⚠️ Could not generate summary: {str(e)}"

def ocr_stub(uploaded_file):
    """Handle uploaded file, extract text, structure it, and summarize."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    text = extract_text_from_image(temp_path)

    structured = structure_with_inference(text)

    summary = None
    if isinstance(structured, dict) and "error" not in structured:    # Structured JSON must be in dict format with no errors
        summary = summarize_prescription(structured)

    return {
        "raw_text": text,
        "parsed": structured,
        "summary": summary or "⚠️ Summary could not be generated."
    }
