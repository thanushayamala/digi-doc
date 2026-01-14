from fastapi import FastAPI, UploadFile, File
import cv2
import pytesseract
import shutil
import json
import re
import sys
import tempfile
from openai import OpenAI

# ===============================
# OPENAI CLIENT
# ===============================
client = OpenAI(api_key="PASTE_YOUR_OPENAI_API_KEY_HERE")

# ===============================
# TESSERACT AUTO-DETECT
# ===============================
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    raise RuntimeError("Tesseract not found")

pytesseract.pytesseract.tesseract_cmd = tesseract_path

app = FastAPI(title="Prescription OCR API")


# ===============================
# OCR FUNCTION
# ===============================
def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return pytesseract.image_to_string(gray)


# ===============================
# OPENAI ANALYSIS
# ===============================
def analyze_with_openai(ocr_text):
    prompt = f"""
Extract and normalize medical prescription data.

Rules:
- Do not hallucinate missing patient info
- Normalize medicine names conservatively
- Expand abbreviations
- Output ONLY JSON

Format:
{{
  "patient_name": "Unknown",
  "age": "Unknown",
  "gender": "Unknown",
  "doctor_name": "Unknown",
  "hospital_name": "Unknown",
  "date": "Unknown",
  "diagnosis": "Unknown",
  "medicines": [
    {{
      "name": "",
      "dosage": "",
      "frequency": "",
      "duration": ""
    }}
  ],
  "additional_notes": ""
}}

OCR TEXT:
{ocr_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured medical data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ===============================
# API ENDPOINT
# ===============================
@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    raw_text = extract_text(temp_path)
    structured = analyze_with_openai(raw_text)

    return {
        "raw_ocr_text": raw_text,
        "structured_data": structured
    }
