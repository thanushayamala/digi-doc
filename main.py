import cv2
import pytesseract
import json
import re
import shutil
import sys
from openai import OpenAI


# =====================================================
# 🔑 OPENAI CONFIGURATION
# =====================================================
# 🔴 PASTE YOUR OPENAI API KEY BELOW
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)




# =====================================================
# 🔒 AUTO-DETECT TESSERACT (NO PATH ISSUES)
# =====================================================
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    print("❌ ERROR: Tesseract not found")
    sys.exit(1)

pytesseract.pytesseract.tesseract_cmd = tesseract_path


# =====================================================
# 1️⃣ OCR: EXTRACT RAW TEXT FROM IMAGE
# =====================================================
def extract_text(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return pytesseract.image_to_string(gray)


# =====================================================
# 2️⃣ BASIC CLEANING (DO NOT OVER-CLEAN)
# =====================================================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s\.\-:/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =====================================================
# 3️⃣ OPENAI: ANALYZE + CORRECT + STRUCTURE
# =====================================================
def analyze_with_openai(ocr_text):
    prompt = f"""
You are an expert medical prescription interpreter.

The text below was extracted from a handwritten medical prescription using OCR.
The text may be incomplete, noisy, and contain spelling mistakes.

GOAL:
- Normalize medicine names to best-known equivalents if reasonably clear
- Expand common medical abbreviations (e.g., Tab, BS, Adv)
- Extract all medically relevant information present or implied
- DO mention  patient names, doctor names, or dates if its visible
- If something is not present, try to guess conservatively
- It is OK to infer medicine names conservatively

OUTPUT ONLY VALID JSON in the format below.

JSON FORMAT:
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
\"\"\"{ocr_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You interpret noisy medical OCR text into structured data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

# =====================================================
# 4️⃣ MAIN PIPELINE
# =====================================================
if __name__ == "__main__":
    image_path = "data/info1.jpeg"   # ✅ ensure filename has NO spaces

    # OCR
    raw_text = extract_text(image_path)
    cleaned_text = clean_text(raw_text)

    print("\n================ RAW OCR TEXT ================\n")
    print(raw_text)

    # OpenAI Interpretation
    print("\n=========== ANALYZING WITH OPENAI ============\n")
    structured_json = analyze_with_openai(cleaned_text)

    print("\n============ FINAL JSON OUTPUT ===============\n")
    print(structured_json)
