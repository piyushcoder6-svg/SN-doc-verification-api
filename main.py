import os
import re
import time
import random
import base64
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from pydantic import BaseModel, EmailStr
import cv2
import numpy as np
import mediapipe as mp
from paddleocr import PaddleOCR
from google import genai
from google.genai import types

app = FastAPI(
    title="ServiceNow KYC Verification Engine",
    version="1.0.0",
    description="Unified API engine for OCR, Active/Passive Liveness, Gemini AI Review, Risk Scoring, and OTP Services."
)

# -------------------------------------------------------------------
# GLOBAL INITIALIZATIONS
# -------------------------------------------------------------------
# Initialize PaddleOCR (English model)
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Initialize MediaPipe Face Mesh for Liveness
mp_face_mesh = mp.solutions.face_mesh

# Initialize Gemini Client (Uses GEMINI_API_KEY environment variable)
gemini_client = genai.Client()

# In-memory OTP Cache (In production, replace with Redis or database)
otp_store: Dict[str, Dict[str, Any]] = {}

# -------------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------------

class AIReviewRequest(BaseModel):
    ocr_text: str
    customer_name: str
    dob: str
    gender: str
    address: str

class RiskAssessmentRequest(BaseModel):
    income: str
    occupation: str
    country: str
    is_pep: bool = False
    is_aml_hit: bool = False



# -------------------------------------------------------------------
# 2. DOCUMENT OCR SERVICE (PaddleOCR)
# -------------------------------------------------------------------
@app.post("/api/v1/ocr/document", tags=["OCR & Document Processing"])
async def process_document_ocr(document: UploadFile = File(...)):
    try:
        contents = await document.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

        # Run PaddleOCR
        result = ocr_engine.ocr(img, cls=True)
        
        extracted_lines = []
        if result and result[0]:
            for line in result[0]:
                text_content = line[1][0]
                confidence = line[1][1]
                if confidence > 0.4:
                    extracted_lines.append(text_content)

        full_extracted_text = "\n".join(extracted_lines)

        return {
            "status": "success",
            "extracted_text": full_extracted_text,
            "raw_lines": extracted_lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Processing Error: {str(e)}")

# -------------------------------------------------------------------
# 3. AI DOCUMENT REVIEW (Gemini API)
# -------------------------------------------------------------------
@app.post("/api/v1/ai/review-document", tags=["AI Document Review"])
def review_document_with_gemini(payload: AIReviewRequest):
    try:
        prompt = f"""
        You are an automated KYC Document Compliance Reviewer.
        Compare the extracted OCR text from a submitted ID document against the applicant's submitted form details.

        Form Details:
        - Name: {payload.customer_name}
        - Date of Birth: {payload.dob}
        - Gender: {payload.gender}
        - Address: {payload.address}

        Extracted OCR Text from Document:
        ---
        {payload.ocr_text}
        ---

        Tasks:
        1. Compare Name, DOB, Gender, and Address between the Form and Document OCR.
        2. Identify any Mismatches, Missing elements, or Expired indicators.
        3. Assign an overall verification status ("MATCH", "MISMATCH", or "SUSPICIOUS").
        4. Give a confidence score between 0.0 and 1.0.
        5. Provide a clear, concise comment summarizing your findings for the Compliance Officer.

        Return strictly valid JSON using this structure:
        {{
            "status": "MATCH | MISMATCH | SUSPICIOUS",
            "confidence": 0.95,
            "mismatches": ["list of discrepancies if any"],
            "comment": "Summary narrative for the officer."
        }}
        """

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        return {
            "status": "success",
            "review": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Review Engine Error: {str(e)}")

# -------------------------------------------------------------------
# 5. RULE-BASED RISK ENGINE
# -------------------------------------------------------------------
@app.post("/api/v1/risk/calculate", tags=["Risk Engine"])
def calculate_risk(payload: RiskAssessmentRequest):
    risk_score = 0
    risk_factors = []

    # Rule 1: Sanction / AML Hit
    if payload.is_aml_hit:
        risk_score += 100
        risk_factors.append("Matches Global AML/Sanction Watchlist")

    # Rule 2: PEP Status
    if payload.is_pep:
        risk_score += 40
        risk_factors.append("Politically Exposed Person (PEP)")

    # Rule 3: High-Risk Geography
    high_risk_countries = ["FATF Blacklist", "Iran", "North Korea", "Myanmar", "Syria"]
    if payload.country in high_risk_countries:
        risk_score += 35
        risk_factors.append(f"High-Risk Jurisdiction: {payload.country}")

    # Rule 4: Occupation / Income Variance
    if payload.income in ["> 50 Lakhs", "1 Crore+"] and payload.occupation in ["Student", "Unemployed"]:
        risk_score += 25
        risk_factors.append("High income tier inconsistent with reported occupation")

    # Determine Risk Classification Level
    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "status": "success",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommended_action": "AUTOMATED_APPROVAL" if risk_level == "LOW" else "MANUAL_COMPLIANCE_REVIEW"
    }

# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "KYC Verification Microservice Suite"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
