import os
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import cv2
import numpy as np
from paddleocr import PaddleOCR
from google import genai
from google.genai import types

app = FastAPI(
    title="ServiceNow KYC Core Verification API",
    version="2.0.0",
    description="Backend microservices for Document OCR, Gemini AI Review, and Risk Scoring."
)

# Initialize Engine Instances
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
gemini_client = genai.Client()

# ===================================================================
# PYDANTIC SCHEMAS (Mapped directly to ServiceNow Table Fields)
# ===================================================================

class AIReviewRequest(BaseModel):
    case_sys_id: Optional[str] = Field(None, description="Sys ID of u_application_case record")
    document_type: str = Field(..., description="e.g., Aadhaar, PAN, Passport")
    ocr_text: str = Field(..., description="Extracted OCR text string")
    customer_name: str
    dob: str
    gender: str
    address: str

class RiskAssessmentRequest(BaseModel):
    case_sys_id: Optional[str] = Field(None, description="Sys ID of u_application_case record")
    annual_income: str = Field(..., description="Mapped from u_application_case.annual_income")
    occupation: str
    account_type: str = Field(..., description="Mapped from u_application_case.account_type")
    country: str = "India"
    is_pep: bool = False
    is_aml_hit: bool = False

# ===================================================================
# 1. DOCUMENT OCR SERVICE (PaddleOCR)
# Mapped to update: u_documents.u_ocr_result
# ===================================================================
@app.post("/api/v1/ocr/document", tags=["Document OCR"])
async def process_document_ocr(
    document: UploadFile = File(...),
    document_type: str = Form("PAN")
):
    try:
        contents = await document.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image upload.")

        result = ocr_engine.ocr(img, cls=True)
        extracted_lines = []
        
        if result and result[0]:
            for line in result[0]:
                text_content = line[1][0]
                confidence = line[1][1]
                if confidence > 0.4:
                    extracted_lines.append(text_content)

        full_text = "\n".join(extracted_lines)

        return {
            "status": "success",
            "document_type": document_type,
            "ocr_result": full_text,
            "line_count": len(extracted_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Engine Failure: {str(e)}")

# ===================================================================
# 2. AI DOCUMENT REVIEW ENGINE (Gemini 2.5 Flash)
# Mapped to update: u_documents.u_ai_review & u_application_case.u_ai_recommendation
# ===================================================================
@app.post("/api/v1/ai/review-document", tags=["AI Compliance Review"])
def review_document_with_gemini(payload: AIReviewRequest):
    try:
        prompt = f"""
        You are a ServiceNow Automated KYC Compliance Reviewer.
        Compare the extracted document text against the customer details submitted in the application case.

        Customer Form Details:
        - Document Type: {payload.document_type}
        - Full Name: {payload.customer_name}
        - Date of Birth: {payload.dob}
        - Gender: {payload.gender}
        - Address: {payload.address}

        Extracted OCR Text:
        ---
        {payload.ocr_text}
        ---

        Perform these checks:
        1. Compare Name, DOB, and ID numbers (PAN/Aadhaar) for exact or fuzzy matches.
        2. Detect discrepancies, missing fields, or invalid layout structure.
        3. Formulate a final recommendation for the assigned Compliance Officer.

        Return strictly valid JSON with this exact structure:
        {{
            "verification_status": "VERIFIED | FAILED | SUSPICIOUS",
            "ai_recommendation": "APPROVE | REJECT | MANUAL_REVIEW",
            "confidence_score": 0.95,
            "discrepancies": ["List any mismatches here"],
            "comments": "Concise 2-sentence summary for the officer workspace."
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
            "case_sys_id": payload.case_sys_id,
            "review_output": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Review Failure: {str(e)}")

# ===================================================================
# 3. RULE-BASED RISK ENGINE
# Mapped to update: u_application_case.u_risk_level
# ===================================================================
@app.post("/api/v1/risk/calculate", tags=["Risk Engine"])
def calculate_case_risk(payload: RiskAssessmentRequest):
    risk_score = 0
    risk_factors = []

    # Rule 1: Watchlist & Sanctions Check
    if payload.is_aml_hit:
        risk_score += 100
        risk_factors.append("AML / Sanctions Watchlist Match")

    # Rule 2: PEP Status
    if payload.is_pep:
        risk_score += 40
        risk_factors.append("Politically Exposed Person (PEP)")

    # Rule 3: High Net Worth / Business Account Discrepancies
    if payload.account_type == "Business" and payload.annual_income in ["10-25 Lakhs", "25+ Lakhs"]:
        risk_score += 15
        risk_factors.append("High volume business account")

    if payload.annual_income == "25+ Lakhs" and payload.occupation in ["Student", "Unemployed"]:
        risk_score += 35
        risk_factors.append("Income bracket mismatched with occupation")

    # Risk Tier Classification (Mapped to Choice List values on u_application_case.u_risk_level)
    if risk_score >= 80:
        risk_level = "Critical"
    elif risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "status": "success",
        "case_sys_id": payload.case_sys_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommended_routing": "AUTO_ASSIGN_SENIOR_OFFICER" if risk_level in ["High", "Critical"] else "STANDARD_QUEUE"
    }

# ===================================================================
# HEALTH ENDPOINT
# ===================================================================
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "engine": "KYC Core FastAPI Microservice"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
