import os
import base64
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Query, Header
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # picks up GEMINI_API_KEY from .env

app = FastAPI(
    title="ServiceNow KYC Core Verification API",
    version="2.0.0",
    description="Backend microservices for Document OCR, Gemini AI Review, and Risk Scoring."
)

# Initialize Gemini Client (reads GEMINI_API_KEY from environment automatically)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ===================================================================
# PYDANTIC SCHEMAS (Mapped directly to ServiceNow Table Fields)
# ===================================================================

class AIReviewRequest(BaseModel):
    case_sys_id: Optional[str] = Field(None, description="Sys ID of x_snc_flow4now_b_0_application_case record")
    document_type: str = Field(..., description="e.g., Aadhaar, PAN, Passport")
    ocr_text: Optional[str] = Field("", description="Extracted OCR text string")
    customer_name: Optional[str] = Field("", description="Customer Name")
    dob: Optional[str] = Field("", description="Date of Birth")
    gender: Optional[str] = Field("", description="Gender")
    address: Optional[str] = Field("", description="Address")

class RiskAssessmentRequest(BaseModel):
    case_sys_id: Optional[str] = Field(None, description="Sys ID of x_snc_flow4now_b_0_application_case record")
    annual_income: str = Field(..., description="Mapped from annual_income")
    occupation: str
    account_type: str = Field(..., description="Mapped from account_type")
    country: str = "India"
    is_pep: bool = False
    is_aml_hit: bool = False

# ===================================================================
# 1. DOCUMENT OCR SERVICE (Accepts Raw Binary Body from ServiceNow)
# Mapped to update: x_snc_flow4now_b_0_documents.u_ocr_result
# Engine: Gemini 2.5 Flash Vision (replaces OCR.Space)
# ===================================================================
@app.post("/api/v1/ocr/document", tags=["Document OCR"])
async def process_document_ocr(
    request: Request,
    document_type: str = Query("PAN"),
    language: str = Query("eng"),
    ocr_engine: int = Query(2),   # kept for API compatibility — Gemini is used regardless
    is_table: bool = Query(False)
):
    try:
        # Read raw binary content sent by setRequestBodyFromAttachment()
        file_bytes = await request.body()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file payload received from ServiceNow.")

        # Detect MIME type from Content-Type header sent by ServiceNow
        content_type = request.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            mime_type = "image/png"
        elif "pdf" in content_type:
            mime_type = "application/pdf"
        elif "webp" in content_type:
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"   # safe default for ID card photos

        # Gemini Vision OCR prompt — extract raw text exactly as printed on document
        ocr_prompt = (
            f"You are a document OCR engine. This is a {document_type} identity document. "
            "Extract ALL text visible in the image exactly as it appears — preserve spacing, "
            "line breaks, and formatting. Return ONLY the raw extracted text with no additional "
            "commentary, summary, or interpretation. Just output the full raw text string."
        )

        # Encode image as base64 inline data for Gemini Vision
        b64_image = base64.standard_b64encode(file_bytes).decode("utf-8")
        start_ms = int(time.time() * 1000)

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=b64_image
                            )
                        ),
                        types.Part(text=ocr_prompt),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,          # fully deterministic — OCR must be exact
                max_output_tokens=2048,
            )
        )

        processing_time_ms = int(time.time() * 1000) - start_ms
        extracted_text = response.text.strip() if response.text else ""

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Gemini OCR returned an empty result. Check image quality, format, or GEMINI_API_KEY."
            )

        return {
            "status": "success",
            "document_type": document_type,
            "ocr_result": extracted_text,
            "processing_time_ms": processing_time_ms,
            "raw_ocr_response": {
                "engine": "gemini-2.5-flash-vision",
                "mime_type_detected": mime_type,
                "char_count": len(extracted_text),
            }
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Engine Failure: {str(e)}")

# ===================================================================
# 2. AI DOCUMENT REVIEW ENGINE (Gemini 2.5 Flash)
# Mapped to update: x_snc_flow4now_b_0_documents.u_ai_review & x_snc_flow4now_b_0_application_case.u_ai_recommendation
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
# Mapped to update: x_snc_flow4now_b_0_application_case.u_risk_level
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

    # Risk Tier Classification (Mapped to Choice List values on x_snc_flow4now_b_0_application_case.u_risk_level)
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
