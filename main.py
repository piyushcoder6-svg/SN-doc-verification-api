import os
import base64
import time
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Query, Header
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # picks up GEMINI_API_KEY from .env

# ── Logging setup (visible in Render log panel) ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
log = logging.getLogger("kyc_api")
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ServiceNow KYC Core Verification API",
    version="2.0.0",
    description="Backend microservices for Document OCR, Gemini AI Review, and Risk Scoring."
)

# ── Request/Response logging middleware ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    log.info(f">>> {request.method} {request.url.path} | params={dict(request.query_params)} | content-type={request.headers.get('content-type','')}")
    response = await call_next(request)
    elapsed_ms = round((time.time() - start) * 1000)
    log.info(f"<<< {request.method} {request.url.path} | status={response.status_code} | {elapsed_ms}ms")
    return response
# ─────────────────────────────────────────────────────────────────────────────

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
log.info(f"Gemini client initialized | key_prefix={os.getenv('GEMINI_API_KEY','')[:10]}...")


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
# Engine: Gemini 2.5 Flash Vision (images) + File API (PDFs)
# ===================================================================

def _detect_mime_from_bytes(file_bytes: bytes, header_content_type: str) -> str:
    """Detect MIME type from magic bytes — more reliable than trusting Content-Type header."""
    if file_bytes[:4] == b'%PDF':
        return "application/pdf"
    if file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if file_bytes[:2] in (b'\xff\xd8',) or file_bytes[:4] in (b'\xff\xe0', b'\xff\xe1'):
        return "image/jpeg"
    if len(file_bytes) > 12 and file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WEBP':
        return "image/webp"
    # Fallback: trust Content-Type header
    ct = header_content_type.lower()
    if "png" in ct:
        return "image/png"
    if "pdf" in ct:
        return "application/pdf"
    if "webp" in ct:
        return "image/webp"
    return "image/jpeg"


@app.post("/api/v1/ocr/document", tags=["Document OCR"])
async def process_document_ocr(
    request: Request,
    document_type: str = Query("PAN"),
    language: str = Query("eng"),
    ocr_engine: int = Query(2),   # kept for API compatibility — Gemini is used regardless
    is_table: bool = Query(False)
):
    import tempfile, os as _os

    try:
        # Read raw binary content sent by setRequestBodyFromAttachment()
        file_bytes = await request.body()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file payload received from ServiceNow.")

        # Reliably detect MIME type from magic bytes (don't trust Content-Type alone)
        header_ct = request.headers.get("content-type", "image/jpeg")
        mime_type = _detect_mime_from_bytes(file_bytes, header_ct)

        ocr_prompt = (
            f"You are a document OCR engine. This is a {document_type} identity document. "
            "Extract ALL text visible in the document exactly as it appears — preserve spacing, "
            "line breaks, and formatting. Return ONLY the raw extracted text with no additional "
            "commentary, summary, or interpretation. Just output the full raw text string."
        )

        start_ms = int(time.time() * 1000)

        if mime_type == "application/pdf":
            # PDFs MUST use the Gemini File API — inline base64 is NOT supported for PDFs
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                uploaded = gemini_client.files.upload(
                    file=tmp_path,
                    config={"mime_type": "application/pdf", "display_name": f"{document_type}_upload.pdf"}
                )
                # Wait until Gemini has finished processing the file
                import time as _time
                for _ in range(15):  # max ~15 seconds
                    file_info = gemini_client.files.get(name=uploaded.name)
                    if hasattr(file_info, 'state') and str(file_info.state).endswith("PROCESSING"):
                        _time.sleep(1)
                    else:
                        break

                doc_part = types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded.uri,
                        mime_type="application/pdf"
                    )
                )
            finally:
                _os.unlink(tmp_path)
        else:
            # Images (JPEG, PNG, WebP) — inline base64 works fine
            b64_image = base64.standard_b64encode(file_bytes).decode("utf-8")
            doc_part = types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=b64_image
                )
            )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        doc_part,
                        types.Part(text=ocr_prompt),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
            )
        )

        processing_time_ms = int(time.time() * 1000) - start_ms
        extracted_text = response.text.strip() if response.text else ""

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Gemini OCR returned an empty result. Check document quality or GEMINI_API_KEY."
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
        log.info(f"AI Review | case={payload.case_sys_id} | doc_type={payload.document_type} | name='{payload.customer_name}' | dob='{payload.dob}' | gender='{payload.gender}'")
        log.info(f"AI Review | ocr_text length={len(payload.ocr_text or '')} chars")

        prompt = f"""
You are a KYC Compliance AI for a bank onboarding system. Your job is to compare the customer's
submitted form data against text extracted from their identity document (via OCR).

CUSTOMER FORM DATA (what the customer entered):
- Document Type : {payload.document_type}
- Full Name     : {payload.customer_name or '[not provided]'}
- Date of Birth : {payload.dob or '[not provided]'}
- Gender        : {payload.gender or '[not provided]'}
- Address       : {payload.address or '[not provided]'}

EXTRACTED OCR TEXT FROM DOCUMENT:
---
{payload.ocr_text}
---

COMPARISON RULES — follow these strictly:
1. DATE FORMAT DIFFERENCES ARE NOT DISCREPANCIES. Dates like "14/06/2006", "2006-06-14",
   "June 14, 2006" all represent the same date. Compare semantically, not character-by-character.
2. DO NOT flag Aadhaar numbers, PAN numbers, VID numbers, or Enrolment numbers as discrepancies.
   These are identity card numbers and are NOT fields submitted on the customer form.
3. If a customer form field is blank or '[not provided]', skip that field entirely — do NOT
   list it as a discrepancy. Only compare fields where the customer actually provided data.
4. ONLY flag a discrepancy if the customer provided a value AND it genuinely conflicts with
   what is written on the document (e.g. name "Rahul" on form but "Ravi" on document).
5. Minor spelling variations or abbreviations (e.g. "Piyush Pareek" vs "PIYUSH PAREEK") are
   NOT discrepancies. Use fuzzy/semantic matching.

Return strictly valid JSON with this exact structure:
{{
    "verification_status": "VERIFIED",
    "ai_recommendation": "APPROVE",
    "confidence_score": 0.95,
    "discrepancies": [],
    "comments": "Concise 2-sentence summary for the compliance officer."
}}

Where:
- verification_status: "VERIFIED" (all provided fields match), "FAILED" (genuine mismatch found),
  or "SUSPICIOUS" (something looks wrong but not a clear mismatch)
- ai_recommendation: "APPROVE", "REJECT", or "MANUAL_REVIEW"
- confidence_score: float between 0.0 and 1.0
- discrepancies: list of actual mismatches only (empty list [] if everything matches)
- comments: 1-2 sentence plain English summary for the officer
"""

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        log.info(f"AI Review | case={payload.case_sys_id} | response={response.text[:200] if response.text else 'EMPTY'}")

        return {
            "status": "success",
            "case_sys_id": payload.case_sys_id,
            "review_output": response.text
        }
    except Exception as e:
        log.error(f"AI Review FAILED | case={payload.case_sys_id} | error={str(e)}")
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
