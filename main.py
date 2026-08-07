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
    applicant_name: Optional[str] = Field("", description="Full name — used for AML/PEP auto-screening")
    annual_income: str = Field(..., description="10-25 Lakhs | 5-10 Lakhs | Below 5 Lakhs")
    occupation: str = Field(..., description="Business owner | Government employee | Salaried | Self employed | Student | Retired | Other")
    account_type: str = Field(..., description="Business | Current | Salary | Joint | Savings | NRI")
    country: str = "India"
    is_pep: bool = False       # auto-detected from name OR set manually
    is_aml_hit: bool = False   # auto-detected from name OR set manually

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

import json as _json

# Load mock AML + PEP lists once at startup
_AML_LIST: List[Dict] = []
_PEP_LIST: List[Dict] = []
try:
    _aml_path = os.path.join(os.path.dirname(__file__), "mock_aml_list.json")
    _pep_path = os.path.join(os.path.dirname(__file__), "mock_pep_list.json")
    with open(_aml_path, encoding="utf-8") as f:
        _AML_LIST = _json.load(f)["entries"]
    with open(_pep_path, encoding="utf-8") as f:
        _PEP_LIST = _json.load(f)["entries"]
    log.info(f"AML list loaded: {len(_AML_LIST)} entries | PEP list loaded: {len(_PEP_LIST)} entries")
except Exception as e:
    log.warning(f"Could not load AML/PEP mock lists: {e}")


# ── Scoring tables ────────────────────────────────────────────────────────────
_INCOME_SCORES: Dict[str, int] = {
    "10-25 Lakhs":  10,
    "5-10 Lakhs":    5,
    "Below 5 Lakhs": 0,
}

_OCCUPATION_SCORES: Dict[str, int] = {
    "Business owner":      20,   # cash-intensive, ownership complexity
    "Self employed":       15,   # irregular income, harder to verify
    "Student":             10,   # income claim unusual
    "Other":               10,   # unknown occupation = elevated scrutiny
    "Retired":              5,   # fixed income, moderate
    "Government employee":  0,   # verifiable, low risk
    "Salaried":             0,   # verifiable, low risk
}

_ACCOUNT_SCORES: Dict[str, int] = {
    "NRI":      20,   # cross-border, FEMA compliance needed
    "Business": 15,   # high volume, cash transactions
    "Current":  10,   # business usage, higher transaction limits
    "Joint":     5,   # multiple beneficial owners
    "Savings":   0,   # standard retail
    "Salary":    0,   # directly linked to employer
}

# Income vs Occupation mismatch rules: (income, occupation, extra_points, reason)
_INCOME_OCCUPATION_MISMATCH = [
    ("10-25 Lakhs", "Student",      35, "Very high income declared by a Student"),
    ("5-10 Lakhs",  "Student",      20, "High income declared by a Student"),
    ("10-25 Lakhs", "Retired",      15, "Unusually high income for Retired person"),
    ("Below 5 Lakhs", "Business owner", 15, "Business owner with very low declared income (possible under-reporting)"),
]

# Account vs Occupation mismatch rules: (account_type, occupation, extra_points, reason)
_ACCOUNT_OCCUPATION_MISMATCH = [
    ("Business", "Student",            20, "Business account held by Student"),
    ("Business", "Salaried",           10, "Business account for a Salaried employee"),
    ("NRI",      "Government employee", 15, "Government employee holding NRI account"),
    ("NRI",      "Student",            15, "Student holding NRI account"),
]
# ─────────────────────────────────────────────────────────────────────────────


def _name_match(name: str, entries: List[Dict]) -> Optional[Dict]:
    """Case-insensitive full-name match (also checks aliases)."""
    if not name:
        return None
    name_lower = name.strip().lower()
    for entry in entries:
        if entry["name"].lower() == name_lower:
            return entry
        for alias in entry.get("alias", []):
            if alias.lower() == name_lower:
                return entry
    return None


@app.post("/api/v1/risk/calculate", tags=["Risk Engine"])
def calculate_case_risk(payload: RiskAssessmentRequest):
    risk_score = 0
    risk_factors: List[str] = []
    is_pep = payload.is_pep
    is_aml_hit = payload.is_aml_hit

    # ── Auto AML/PEP screening by name ───────────────────────────────────────
    if payload.applicant_name:
        aml_match = _name_match(payload.applicant_name, _AML_LIST)
        if aml_match:
            is_aml_hit = True
            risk_factors.append(f"AML/Sanctions hit: {aml_match['reason']}")
            log.warning(f"AML hit for name='{payload.applicant_name}' | reason={aml_match['reason']}")

        pep_match = _name_match(payload.applicant_name, _PEP_LIST)
        if pep_match:
            is_pep = True
            risk_factors.append(f"PEP match: {pep_match['role']} ({pep_match['state']})")
            log.warning(f"PEP hit for name='{payload.applicant_name}' | role={pep_match['role']}")

    # ── Rule 1: AML — always forces Critical ─────────────────────────────────
    if is_aml_hit:
        risk_score += 100
        if "AML/Sanctions hit" not in " ".join(risk_factors):   # avoid duplicate
            risk_factors.append("AML / Sanctions Watchlist Match")

    # ── Rule 2: PEP ──────────────────────────────────────────────────────────
    if is_pep:
        risk_score += 40
        if "PEP match" not in " ".join(risk_factors):
            risk_factors.append("Politically Exposed Person (PEP)")

    # ── Rule 3: Income band base score ───────────────────────────────────────
    income_pts = _INCOME_SCORES.get(payload.annual_income, 5)
    if income_pts > 0:
        risk_score += income_pts
        risk_factors.append(f"Income band: {payload.annual_income} (+{income_pts} pts)")

    # ── Rule 4: Occupation base score ────────────────────────────────────────
    occ_pts = _OCCUPATION_SCORES.get(payload.occupation, 10)
    if occ_pts > 0:
        risk_score += occ_pts
        risk_factors.append(f"Occupation risk: {payload.occupation} (+{occ_pts} pts)")

    # ── Rule 5: Account type base score ──────────────────────────────────────
    acc_pts = _ACCOUNT_SCORES.get(payload.account_type, 5)
    if acc_pts > 0:
        risk_score += acc_pts
        risk_factors.append(f"Account type: {payload.account_type} (+{acc_pts} pts)")

    # ── Rule 6: Income vs Occupation mismatch ────────────────────────────────
    for income, occ, pts, reason in _INCOME_OCCUPATION_MISMATCH:
        if payload.annual_income == income and payload.occupation == occ:
            risk_score += pts
            risk_factors.append(f"Mismatch: {reason} (+{pts} pts)")

    # ── Rule 7: Account vs Occupation mismatch ───────────────────────────────
    for acc, occ, pts, reason in _ACCOUNT_OCCUPATION_MISMATCH:
        if payload.account_type == acc and payload.occupation == occ:
            risk_score += pts
            risk_factors.append(f"Mismatch: {reason} (+{pts} pts)")

    # ── Rule 8: Non-India country ─────────────────────────────────────────────
    if payload.country.strip().lower() not in ("india", "in", ""):
        risk_score += 15
        risk_factors.append(f"Foreign country exposure: {payload.country} (+15 pts)")

    # ── Cap & Tier ────────────────────────────────────────────────────────────
    risk_score = min(risk_score, 100)

    if risk_score >= 75 or is_aml_hit:
        risk_level = "Critical"
        routing = "SAR_INVESTIGATOR"
    elif risk_score >= 50 or is_pep:
        risk_level = "High"
        routing = "COMPLIANCE_OFFICER"
    elif risk_score >= 20:
        risk_level = "Medium"
        routing = "SENIOR_OFFICER"
    else:
        risk_level = "Low"
        routing = "STANDARD_QUEUE"

    log.info(f"Risk calculated | name={payload.applicant_name!r} income={payload.annual_income} occ={payload.occupation} acc={payload.account_type} score={risk_score} tier={risk_level}")

    return {
        "status": "success",
        "case_sys_id": payload.case_sys_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "aml_detected": is_aml_hit,
        "pep_detected": is_pep,
        "recommended_routing": routing
    }

# ===================================================================
# HEALTH ENDPOINT
# ===================================================================
@app.get("/health", tags=["System"])
def health_check():
    aml_count = len(_AML_LIST)
    pep_count = len(_PEP_LIST)
    return {
        "status": "healthy",
        "engine": "KYC Core FastAPI Microservice",
        "aml_list_entries": aml_count,
        "pep_list_entries": pep_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
