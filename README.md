# ServiceNow KYC Core Verification API

FastAPI backend microservices for **Document OCR**, **AI Compliance Review**, and **Risk Scoring** — built for the Deloitte × ServiceNow HackNow 2026 KYC onboarding platform.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/ocr/document` | Extract text from ID document image (Aadhaar/PAN/Passport) |
| `POST` | `/api/v1/ai/review-document` | AI compliance review via Gemini 2.5 Flash |
| `POST` | `/api/v1/risk/calculate` | Rule-based risk scoring (Low / Medium / High / Critical) |

---

## Setup

### 1. Clone & create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get it free from: https://aistudio.google.com/apikey
```

### 3. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

---

## API Reference

### `POST /api/v1/ocr/document`

Accepts a raw binary image sent directly in the request body (as `setRequestBodyFromAttachment()` does from ServiceNow).

**Query Parameters:**
| Param | Default | Description |
|-------|---------|-------------|
| `document_type` | `PAN` | e.g. `Aadhaar`, `PAN`, `Passport` |
| `language` | `eng` | Language hint (kept for API compatibility) |
| `ocr_engine` | `2` | Engine hint (kept for API compatibility — Gemini Vision is used) |
| `is_table` | `false` | Table detection hint |

**Headers:**
```
Content-Type: image/jpeg   (or image/png, application/pdf)
```

**Response:**
```json
{
  "status": "success",
  "document_type": "Aadhaar",
  "ocr_result": "भारत सरकार\nGOVERNMENT OF INDIA\nAnjali\nDOB: 18/09/1999\n...",
  "processing_time_ms": 9882,
  "raw_ocr_response": {
    "engine": "gemini-2.5-flash-vision",
    "mime_type_detected": "image/png",
    "char_count": 557
  }
}
```

---

### `POST /api/v1/ai/review-document`

**Request Body (JSON):**
```json
{
  "case_sys_id": "abc123",
  "document_type": "Aadhaar",
  "ocr_text": "Extracted text from /ocr/document...",
  "customer_name": "Anjali",
  "dob": "18/09/1999",
  "gender": "FEMALE",
  "address": "Delhi - 110093"
}
```

**Response:**
```json
{
  "status": "success",
  "case_sys_id": "abc123",
  "review_output": {
    "verification_status": "VERIFIED",
    "ai_recommendation": "APPROVE",
    "confidence_score": 0.95,
    "discrepancies": [],
    "comments": "All fields match. Document appears authentic."
  }
}
```

---

### `POST /api/v1/risk/calculate`

**Request Body (JSON):**
```json
{
  "case_sys_id": "abc123",
  "annual_income": "5-10 Lakhs",
  "occupation": "Salaried",
  "account_type": "Savings",
  "country": "India",
  "is_pep": false,
  "is_aml_hit": false
}
```

**Response:**
```json
{
  "status": "success",
  "case_sys_id": "abc123",
  "risk_score": 0,
  "risk_level": "Low",
  "risk_factors": [],
  "recommended_routing": "STANDARD_QUEUE"
}
```

---

## OCR Engine

Uses **Gemini 2.5 Flash Vision** (multimodal) for document OCR — supports Aadhaar, PAN, Passport and other Indian ID documents. Extracts both Hindi/Devanagari and English text accurately.

## ServiceNow Integration

In Flow Designer, use Integration Hub → REST spoke:
- Set base URL to your deployed instance
- For OCR: use `setRequestBodyFromAttachment()` to send the raw attachment binary
- Pass `document_type` as a query parameter
