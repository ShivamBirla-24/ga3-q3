import os
import json
import re
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Invoice Extraction API")

# --- 1. Allow ANY website/grader to call this API (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Set up the AI client ---
# The API key is read from an environment variable (set this on Render, never in code!)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash-lite"  # much higher free-tier quota than gemini-2.5-flash

# The 6 keys the assignment requires, always in the response.
REQUIRED_KEYS = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]

# This tells Gemini exactly what shape of JSON to return.
# "nullable": True means "it's ok to return null for this field".
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_no": {"type": "string", "nullable": True},
        "date": {
            "type": "string",
            "nullable": True,
            "description": "The invoice date, converted to ISO format YYYY-MM-DD",
        },
        "vendor": {"type": "string", "nullable": True},
        "amount": {
            "type": "number",
            "nullable": True,
            "description": "The subtotal BEFORE tax, as a plain number",
        },
        "tax": {
            "type": "number",
            "nullable": True,
            "description": "The tax amount only (e.g. GST/VAT), as a plain number",
        },
        "currency": {
            "type": "string",
            "nullable": True,
            "description": "3-letter currency code, e.g. INR, USD, EUR",
        },
    },
    "required": REQUIRED_KEYS,
}


class InvoiceText(BaseModel):
    invoice_text: str


def clean_placeholder(value):
    """Sometimes the model writes the literal word 'null' (or 'N/A', 'none',
    empty string) as TEXT instead of a real JSON null. Catch that here so
    the grader always sees a true null, not the string "null"."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("null", "none", "n/a", "na", ""):
        return None
    return value


def normalize_date(value):
    """Make extra sure the date is in strict YYYY-MM-DD format,
    even if the model returned something slightly different."""
    if not value:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value  # couldn't normalize further, return whatever we got


def normalize_number(value):
    """Make sure amount/tax are plain floats, not strings with
    currency symbols or commas."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    return float(cleaned) if cleaned else None


# --- 3. A simple "is it alive" route, useful for testing ---
@app.get("/")
def home():
    return {"status": "Invoice Extraction API is running"}


# --- 4. The actual endpoint the grader will call ---
@app.post("/extract")
def extract(payload: InvoiceText):
    result, _ = run_extraction(payload.invoice_text)
    return result


# --- TEMPORARY debug endpoint: same logic, but also shows the real error.
# The grader only calls /extract, so this is safe to leave in, but you
# can delete this whole function once things are working.
@app.post("/extract-debug")
def extract_debug(payload: InvoiceText):
    result, error = run_extraction(payload.invoice_text)
    result["_debug_error"] = error
    return result


def run_extraction(invoice_text: str):
    """Does the actual work. Returns (result_dict, error_message_or_None)."""
    # Start with all 6 keys set to null. Whatever happens below,
    # we always return this same shape.
    result = {key: None for key in REQUIRED_KEYS}

    try:
        prompt = (
            "You are an invoice data extraction assistant. Read the raw invoice "
            "text below and extract exactly these six fields:\n"
            "- invoice_no: the invoice/bill/document identifier. It may be labeled "
            "'Invoice No', 'Invoice #', 'Invoice Number', 'Bill No', 'Ref No', "
            "'Document No', 'Order No', or similar — extract the ID value regardless of the label used\n"
            "- date: the invoice date, converted to ISO format YYYY-MM-DD\n"
            "- vendor: the company or person who issued the invoice\n"
            "- amount: the SUBTOTAL before tax, as a plain number (no currency symbols, no commas)\n"
            "- tax: the tax amount only (e.g. GST/VAT), as a plain number\n"
            "- currency: the 3-letter currency code. Infer it from symbols "
            "(Rs./₹ = INR, $ = USD, € = EUR, £ = GBP) if not stated directly\n\n"
            "If a field genuinely cannot be found in the text, use JSON null for it "
            "(never the text string \"null\", \"N/A\", or \"none\").\n\n"
            f"Invoice text:\n{invoice_text}"
        )

        response = None
        last_error = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    ),
                )
                break  # success, stop retrying
            except Exception as api_error:
                last_error = api_error
                # Only retry on rate-limit (429) or temporary server (503) errors.
                if "429" in str(api_error) or "503" in str(api_error):
                    time.sleep(2 * (attempt + 1))  # wait a bit longer each retry
                    continue
                raise  # any other error: fail immediately, don't retry

        if response is None:
            raise last_error

        extracted = json.loads(response.text)

        for key in REQUIRED_KEYS:
            result[key] = clean_placeholder(extracted.get(key))

        # Extra safety net on top of what the model already did.
        result["date"] = normalize_date(result["date"])
        result["amount"] = normalize_number(result["amount"])
        result["tax"] = normalize_number(result["tax"])

        return result, None

    except Exception as e:
        # Something went wrong (bad API key, network issue, model refused
        # to produce valid JSON, etc). We still return all 6 keys as nulls,
        # so the response shape is never broken - but we now ALSO return
        # the real error message via the second value, so /extract-debug
        # can show it to you.
        print(f"extract() error: {e}")
        return result, str(e)
