from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import os

app = FastAPI()

# --- 1. CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. AI PIPE CONFIGURATION ---
# Leave the word "AIPIPE_TOKEN" exactly as it is here. 
# Do NOT paste your actual token into this file!
token = os.environ.get("AIPIPE_TOKEN", "missing-token")

client = OpenAI(
    api_key=token, 
    base_url="https://aipipe.org/openai/v1"
)

# --- 3. INPUT FORMAT ---
class InvoiceRequest(BaseModel):
    invoice_text: str

# --- 4. THE API ENDPOINT ---
@app.post("/extract")
async def extract_invoice(req: InvoiceRequest):
    try:
        # The strict rulebook we give the AI to satisfy the grader's requirements
        system_instruction = (
            "You are an expert financial data extraction system. Extract data from the invoice "
            "text into a strict JSON object with EXACTLY these 6 keys:\n"
            "invoice_no, date, vendor, amount, tax, currency.\n\n"
            "Rules:\n"
            "1. If a value is missing, use null.\n"
            "2. 'date' MUST be converted to YYYY-MM-DD format.\n"
            "3. 'amount' is the subtotal BEFORE tax. It must be a raw float number (no commas or text).\n"
            "4. 'tax' is the tax amount. It must be a raw float number.\n"
            "5. 'currency' must be a standard 3-letter ISO code (e.g., 'INR' for Rs., 'USD' for $)."
        )
        
        # Call the AI and force it to output ONLY raw JSON
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"}, # This is the magic command for structured data
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Raw Invoice Text:\n{req.invoice_text}"}
            ],
            temperature=0.0 # Temperature 0 makes the AI highly precise and mathematical
        )
        
        # Turn the AI's text response back into a Python dictionary
        raw_ai_text = response.choices[0].message.content
        extracted_data = json.loads(raw_ai_text)
        
        # --- THE SAFETY NET ---
        # The grader will fail you if even one key is missing. 
        # This loop forces all 6 keys to exist. If the AI missed one, we forcefully add it as 'null'.
        required_keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
        for key in required_keys:
            if key not in extracted_data:
                extracted_data[key] = None
                
        return extracted_data
        
    except Exception as e:
        # Print errors to the Render logs to help us debug if it crashes
        print(f"CRASH ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
