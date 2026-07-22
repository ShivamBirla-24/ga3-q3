"""
Quick local test. Run this AFTER starting the server locally with:
    uvicorn main:app --reload

Usage:
    python test_local.py
"""
import requests

sample_invoice = """Invoice No: INV-2026-0041
Date: 15 March 2026
Vendor: TechParts Pvt Ltd
Subtotal: Rs. 2,199.00
GST (18%): Rs. 395.82
TOTAL: Rs. 2,594.82"""

resp = requests.post(
    "http://127.0.0.1:8000/extract",
    json={"invoice_text": sample_invoice},
)

print("Status code:", resp.status_code)
print("Response:", resp.json())
