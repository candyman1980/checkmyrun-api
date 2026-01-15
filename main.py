import os
import json
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # can set to gpt-4o-mini for cheaper tests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://checkmyrun.com", "https://www.checkmyrun.com", "http://checkmyrun.com", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    # This will prove you’re on the OpenAI version when you refresh / after deploy
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V2", "model": MODEL}

def to_data_url(upload: UploadFile) -> str:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")

    name = (upload.filename or "").lower()
    if name.endswith(".png"):
        mime = "image/png"
    else:
        mime = "image/jpeg"

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(out).strip()

@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url = to_data_url(left)
        right_url = to_data_url(right)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    # Strict JSON schema response
    response_schema = {
        "name": "checkmyrun_pronation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "left": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "notes": {"type": "string"},
                    },
                    "required": ["pronation", "confidence", "notes"],
                },
                "right": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "notes": {"type": "string"},
                    },
                    "required": ["pronation", "confidence", "notes"],
                },
                "overall": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                        "shoe_category": {"type": "string", "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["pronation", "shoe_category", "confidence"],
                },
                "photo_quality": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "ok": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ok", "issues"],
                },
            },
            "required": ["left", "right", "overall", "photo_quality"],
        },
    }

    instruction = (
        "You are a running shoe fitting assistant.\n"
        "You will be given two outsole (sole) photos: LEFT and RIGHT shoe.\n"
        "Infer pronation style from wear patterns.\n"
        "Be conservative: if wear is unclear, output 'unclear' with low confidence.\n"
        "No medical advice. Notes must be short (1–2 sentences).\n"
        "Also assess photo quality and list issues.\n"
        "Return ONLY valid JSON matching the schema.\n"
    )

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": "LEFT SOLE:"},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_text", "text": "RIGHT SOLE:"},
                    {"type": "input_image", "image_url": right_url},
                ],
            }
        ],
        "response_format": {"type": "json_schema", "json_schema": response_schema},
        "max_output_tokens": 500,
    }

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text}")

    resp_json = r.json()
    text = extract_output_text(resp_json)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Model returned non-JSON unexpectedly", "raw": text}
