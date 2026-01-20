import os
import json
import base64
import requests
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # cheaper default for testing
DEBUG_UPLOADS = os.environ.get("DEBUG_UPLOADS", "0") in ("1", "true", "True", "yes", "YES")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://checkmyrun.com", "https://www.checkmyrun.com", "http://checkmyrun.com", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V3-HEATMAP", "model": MODEL}

def to_data_url(upload: UploadFile) -> str:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")

    if DEBUG_UPLOADS:
        print(f"DEBUG_UPLOADS: filename={upload.filename} size={len(b)} bytes")

    name = (upload.filename or "").lower()
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(out).strip()

# ---------- OpenAI JSON schema (strict) ----------
heat_point_schema: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "x": {"type": "number", "minimum": 0, "maximum": 1, "description": "0..1 across the image width"},
        "y": {"type": "number", "minimum": 0, "maximum": 1, "description": "0..1 down the image height"},
        "radius": {"type": "number", "minimum": 0.03, "maximum": 0.6, "description": "0..1 relative radius (use 0.10–0.25 typically)"},
        "intensity": {"type": "number", "minimum": 0.1, "maximum": 1, "description": "relative heat strength"},
        "label": {"type": "string", "description": "short label e.g. 'lateral heel'"},
    },
    "required": ["x", "y", "radius", "intensity", "label"],
}

side_schema: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
        "notes": {"type": "string"},
        "heatmap": {"type": "array", "items": heat_point_schema, "maxItems": 6},
    },
    "required": ["pronation", "notes", "heatmap"],
}

response_schema = {
    "name": "checkmyrun_pronation_heatmap",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "left": side_schema,
            "right": side_schema,
            "overall": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "shoe_category": {"type": "string", "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]},
                    "summary": {"type": "string", "description": "1–2 sentence human-friendly outcome"},
                    "why_these_shoes": {"type": "string", "description": "Short explanation for the category"},
                },
                "required": ["pronation", "shoe_category", "summary", "why_these_shoes"],
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
    "You will be given three photos: LEFT outsole, RIGHT outsole, and a REAR heel view showing both shoes.\n\n"
    "Tasks:\n"
    "1) Identify visible wear hotspots on each outsole (heel: lateral/central/medial; forefoot: lateral/central/medial; toe-off).\n"
    "2) Use outsole wear + the rear heel view (heel alignment / inward collapse indicators / left-right asymmetry) to infer: "
    "overpronation / underpronation / neutral / unclear.\n"
    "3) Recommend shoe category: stability / neutral / cushioned-neutral / unclear.\n"
    "4) Provide a 'heatmap' for EACH outsole: return 2–6 heat points where wear is most visible.\n\n"
    "Heatmap rules:\n"
    "- Each heat point must include x,y,radius,intensity,label.\n"
    "- x,y are 0..1 relative to the outsole image (0,0 top-left; 1,1 bottom-right).\n"
    "- radius is 0..1 (use ~0.10–0.25 typically).\n"
    "- intensity is 0.1..1 (stronger wear = higher intensity).\n"
    "- Only place heat points where wear is actually visible.\n\n"
    "Quality rules:\n"
    "- Base decisions ONLY on what is visible.\n"
    "- If photos are too new, dirty, blurry, shadowed, or angled badly: set pronation 'unclear' and list issues.\n"
    "- If the rear heel photo is unclear, mention it in photo_quality.issues.\n"
    "- Keep notes concise and specific.\n"
    "- Informational only; no medical claims.\n\n"
    "Return ONLY valid JSON matching the provided schema.\n"
)

def _do_analyze(left: UploadFile, right: UploadFile, rear: UploadFile):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url = to_data_url(left)
        right_url = to_data_url(right)
        rear_url = to_data_url(rear)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": "LEFT OUTSOLE:"},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_text", "text": "RIGHT OUTSOLE:"},
                    {"type": "input_image", "image_url": right_url},
                    {"type": "input_text", "text": "REAR HEEL VIEW (BOTH SHOES):"},
                    {"type": "input_image", "image_url": rear_url},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": response_schema["name"],
                "schema": response_schema["schema"],
                "strict": True,
            }
        },
        "max_output_tokens": 700,
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

@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return _do_analyze(left, right, rear)

# UK spelling alias (optional but nice)
@app.post("/analyse")
async def analyse(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return _do_analyze(left, right, rear)
