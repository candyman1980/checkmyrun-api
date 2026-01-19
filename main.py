import os
import json
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEBUG_UPLOADS = os.getenv("DEBUG_UPLOADS") == "1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://checkmyrun.com",
        "https://www.checkmyrun.com",
        "http://checkmyrun.com",
        "*",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V3", "model": MODEL}

def file_to_data_url_and_bytes(upload: UploadFile, label: str):
    b = upload.file.read()
    if not b:
        raise ValueError(f"Empty upload for {label}")

    if DEBUG_UPLOADS:
        print(f"DEBUG_UPLOADS: {label} filename={upload.filename} size={len(b)} bytes")

    name = (upload.filename or "").lower()
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}", len(b)

def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(out).strip()

RESPONSE_SCHEMA = {
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

INSTRUCTION = (
    "You are a running shoe fitting assistant.\n"
    "You will be given three photos:\n"
    "1) LEFT outsole\n"
    "2) RIGHT outsole\n"
    "3) A rear heel view showing BOTH shoes side-by-side\n\n"
    "Task:\n"
    "1) Identify visible wear hotspots (heel: lateral/central/medial; forefoot: lateral/central/medial; toe-off).\n"
    "2) Use outsole wear + rear heel alignment to infer gait category: overpronation / underpronation / neutral / unclear.\n"
    "3) Recommend shoe category: stability / neutral / cushioned-neutral / unclear.\n\n"
    "Rules:\n"
    "- Base decisions ONLY on what is visible.\n"
    "- Rear heel photo is helpful for heel alignment / inward collapse / asymmetry, BUT it should NOT invalidate outsole evidence.\n"
    "- If rear heel photo is unclear, add an issue in photo_quality.issues and proceed using outsole evidence.\n"
    "- Only output 'unclear' when the photos genuinely do not show enough to make a reasonable inference.\n"
    "- If photos are imperfect, still make the most likely inference and list the issues.\n"
    "- If left and right differ, reflect that; set overall to the more supportive recommendation.\n"
    "- Keep notes concise and specific (mention observed wear areas and/or heel alignment cues).\n"
    "- Informational only; no medical claims.\n"
    "Return ONLY valid JSON matching the schema."
)

def call_openai(left_url: str, right_url: str, rear_url: str) -> dict:
    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": INSTRUCTION},
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
                "name": RESPONSE_SCHEMA["name"],
                "schema": RESPONSE_SCHEMA["schema"],
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

async def run_analysis(left: UploadFile, right: UploadFile, rear: UploadFile) -> dict:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url, left_bytes = file_to_data_url_and_bytes(left, "LEFT")
        right_url, right_bytes = file_to_data_url_and_bytes(right, "RIGHT")
        rear_url, rear_bytes = file_to_data_url_and_bytes(rear, "REAR")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    # If DEBUG_UPLOADS=1, do not call OpenAI — just prove uploads arrived.
    if DEBUG_UPLOADS:
        return {
            "debug": {
                "model": MODEL,
                "left": {"filename": left.filename, "bytes": left_bytes},
                "right": {"filename": right.filename, "bytes": right_bytes},
                "rear": {"filename": rear.filename, "bytes": rear_bytes},
            }
        }

    return call_openai(left_url, right_url, rear_url)

@app.post("/analyse")
async def analyse(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await run_analysis(left, right, rear)

@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await run_analysis(left, right, rear)
