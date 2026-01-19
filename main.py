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
    "name": "checkmyrun_pronation_v2",
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
                    "certainty": {"type": "string", "enum": ["high", "medium", "low"]},
                    "wear_hotspots": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["pronation", "certainty", "wear_hotspots", "notes"],
            },
            "right": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "certainty": {"type": "string", "enum": ["high", "medium", "low"]},
                    "wear_hotspots": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["pronation", "certainty", "wear_hotspots", "notes"],
            },
            "rear_heel": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "alignment": {"type": "string", "enum": ["neutral", "inward_collapse", "outward_roll", "unclear"]},
                    "asymmetry": {"type": "string", "enum": ["none", "left_more", "right_more", "unclear"]},
                    "notes": {"type": "string"},
                },
                "required": ["alignment", "asymmetry", "notes"],
            },
            "overall": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "shoe_category": {"type": "string", "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]},
                    "certainty": {"type": "string", "enum": ["high", "medium", "low"]},
                    "why": {"type": "string"},
                },
                "required": ["pronation", "shoe_category", "certainty", "why"],
            },
            "photo_quality": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "boolean"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "retake_tips": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["ok", "issues", "retake_tips"],
            },
        },
        "required": ["left", "right", "rear_heel", "overall", "photo_quality"],
    },
}

INSTRUCTION = (
    "You are a running shoe fitting assistant.\n"
    "You will be given three photos:\n"
    "1) LEFT outsole\n"
    "2) RIGHT outsole\n"
    "3) Rear heel view showing BOTH shoes\n\n"
    "Goal: Provide a best-effort pronation/gait indication for each shoe and an overall recommendation.\n\n"
    "How to reason:\n"
    "- Primary signal: outsole wear distribution (medial vs lateral heel, midfoot scuffing, toe-off zone).\n"
    "- Secondary signal: rear heel alignment (inward collapse/eversion vs neutral vs outward roll) and left/right asymmetry.\n"
    "- If signals conflict, choose the more conservative/supportive shoe category and explain why.\n\n"
    "Rules:\n"
    "- Base decisions ONLY on what is visible.\n"
    "- Rear heel photo is REQUIRED input, but if it is unclear, do NOT automatically return 'unclear' overall.\n"
    "- Only use 'unclear' when you truly cannot infer a likely category from the visible evidence.\n"
    "- Always list photo issues if present (blur, shadows, angle, too new/too dirty, cropped heel, not both shoes in rear view).\n"
    "- Write notes in plain English. Avoid medical claims.\n"
    "- Use certainty levels: high / medium / low (no numeric percentages).\n"
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
