import os
import json
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEBUG_UPLOADS = os.environ.get("DEBUG_UPLOADS", "0") == "1"

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
    return {
        "ok": True,
        "service": "checkmyrun-api",
        "marker": "OPENAI-V3-HEATMAP",
        "model": MODEL,
    }

def to_data_url(upload: UploadFile) -> str:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")

    name = (upload.filename or "").lower()
    if name.endswith(".png"):
        mime = "image/png"
    else:
        mime = "image/jpeg"

    if DEBUG_UPLOADS:
        print(f"DEBUG_UPLOADS: filename={upload.filename} size={len(b)} bytes")

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(out).strip()

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))

def sanitize_visual_block(block: dict) -> dict:
    """
    Ensures polygon/heatmap numbers are within 0..1 and minimally well-formed.
    This protects the frontend renderer from weird outputs.
    """
    out = {"outsole_polygon": [], "heatmap": []}

    poly = block.get("outsole_polygon", [])
    if isinstance(poly, list):
        for p in poly:
            if isinstance(p, dict) and "x" in p and "y" in p:
                out["outsole_polygon"].append({"x": clamp01(p["x"]), "y": clamp01(p["y"])})

    pts = block.get("heatmap", [])
    if isinstance(pts, list):
        for p in pts:
            if not isinstance(p, dict):
                continue
            out["heatmap"].append({
                "x": clamp01(p.get("x", 0.5)),
                "y": clamp01(p.get("y", 0.5)),
                "radius": clamp01(p.get("radius", 0.12)),
                "intensity": clamp01(p.get("intensity", 0.6)),
                "label": str(p.get("label", ""))[:80],
            })

    return out

# --------- Core analysis function (shared by /analyse and /analyze) ---------
async def _analyse(left: UploadFile, right: UploadFile, rear: UploadFile):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url = to_data_url(left)
        right_url = to_data_url(right)
        rear_url = to_data_url(rear)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    response_schema = {
        "name": "checkmyrun_pronation_heatmap",
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

                "left_visual": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "outsole_polygon": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["x", "y"],
                            },
                            "minItems": 6
                        },
                        "heatmap": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                    "radius": {"type": "number", "minimum": 0, "maximum": 1},
                                    "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                                    "label": {"type": "string"},
                                },
                                "required": ["x", "y", "radius", "intensity", "label"],
                            },
                            "minItems": 0
                        },
                    },
                    "required": ["outsole_polygon", "heatmap"],
                },

                "right_visual": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "outsole_polygon": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["x", "y"],
                            },
                            "minItems": 6
                        },
                        "heatmap": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                    "radius": {"type": "number", "minimum": 0, "maximum": 1},
                                    "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                                    "label": {"type": "string"},
                                },
                                "required": ["x", "y", "radius", "intensity", "label"],
                            },
                            "minItems": 0
                        },
                    },
                    "required": ["outsole_polygon", "heatmap"],
                },
            },
            "required": ["left", "right", "overall", "photo_quality", "left_visual", "right_visual"],
        },
    }

    instruction = (
        "You are a running shoe fitting assistant.\n"
        "You will be given:\n"
        "- LEFT outsole photo\n"
        "- RIGHT outsole photo\n"
        "- REAR heel photo showing both shoes (if possible)\n\n"
        "IMPORTANT:\n"
        "First, isolate the outsole ONLY.\n"
        "For LEFT and RIGHT outsole photos, return a tight outsole_polygon outlining the visible rubber outsole.\n"
        "Ignore EVERYTHING else: floor/background, wall, hands, socks, laces, shadows.\n"
        "If you cannot confidently outline the outsole, set photo_quality.ok=false and add issues, "
        "but still return your best attempt polygon.\n\n"
        "Tasks:\n"
        "1) Identify visible wear hotspots (heel: lateral/central/medial; forefoot: lateral/central/medial; toe-off).\n"
        "2) Infer likely gait category from wear patterns + rear heel alignment: overpronation / underpronation / neutral / unclear.\n"
        "3) Recommend shoe category: stability / neutral / cushioned-neutral / unclear.\n"
        "4) Produce a heatmap overlay for LEFT and RIGHT as a list of up to 4 points each:\n"
        "   - x,y in 0..1 (relative position)\n"
        "   - radius in 0..1 (relative size)\n"
        "   - intensity in 0..1\n"
        "   - label (e.g. 'lateral heel', 'medial forefoot', 'toe-off')\n"
        "   Heatmap points MUST lie within the outsole_polygon region.\n\n"
        "Rules:\n"
        "- Base decisions ONLY on what is visible.\n"
        "- Use the REAR photo to assess heel alignment/inward collapse and left-right asymmetry.\n"
        "- If the REAR photo is unclear, explicitly mention it in photo_quality.issues.\n"
        "- If outsole is too new/dirty/blurry/heavily shadowed/angle poor, output 'unclear' and list issues.\n"
        "- If left and right differ, reflect that and set overall to the more supportive recommendation.\n"
        "- Keep notes concise and specific (mention observed areas).\n"
        "- Informational only; no medical claims.\n"
        "Return ONLY valid JSON matching the schema."
    )

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": "LEFT OUTSOLE PHOTO:"},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_text", "text": "RIGHT OUTSOLE PHOTO:"},
                    {"type": "input_image", "image_url": right_url},
                    {"type": "input_text", "text": "REAR HEEL PHOTO (BOTH SHOES):"},
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
        "max_output_tokens": 650,
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
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Model returned non-JSON unexpectedly", "raw": text}

    # Light sanitisation to protect the renderer
    if isinstance(data, dict):
        if "left_visual" in data and isinstance(data["left_visual"], dict):
            data["left_visual"] = sanitize_visual_block(data["left_visual"])
        if "right_visual" in data and isinstance(data["right_visual"], dict):
            data["right_visual"] = sanitize_visual_block(data["right_visual"])

    return data

# UK spelling endpoint (frontend uses this)
@app.post("/analyse")
async def analyse(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await _analyse(left, right, rear)

# Keep US spelling as alias so nothing breaks
@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await _analyse(left, right, rear)
