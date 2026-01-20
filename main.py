import os
import json
import base64
import math
import requests
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://checkmyrun.com", "https://www.checkmyrun.com", "http://checkmyrun.com", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V3-HEATMAP-POLYRETRY", "model": MODEL}

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

# ---------- polygon validation (pure python, no numpy) ----------

def _poly_points(poly: Any) -> List[Dict[str, float]]:
    if not isinstance(poly, list):
        return []
    pts = []
    for p in poly:
        if isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float)):
            x = float(p["x"])
            y = float(p["y"])
            # keep within [0,1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            pts.append({"x": x, "y": y})
    return pts

def _bbox_area_fraction(pts: List[Dict[str, float]]) -> float:
    if len(pts) < 3:
        return 0.0
    minx = min(p["x"] for p in pts)
    maxx = max(p["x"] for p in pts)
    miny = min(p["y"] for p in pts)
    maxy = max(p["y"] for p in pts)
    return max(0.0, (maxx - minx) * (maxy - miny))

def _centroid(pts: List[Dict[str, float]]) -> Dict[str, float]:
    if not pts:
        return {"x": 0.5, "y": 0.5}
    return {
        "x": sum(p["x"] for p in pts) / len(pts),
        "y": sum(p["y"] for p in pts) / len(pts),
    }

def is_polygon_plausible(poly: Any) -> (bool, str):
    pts = _poly_points(poly)

    if len(pts) < 8:
        return False, f"too_few_points:{len(pts)}"

    area_box = _bbox_area_fraction(pts)
    # If the polygon bbox covers most of the image, it's including background/hand.
    if area_box > 0.65:
        return False, f"bbox_too_large:{area_box:.3f}"

    if area_box < 0.08:
        return False, f"bbox_too_small:{area_box:.3f}"

    c = _centroid(pts)
    # If centroid is way off centre, it's probably latched onto background.
    if (c["x"] < 0.20 or c["x"] > 0.80 or c["y"] < 0.20 or c["y"] > 0.80):
        return False, f"centroid_offcentre:{c['x']:.2f},{c['y']:.2f}"

    return True, "ok"

# ---------- OpenAI call ----------

def call_openai(left_url: str, right_url: str, rear_url: str, instruction: str, response_schema: dict) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "temperature": 0,  # reduce “neutral vs overpronation” randomness
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": "LEFT SOLE:"},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_text", "text": "RIGHT SOLE:"},
                    {"type": "input_image", "image_url": right_url},
                    {"type": "input_text", "text": "REAR HEEL VIEW (BOTH SHOES IN ONE PHOTO):"},
                    {"type": "input_image", "image_url": rear_url},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": response_schema["name"],
                "schema": response_schema["schema"],
                "strict": True
            }
        },
        "max_output_tokens": 900,
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text}")

    resp_json = r.json()
    text = extract_output_text(resp_json)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Model returned non-JSON unexpectedly: {text[:500]}")

# ---------- schema (includes visuals) ----------

RESPONSE_SCHEMA = {
    "name": "checkmyrun_pronation_heatmap_v1",
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
                    # Normalised [0..1] polygon around OUTSOLE ONLY
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
                    },
                    # heat points normalised [0..1]
                    "heatmap": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "radius": {"type": "number", "minimum": 0.03, "maximum": 0.35},
                                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": ["x", "y", "radius", "intensity"],
                        },
                    },
                    # optional: return a cropped preview to match polygon alignment
                    "preview_data_url": {"type": "string"},
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
                    },
                    "heatmap": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0.03, "maximum": 1},
                                "radius": {"type": "number", "minimum": 0.03, "maximum": 0.35},
                                "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": ["x", "y", "radius", "intensity"],
                        },
                    },
                    "preview_data_url": {"type": "string"},
                },
                "required": ["outsole_polygon", "heatmap"],
            },
        },
        "required": ["left", "right", "overall", "photo_quality", "left_visual", "right_visual"],
    },
}

BASE_INSTRUCTION = (
    "You are a running shoe fitting assistant.\n"
    "You will be given two outsole photos: LEFT and RIGHT shoe, plus a rear heel photo showing both shoes.\n\n"
    "Task:\n"
    "1) Identify visible wear hotspots (heel: lateral/central/medial; forefoot: lateral/central/medial; toe-off).\n"
    "2) Infer likely gait category from wear patterns and heel alignment: overpronation / underpronation / neutral / unclear.\n"
    "3) Recommend shoe category: stability / neutral / cushioned-neutral / unclear.\n"
    "4) Produce a visual overlay description:\n"
    "   - For each outsole image, return an OUTSOLE_POLYGON tracing ONLY the shoe outsole perimeter.\n"
    "   - Return 4–8 HEATMAP points ONLY where outsole wear is visible (worn smooth/less tread).\n\n"
    "Rules:\n"
    "- Base your decision ONLY on what is visible in the photos.\n"
    "- Use the rear heel photo to assess heel alignment / inward collapse cues / left-right asymmetry.\n"
    "- If photos are blurry/shadowed/dirty or angle is poor, set pronation/shoe_category to 'unclear' and list issues.\n"
    "- Notes must mention specific areas (e.g., 'lateral heel', 'medial forefoot').\n"
    "- IMPORTANT: OUTSOLE_POLYGON must NOT include floor, hand, background, furniture, or empty space.\n"
    "- OUTSOLE_POLYGON should be tight to the outsole (think: cut-out sticker of the shoe sole).\n"
    "- Place HEATMAP points inside the outsole only. Do NOT mark background.\n"
    "- This is informational only; no medical claims.\n"
    "Return ONLY valid JSON matching the schema."
)

STRICT_POLY_ADDON = (
    "\n\nPOLYGON QUALITY REQUIREMENTS (critical):\n"
    "- The outsole polygon must cover MOST of the shoe sole but as LITTLE background as possible.\n"
    "- If you cannot confidently trace the outsole, still return a polygon around the outsole area only (not the room).\n"
    "- Do not include the hand holding the shoe.\n"
    "- If the image contains a lot of background, ignore it: trace the outsole only.\n"
)

async def _analyse_impl(left: UploadFile, right: UploadFile, rear: UploadFile):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url = to_data_url(left)
        right_url = to_data_url(right)
        rear_url = to_data_url(rear)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    # First attempt
    data = call_openai(left_url, right_url, rear_url, BASE_INSTRUCTION, RESPONSE_SCHEMA)

    # Validate polygons; if bad, retry once with stricter instruction
    okL, whyL = is_polygon_plausible(data.get("left_visual", {}).get("outsole_polygon"))
    okR, whyR = is_polygon_plausible(data.get("right_visual", {}).get("outsole_polygon"))

    if (not okL) or (not okR):
        data2 = call_openai(left_url, right_url, rear_url, BASE_INSTRUCTION + STRICT_POLY_ADDON, RESPONSE_SCHEMA)

        okL2, whyL2 = is_polygon_plausible(data2.get("left_visual", {}).get("outsole_polygon"))
        okR2, whyR2 = is_polygon_plausible(data2.get("right_visual", {}).get("outsole_polygon"))

        # Keep the better set (prefer both ok; otherwise keep first and add a quality issue)
        if okL2 and okR2:
            data = data2
        else:
            # We keep original but add issues so the frontend can decide to hide heatmap if needed
            issues = data.get("photo_quality", {}).get("issues", [])
            if not isinstance(issues, list):
                issues = []
            issues.append(f"Heatmap outline may be inaccurate (left:{whyL}, right:{whyR}). Try a cleaner background / stronger contrast.")
            data["photo_quality"]["issues"] = issues
            data["photo_quality"]["ok"] = False

    return data

# UK spelling route (what your website calls)
@app.post("/analyse")
async def analyse(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await _analyse_impl(left, right, rear)

# US spelling alias (handy for older frontends)
@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await _analyse_impl(left, right, rear)
