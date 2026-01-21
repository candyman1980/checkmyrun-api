# main.py
import os
import json
import base64
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---- OpenAI (new SDK) ----
# requirements.txt must include: openai>=1.0.0
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allows boot even if package missing

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

app = FastAPI(title="CheckMyRun API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# STRICT schema (matches your UI keys: left_poly_points, left_heat_points, etc.)
# IMPORTANT: In OpenAI structured outputs, every key in properties must be required.
# So we keep the schema minimal and NEVER include preview_data_url here.
# -------------------------------------------------------------------
PRONATION_SCHEMA: Dict[str, Any] = {
    "name": "checkmyrun_pronation_heatmap_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "has_left_visual": {"type": "boolean"},
            "has_right_visual": {"type": "boolean"},
            "left_poly_points": {
                "type": ["array", "null"],
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "right_poly_points": {
                "type": ["array", "null"],
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "left_heat_points": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "intensity": {"type": "number"},
                    },
                    "required": ["x", "y", "intensity"],
                },
            },
            "right_heat_points": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "intensity": {"type": "number"},
                    },
                    "required": ["x", "y", "intensity"],
                },
            },
            "left_debug": {"type": ["string", "null"]},
            "right_debug": {"type": ["string", "null"]},
        },
        "required": [
            "ok",
            "has_left_visual",
            "has_right_visual",
            "left_poly_points",
            "right_poly_points",
            "left_heat_points",
            "right_heat_points",
            "left_debug",
            "right_debug",
        ],
    },
}


def _default_payload(error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "has_left_visual": False,
        "has_right_visual": False,
        "left_poly_points": None,
        "right_poly_points": None,
        "left_heat_points": None,
        "right_heat_points": None,
        "left_debug": error,
        "right_debug": error,
    }


def _file_to_data_url(file_bytes: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".png"):
        mime = "image/png"
    elif name.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_structured_json(resp: Any) -> Dict[str, Any]:
    """
    Works across SDK output shapes.
    Tries output_text first, then walks output -> content blocks.
    """
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return json.loads(txt)

    out = getattr(resp, "output", None)
    if isinstance(out, list):
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    # block may be dict-like or object-like
                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                    if btype in ("output_text", "text"):
                        btext = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                        if isinstance(btext, str) and btext.strip():
                            return json.loads(btext)

    raise ValueError("Could not extract JSON from model response")


async def _analyse_impl(
    left: UploadFile = None,
    right: UploadFile = None,
    rear: UploadFile = None,
):
    left_bytes = await left.read() if left else None
    right_bytes = await right.read() if right else None
    rear_bytes = await rear.read() if rear else None

    if not left_bytes and not right_bytes and not rear_bytes:
        return JSONResponse(_default_payload("No images provided"), status_code=200)

    if client is None:
        return JSONResponse(
            _default_payload("OpenAI not configured: missing OPENAI_API_KEY or openai package"),
            status_code=200,
        )

    left_url = _file_to_data_url(left_bytes, left.filename) if left_bytes else None
    right_url = _file_to_data_url(right_bytes, right.filename) if right_bytes else None
    rear_url = _file_to_data_url(rear_bytes, rear.filename) if rear_bytes else None

    prompt = (
        "You analyze running shoe outsole wear from photos.\n"
        "Return JSON matching the provided schema.\n\n"
        "For each side (left/right) if present:\n"
        "- left_poly_points/right_poly_points: an ordered polygon of the main worn region as [[x,y],...], at least 3 points.\n"
        "- left_heat_points/right_heat_points: 10–40 points {x,y,intensity} with intensity in [0..1] showing strongest wear.\n"
        "- has_left_visual/has_right_visual: true only if that sole image is present and you can see wear clearly.\n"
        "- left_debug/right_debug: short notes if something is unclear.\n\n"
        "If a side is missing or unclear, set that side's poly/heat to null and has_*_visual false.\n"
        "Use only what is clearly visible.\n"
    )

    content = [{"type": "input_text", "text": prompt}]

    if left_url:
        content.append({"type": "input_text", "text": "LEFT SOLE IMAGE:"})
        content.append({"type": "input_image", "image_url": left_url})

    if right_url:
        content.append({"type": "input_text", "text": "RIGHT SOLE IMAGE:"})
        content.append({"type": "input_image", "image_url": right_url})

    if rear_url:
        content.append({"type": "input_text", "text": "REAR HEEL IMAGE (both shoes):"})
        content.append({"type": "input_image", "image_url": rear_url})

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": PRONATION_SCHEMA},
        )

        data = _extract_structured_json(resp)

        # Hardening: ensure all required keys exist (never missing)
        for k in PRONATION_SCHEMA["schema"]["required"]:
            if k not in data:
                data[k] = None

        # Make ok true if model call succeeded; keep booleans sane
        data["ok"] = True

        # If model forgot booleans, infer from presence
        if data.get("has_left_visual") is None:
            data["has_left_visual"] = bool(left_url and data.get("left_poly_points"))
        if data.get("has_right_visual") is None:
            data["has_right_visual"] = bool(right_url and data.get("right_poly_points"))

        return JSONResponse(data, status_code=200)

    except Exception as e:
        # Return 200 with an error payload so your UI can display it
        return JSONResponse(_default_payload(f"Analyse failed: {str(e)}"), status_code=200)


# --------------------------
# Routes (WITH ALIASES)
# This fixes your "Error: Not Found" regardless of whether the frontend calls:
# /api/analyse, /analyse, /api/analyze, /analyze
# --------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/analyse")
async def analyse_api(
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    rear: UploadFile = File(None),
):
    return await _analyse_impl(left, right, rear)


@app.post("/analyse")
async def analyse_root(
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    rear: UploadFile = File(None),
):
    return await _analyse_impl(left, right, rear)


@app.post("/api/analyze")
async def analyze_api(
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    rear: UploadFile = File(None),
):
    return await _analyse_impl(left, right, rear)


@app.post("/analyze")
async def analyze_root(
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    rear: UploadFile = File(None),
):
    return await _analyse_impl(left, right, rear)
