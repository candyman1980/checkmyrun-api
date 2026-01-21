# main.py
import os
import base64
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------- OpenAI client (new SDK) ----------
# pip: openai>=1.0
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # lets the app boot even if openai isn't installed

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # safe default; override in Render env vars

client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

app = FastAPI(title="CheckMyRun API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Strict JSON schema (NO optional keys) ----------
# IMPORTANT: In OpenAI structured outputs, every key in properties must be in required.
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

# ---------- Helpers ----------
def _file_to_data_url(file_bytes: bytes, filename: str) -> str:
    # Best-effort mime detection by extension
    ext = (filename or "").lower()
    if ext.endswith(".png"):
        mime = "image/png"
    elif ext.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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


def _extract_json_from_response(resp: Any) -> Dict[str, Any]:
    """
    Robustly extract the structured JSON from the Responses API object.
    Handles both:
      - resp.output_text (string) if present
      - content blocks with text that is JSON
    """
    # 1) Some SDK versions provide output_text
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return json.loads(txt)

    # 2) Walk output blocks
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    # block could be dict-like or obj-like
                    btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                    if btype in ("output_text", "text"):
                        btext = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                        if isinstance(btext, str) and btext.strip():
                            return json.loads(btext)

    raise ValueError("Could not extract JSON from model response")


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/analyse")
async def analyse(
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    rear: UploadFile = File(None),
):
    # Read files (if provided)
    left_bytes = await left.read() if left else None
    right_bytes = await right.read() if right else None
    rear_bytes = await rear.read() if rear else None

    has_left = bool(left_bytes)
    has_right = bool(right_bytes)

    if not has_left and not has_right and not rear_bytes:
        return JSONResponse(_default_payload("No images provided"), status_code=200)

    if client is None:
        return JSONResponse(_default_payload("OpenAI client not configured (missing OPENAI_API_KEY or openai package)"), status_code=200)

    # Build data URLs for vision
    left_url = _file_to_data_url(left_bytes, left.filename) if left_bytes else None
    right_url = _file_to_data_url(right_bytes, right.filename) if right_bytes else None
    rear_url = _file_to_data_url(rear_bytes, rear.filename) if rear_bytes else None

    # Vision prompt: return polygon (wear zone) + heat points (wear intensity)
    prompt = (
        "You are analyzing running shoe outsole wear from photos.\n"
        "Goal:\n"
        "1) Identify the main visible worn area on each sole (left and right) as a polygon in image coordinates.\n"
        "2) Produce heat points (x,y,intensity 0..1) indicating where wear is strongest.\n"
        "Rules:\n"
        "- Use only what is clearly visible.\n"
        "- If a sole image is missing or unclear, return null arrays for that side and set has_*_visual accordingly.\n"
        "- Polygon points should be an ordered list of [x,y] pairs (at least 3 points), image pixel coordinates.\n"
        "- Heat points should be 10–40 points spread over the worn region.\n"
        "- Debug fields should be short and helpful.\n"
    )

    # Build vision input (include whichever images were provided)
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

        data = _extract_json_from_response(resp)

        # Ensure booleans match presence (model sometimes gets cute)
        data["has_left_visual"] = bool(has_left and data.get("left_poly_points") is not None)
        data["has_right_visual"] = bool(has_right and data.get("right_poly_points") is not None)
        data["ok"] = True

        # Harden keys (never missing)
        for k in PRONATION_SCHEMA["schema"]["required"]:
            if k not in data:
                data[k] = None if k not in ("ok", "has_left_visual", "has_right_visual") else False

        return JSONResponse(data, status_code=200)

    except Exception as e:
        return JSONResponse(_default_payload(f"Analyse failed: {str(e)}"), status_code=200)
