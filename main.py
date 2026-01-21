# main.py
import os
import json
import base64
import io
import math
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---- OpenAI (new SDK) ----
# requirements.txt must include: openai>=1.0.0
from openai import OpenAI

# ---- Pillow for heatmap overlays ----
# requirements.txt add: Pillow
from PIL import Image, ImageDraw, ImageFilter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="CheckMyRun API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# JSON schema for INTERNAL STRUCTURE (we don't show this raw in the UI)
# -------------------------------------------------------------------

PRONATION_SCHEMA: Dict[str, Any] = {
    "name": "checkmyrun_pronation_heatmap_v2",
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
                        "intensity": {"type": "number"},  # 0..1
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
                        "intensity": {"type": "number"},  # 0..1
                    },
                    "required": ["x", "y", "intensity"],
                },
            },

            # User-facing text (what you WANT the UI to show)
            "analysis_text": {"type": "string"},

            # 0..1: how confident the model feels about any left/right bias (not "medical certainty")
            "confidence": {"type": "number"},

            # Extra short notes
            "notes": {"type": ["string", "null"]},
        },
        "required": [
            "ok",
            "has_left_visual",
            "has_right_visual",
            "left_poly_points",
            "right_poly_points",
            "left_heat_points",
            "right_heat_points",
            "analysis_text",
            "confidence",
            "notes",
        ],
    },
}


# --------------------------
# Helpers
# --------------------------

def _default_payload(error: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "analysis_text": error,
        "confidence": 0.0,
        "notes": error,
        "left_overlay_data_url": None,
        "right_overlay_data_url": None,
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
    # SDKs vary; try output_text first
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return json.loads(txt)

    out = getattr(resp, "output", None)
    if isinstance(out, list):
        for item in out:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                    if btype in ("output_text", "text"):
                        btext = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                        if isinstance(btext, str) and btext.strip():
                            return json.loads(btext)

    raise ValueError("Could not extract JSON from model response")


def _clamp01(x: float) -> float:
    try:
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return float(x)
    except Exception:
        return 0.0


def _make_overlay_png(
    base_img_bytes: bytes,
    poly_points: Optional[List[List[float]]],
    heat_points: Optional[List[Dict[str, float]]],
    blur_radius: int = 18,
    point_radius: int = 26,
    overall_alpha: int = 160,
) -> Optional[str]:
    """
    Returns a data:image/png;base64,... overlayed image (original + heatmap/polygon)
    """
    if not base_img_bytes:
        return None

    base = Image.open(io.BytesIO(base_img_bytes)).convert("RGBA")
    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # 1) Heat points as soft blobs (red-ish), intensity controls alpha
    if heat_points:
        # Paint blobs on a separate layer then blur it
        blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(blobs, "RGBA")

        for p in heat_points:
            try:
                x = float(p.get("x", 0))
                y = float(p.get("y", 0))
                inten = _clamp01(float(p.get("intensity", 0.2)))
            except Exception:
                continue

            r = point_radius
            a = int(overall_alpha * (0.25 + 0.75 * inten))  # never invisible
            # Red/orange blob
            bdraw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 60, 0, a))

        blobs = blobs.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        overlay = Image.alpha_composite(overlay, blobs)

    # 2) Polygon outline (green-ish) and faint fill
    if poly_points and isinstance(poly_points, list) and len(poly_points) >= 3:
        try:
            pts = [(float(x), float(y)) for x, y in poly_points if isinstance(x, (int, float)) and isinstance(y, (int, float))]
            if len(pts) >= 3:
                # faint fill
                draw.polygon(pts, fill=(0, 255, 140, 60))
                # outline
                draw.line(pts + [pts[0]], fill=(0, 255, 140, 180), width=4)
        except Exception:
            pass

    combined = Image.alpha_composite(base, overlay)

    out = io.BytesIO()
    combined.save(out, format="PNG")
    out_b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{out_b64}"


async def _analyse_impl(left: UploadFile = None, right: UploadFile = None, rear: UploadFile = None):
    left_bytes = await left.read() if left else None
    right_bytes = await right.read() if right else None
    rear_bytes = await rear.read() if rear else None

    if not left_bytes and not right_bytes and not rear_bytes:
        return JSONResponse(_default_payload("No images provided."), status_code=200)

    if client is None:
        return JSONResponse(_default_payload("OPENAI_API_KEY not set on the server."), status_code=200)

    left_url = _file_to_data_url(left_bytes, left.filename) if left_bytes else None
    right_url = _file_to_data_url(right_bytes, right.filename) if right_bytes else None
    rear_url = _file_to_data_url(rear_bytes, rear.filename) if rear_bytes else None

    # IMPORTANT: best-effort, no more "indeterminable"
    prompt = (
        "You analyze running shoe outsole wear patterns from photos.\n"
        "You MUST return JSON that matches the provided schema.\n\n"
        "Rules:\n"
        "- Always make a BEST-EFFORT estimate if the sole is visible.\n"
        "- Only return null for a side if the image for that side is missing OR totally unusable.\n"
        "- If wear looks mild/symmetrical, still return a broad polygon + low-intensity heat points.\n"
        "- Set has_left_visual/has_right_visual TRUE whenever that sole image is present and visible.\n"
        "- Heat points intensity must be between 0 and 1.\n\n"
        "Output fields:\n"
        "- left_poly_points/right_poly_points: polygon around most worn region as [[x,y],...], at least 3 points.\n"
        "- left_heat_points/right_heat_points: 10–40 points {x,y,intensity} over the worn region.\n"
        "- analysis_text: a helpful, hedged explanation. If unsure, say something like:\n"
        "  'Looks a bit neutral with a slight bias to X' or 'Mostly even, with a touch more wear on Y'.\n"
        "- confidence: 0..1 indicating how strong the left/right bias seems (not medical certainty).\n"
        "- notes: short optional notes.\n"
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

        # Hardening: ensure required keys exist
        for k in PRONATION_SCHEMA["schema"]["required"]:
            if k not in data:
                data[k] = None

        # If images exist, force has_* true (we want informative output)
        if left_bytes:
            data["has_left_visual"] = True
        if right_bytes:
            data["has_right_visual"] = True

        data["ok"] = True
        data["confidence"] = _clamp01(float(data.get("confidence", 0.35)))

        # Generate overlays (what your UI should show)
        left_overlay = _make_overlay_png(left_bytes, data.get("left_poly_points"), data.get("left_heat_points")) if left_bytes else None
        right_overlay = _make_overlay_png(right_bytes, data.get("right_poly_points"), data.get("right_heat_points")) if right_bytes else None

        # Return a CLEAN response payload for the frontend (no raw JSON needed)
        return JSONResponse(
            {
                "ok": True,
                "analysis_text": data.get("analysis_text", "Analysis complete."),
                "confidence": data.get("confidence", 0.35),
                "notes": data.get("notes"),
                "left_overlay_data_url": left_overlay,
                "right_overlay_data_url": right_overlay,
                # optional debug if you want it (leave it out of UI)
                "debug_internal": {
                    "left_poly_points": data.get("left_poly_points"),
                    "right_poly_points": data.get("right_poly_points"),
                    "left_heat_points": data.get("left_heat_points"),
                    "right_heat_points": data.get("right_heat_points"),
                },
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(_default_payload(f"Analyse failed: {str(e)}"), status_code=200)


# --------------------------
# Routes (aliases to avoid 404s)
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
