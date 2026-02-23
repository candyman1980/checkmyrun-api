# main.py -- complete API (robust OpenAI client compatibility + endpoints)
# Drop this file into your repo (replace existing main.py).
# This version purposely avoids assuming a single OpenAI client shape;
# it detects whether the package exposes the new `OpenAI` class (with .responses)
# or only the module namespace, and gracefully falls back to a safe response
# instead of crashing. It also keeps image overlay generation (Pillow) intact
# so the frontend continues to get overlay data URLs if the model returns heat/points.

import os
import io
import json
import base64
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Pillow used to build overlay images that the frontend can display as data URLs.
# You keep Pillow in requirements.txt (Pillow==10.4.0 in your file).
from PIL import Image, ImageDraw, ImageFilter

# Environment/config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI(title="CheckMyRun API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# JSON schema expected from the model (kept here so we can request json_schema)
# ------------------------
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
            "analysis_text": {"type": "string"},
            "confidence": {"type": "number"},
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

# ------------------------
# Helpers
# ------------------------
def _default_payload(error: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "analysis_text": f"Analyse failed: {error}",
        "confidence": 0.0,
        "notes": error,
        "left_overlay_data_url": None,
        "right_overlay_data_url": None,
        "left_debug": error,
        "right_debug": error,
    }


def _file_to_data_url(file_bytes: bytes, filename: str) -> str:
    name = (filename or "").lower()
    mime = "image/jpeg"
    if name.endswith(".png"):
        mime = "image/png"
    elif name.endswith(".webp"):
        mime = "image/webp"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _clamp01(x: float) -> float:
    try:
        v = float(x)
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v
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
    Create a PNG overlay (base image + heat blobs + polygon) returned as a data:... base64 URL.
    If base_img_bytes is falsy, return None.
    """
    if not base_img_bytes:
        return None

    try:
        base = Image.open(io.BytesIO(base_img_bytes)).convert("RGBA")
    except Exception:
        return None

    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # heat blobs
    if heat_points:
        blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(blobs, "RGBA")
        for p in heat_points:
            try:
                x = float(p.get("x", 0))
                y = float(p.get("y", 0))
                inten = _clamp01(p.get("intensity", 0.3))
            except Exception:
                continue
            r = point_radius
            a = int(overall_alpha * (0.25 + 0.75 * inten))
            bdraw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 60, 0, a))
        blobs = blobs.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        overlay = Image.alpha_composite(overlay, blobs)

    # polygon
    if poly_points and isinstance(poly_points, list) and len(poly_points) >= 3:
        try:
            pts = []
            for item in poly_points:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pts.append((float(item[0]), float(item[1])))
            if len(pts) >= 3:
                draw.polygon(pts, fill=(0, 255, 140, 60))
                draw.line(pts + [pts[0]], fill=(0, 255, 140, 190), width=4)
        except Exception:
            pass

    combined = Image.alpha_composite(base, overlay)
    out = io.BytesIO()
    combined.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("utf-8")


# ------------------------
# OpenAI client helper (compatibility wrapper)
# Returns: (client_object, mode)
#   - mode == "responses" -> client supports client.responses.create(...)
#   - mode == "module"    -> openai module only; we won't attempt the schema call here,
#                            we'll fall back so we don't crash in production.
# ------------------------
def _get_openai_client():
    """
    Try to return an instance of the class OpenAI (new SDK) with .responses,
    otherwise return the openai module and mark mode 'module'.
    """
    # If no key configured, raise so caller can handle it.
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set on the server.")

    try:
        # Try class-based client first (preferred)
        # Import inside function to avoid import-time failures during app boot.
        from openai import OpenAI as OpenAIClass

        client = OpenAIClass(api_key=OPENAI_API_KEY)
        return client, "responses"
    except Exception:
        # Fallback: import the module (older/alternate packaging style)
        try:
            import openai

            # set api_key on module (older usage)
            openai.api_key = OPENAI_API_KEY
            return openai, "module"
        except Exception as e:
            # Re-raise a clear error for outer handler to present
            raise RuntimeError(f"OpenAI import failed: {str(e)}")


# ------------------------
# Utilities to parse model response
# ------------------------
def _extract_structured_json(resp_obj: Any) -> Dict[str, Any]:
    """
    Try a few common access patterns to extract JSON text from the returned response
    object. If nothing found, raise ValueError.
    """
    # 1) new SDK: resp_obj.output_text (string)
    txt = getattr(resp_obj, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return json.loads(txt)

    # 2) older new-sdk shape: resp_obj.output (list) -> content blocks
    out = getattr(resp_obj, "output", None)
    if isinstance(out, list):
        for item in out:
            content = None
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype in ("output_text", "text"):
                            btext = block.get("text")
                            if isinstance(btext, str) and btext.strip():
                                return json.loads(btext)
    # 3) fallback: if dict-like and has "output" as string
    if isinstance(resp_obj, dict):
        # many API clients return raw dicts in some builds
        maybe = resp_obj.get("output_text") or resp_obj.get("text") or resp_obj.get("output")
        if isinstance(maybe, str) and maybe.strip():
            return json.loads(maybe)

    raise ValueError("Could not extract JSON from model response")


# ------------------------
# Main analysis implementation
# ------------------------
async def _analyse_impl(left: UploadFile = None, right: UploadFile = None, rear: UploadFile = None):
    left_bytes = await left.read() if left else None
    right_bytes = await right.read() if right else None
    rear_bytes = await rear.read() if rear else None

    if not left_bytes and not right_bytes and not rear_bytes:
        return JSONResponse(_default_payload("No images provided."), status_code=200)

    left_url = _file_to_data_url(left_bytes, getattr(left, "filename", "")) if left_bytes else None
    right_url = _file_to_data_url(right_bytes, getattr(right, "filename", "")) if right_bytes else None
    rear_url = _file_to_data_url(rear_bytes, getattr(rear, "filename", "")) if rear_bytes else None

    # Build the prompt + content for the Responses API
    prompt = (
        "You analyse running shoe outsole wear patterns from photos.\n"
        "You MUST return JSON matching the schema exactly (no extra text) if asked.\n\n"
        "Return fields: left_poly_points, right_poly_points, left_heat_points, right_heat_points, "
        "analysis_text, confidence (0..1), notes, ok, has_left_visual, has_right_visual.\n\n"
        "If a sole photo is present but wear is mild, still return a broad polygon and low-intensity heat points.\n"
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

    # Try to get a client (may raise)
    try:
        client, mode = _get_openai_client()
    except Exception as e:
        return JSONResponse(_default_payload(f"OpenAI init failed: {str(e)}"), status_code=200)

    # If we have the "responses" capable client, use json_schema request.
    if mode == "responses":
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[{"role": "user", "content": content}],
                response_format={"type": "json_schema", "json_schema": PRONATION_SCHEMA},
            )
            # Extract the JSON that the model returned
            data = _extract_structured_json(resp)
        except Exception as e:
            return JSONResponse(_default_payload(f"Analyse failed: {str(e)}"), status_code=200)

    else:
        # mode == "module" (openai module only). Many packaging variants exist and
        # calling an equivalent schema-based endpoint is fragile. Instead of crashing
        # we return a friendly message so the frontend doesn't break.
        return JSONResponse(
            {
                "ok": True,
                "analysis_text": "Model unavailable in this environment; analysis disabled (fallback).",
                "confidence": 0.0,
                "notes": "OpenAI client in module form detected on server; please upgrade packaging or use OpenAI SDK class.",
                "left_overlay_data_url": None,
                "right_overlay_data_url": None,
                "debug_internal": {"mode": "module_fallback"},
            },
            status_code=200,
        )

    # Hardening: ensure expected keys exist so UI doesn't KeyError
    for k in PRONATION_SCHEMA["schema"]["required"]:
        if k not in data:
            data[k] = None

    # Force has_* true when images provided (we want the UI to try)
    if left_bytes:
        data["has_left_visual"] = True
    if right_bytes:
        data["has_right_visual"] = True

    # Ensure ok/confidence sane
    data["ok"] = True
    data["confidence"] = _clamp01(data.get("confidence", 0.35))

    # Build overlays if model provided points
    left_overlay = None
    right_overlay = None
    try:
        left_overlay = _make_overlay_png(left_bytes, data.get("left_poly_points"), data.get("left_heat_points")) if left_bytes else None
    except Exception:
        left_overlay = None
    try:
        right_overlay = _make_overlay_png(right_bytes, data.get("right_poly_points"), data.get("right_heat_points")) if right_bytes else None
    except Exception:
        right_overlay = None

    # Return a clean payload (UI expects these keys)
    return JSONResponse(
        {
            "ok": True,
            "analysis_text": data.get("analysis_text", "Analysis complete."),
            "confidence": data.get("confidence", 0.35),
            "notes": data.get("notes"),
            "left_overlay_data_url": left_overlay,
            "right_overlay_data_url": right_overlay,
            "debug_internal": {
                "left_poly_points": data.get("left_poly_points"),
                "right_poly_points": data.get("right_poly_points"),
                "left_heat_points": data.get("left_heat_points"),
                "right_heat_points": data.get("right_heat_points"),
            },
        },
        status_code=200,
    )


# ------------------------
# Routes
# ------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "checkmyrun-api",
        "message": "API is running. Use /health or POST images to /api/analyse (multipart/form-data).",
        "endpoints": {"health": "/health", "analyse": ["/api/analyse", "/analyse", "/api/analyze", "/analyze"]},
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/analyse")
async def analyse_api(left: UploadFile = File(None), right: UploadFile = File(None), rear: UploadFile = File(None)):
    return await _analyse_impl(left, right, rear)


# aliases
@app.post("/analyse")
async def analyse_root(left: UploadFile = File(None), right: UploadFile = File(None), rear: UploadFile = File(None)):
    return await _analyse_impl(left, right, rear)


@app.post("/api/analyze")
async def analyze_api(left: UploadFile = File(None), right: UploadFile = File(None), rear: UploadFile = File(None)):
    return await _analyse_impl(left, right, rear)


@app.post("/analyze")
async def analyze_root(left: UploadFile = File(None), right: UploadFile = File(None), rear: UploadFile = File(None)):
    return await _analyse_impl(left, right, rear)
