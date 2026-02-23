# main.py
"""
CheckMyRun API (compatibility-friendly)

This file:
- Implements the analyse endpoints (/api/analyse, /analyse, /api/analyze, /analyze)
- Generates overlay PNGs (heat + polygon) using Pillow
- Uses OpenAI SDK if present; supports both new 1.x SDK (OpenAI.responses.create)
  and older OpenAI SDK fallback (openai.ChatCompletion.create)
- Adds a small diagnostic endpoint /_diag_openai to inspect installed openai package
"""

import os
import json
import base64
import io
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Pillow for overlay generation (keep this)
from PIL import Image, ImageDraw, ImageFilter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# If you have a different default model (gpt-4.1-mini or gpt-4o), set OPENAI_MODEL in env

app = FastAPI(title="CheckMyRun API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# JSON schema: internal / expected model output (same shape you provided)
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

# --------------------------
# Helpers
# --------------------------
def _default_payload(error: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "analysis_text": f"Analyse failed: {error}",
        "confidence": 0.0,
        "notes": f"Analyse failed: {error}",
        "left_overlay_data_url": None,
        "right_overlay_data_url": None,
        "left_debug": f"Analyse failed: {error}",
        "right_debug": f"Analyse failed: {error}",
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


def _clamp01(x: float) -> float:
    try:
        x = float(x)
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return x
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
    if not base_img_bytes:
        return None

    base = Image.open(io.BytesIO(base_img_bytes)).convert("RGBA")
    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Heat blobs
    if heat_points:
        blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(blobs, "RGBA")
        for p in heat_points:
            try:
                # Accept either normalized [0..1] coords or pixel coords:
                x = float(p.get("x", 0))
                y = float(p.get("y", 0))
                inten = _clamp01(p.get("intensity", 0.2))
                # Heuristic: if x<=1.0 consider normalized and convert to pixels
                if x <= 1.0:
                    x = x * w
                if y <= 1.0:
                    y = y * h
            except Exception:
                continue
            r = point_radius
            a = int(overall_alpha * (0.25 + 0.75 * inten))
            bdraw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 60, 0, a))
        blobs = blobs.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        overlay = Image.alpha_composite(overlay, blobs)

    # Polygon
    if poly_points and isinstance(poly_points, list) and len(poly_points) >= 3:
        try:
            pts = []
            for x, y in poly_points:
                xf = float(x)
                yf = float(y)
                if xf <= 1.0:
                    xf = xf * w
                if yf <= 1.0:
                    yf = yf * h
                pts.append((xf, yf))
            if len(pts) >= 3:
                draw.polygon(pts, fill=(0, 255, 140, 60))
                draw.line(pts + [pts[0]], fill=(0, 255, 140, 180), width=4)
        except Exception:
            pass

    combined = Image.alpha_composite(base, overlay)
    out = io.BytesIO()
    combined.save(out, format="PNG")
    out_b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{out_b64}"


# --------------------------
# OpenAI compatibility wrapper
# --------------------------
# We'll attempt to use the new OpenAI 1.x SDK first (from openai import OpenAI),
# then fall back to older "openai" module ChatCompletion calls if necessary.
#
# _get_openai_client() returns a tuple (client_obj, client_type_string)
# _call_model(...) sends the prompt and returns a parsed dict (the JSON from model)
#

def _get_openai_client():
    """
    Returns a tuple (client, info_dict)
    client: either an instance of new OpenAI client, or the legacy openai module.
    info: dict with introspection (for diagnostics)
    """
    info: Dict[str, Any] = {}
    # Try new-style SDK first
    try:
        from openai import OpenAI as OpenAIClass  # type: ignore
        client = OpenAIClass(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAIClass()
        info["client_type"] = "new_sdk"
        try:
            # Check attribute presence
            info["has_responses"] = hasattr(client, "responses")
        except Exception:
            info["has_responses"] = False
        # Add module version if available
        try:
            import openai as openai_mod  # type: ignore
            info["openai_version"] = getattr(openai_mod, "__version__", "unknown")
        except Exception:
            info["openai_version"] = "unknown"
        return client, info
    except Exception as e_new:
        # Fall back to legacy module
        try:
            import openai as openai_mod  # type: ignore
            info["client_type"] = "legacy_module"
            info["openai_version"] = getattr(openai_mod, "__version__", "unknown")
            info["legacy_has_ChatCompletion"] = hasattr(openai_mod, "ChatCompletion")
            return openai_mod, info
        except Exception as e_legacy:
            # No OpenAI presence
            info["client_type"] = "none"
            info["error_new"] = str(e_new)
            info["error_legacy"] = str(e_legacy)
            return None, info


def _extract_structured_json_from_response_text(txt: str) -> Dict[str, Any]:
    """
    Given some textual output from the model (string), try to parse JSON.
    Tries to locate the first JSON object in the text.
    """
    if not txt or not isinstance(txt, str):
        raise ValueError("No text output to parse")

    # Simple: find first "{" and last "}" and attempt json.loads
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # fallback: entire string attempt
        try:
            return json.loads(txt)
        except Exception as e:
            raise ValueError(f"Could not find JSON in model text: {str(e)}")
    jtext = txt[start : end + 1]
    return json.loads(jtext)


def _call_model_for_schema(client_tuple, user_content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    client_tuple: (client, info)
    user_content: list of {"type":"input_text"/"input_image", "text":...}/the prompt content
    Returns: parsed dict matching PRONATION_SCHEMA (best-effort)
    """
    client, cinfo = client_tuple
    # Prepare a textual prompt summarizing the content (images are passed as data URLs in content list)
    # We reuse the same prompt structure you already had.
    base_prompt = (
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
        "- analysis_text: a short helpful explanation.\n"
        "- confidence: 0..1 indicating left/right bias confidence (not medical certainty).\n"
        "- notes: optional short notes or null.\n"
    )

    # Build a single textual message that includes any image URLs (data urls)
    parts: List[str] = [base_prompt, "\n--- ATTACHED CONTENT ---\n"]
    for item in user_content:
        t = item.get("type")
        if t == "input_text":
            parts.append(str(item.get("text", "")))
        elif t == "input_image":
            # include short text pointing to the image - in many cases the responses API can accept input_image typed content
            parts.append(f"[IMAGE] {item.get('image_url')}")
    prompt_text = "\n\n".join(parts)

    # First try new SDK flow if available
    if client is None:
        raise RuntimeError("No OpenAI client present")
    # New SDK client object (instance) typically has attribute 'responses'
    if getattr(cinfo, "get", lambda k, d=None: d)("client_type", cinfo.get("client_type", "")) == "new_sdk" or hasattr(client, "responses"):
        try:
            # Use JSON schema response_format if possible (best outcome)
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
                response_format={"type": "json_schema", "json_schema": PRONATION_SCHEMA},
            )
            # Try to extract structured JSON
            # Many SDK response objects expose .output_text or .output; handle both
            return _extract_structured_json(resp)
        except Exception as e:
            # If JSON schema failed for any reason, try a plain responses.create -> text fallback
            try:
                resp = client.responses.create(
                    model=OPENAI_MODEL,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
                )
                # Try reading textual content
                txt = getattr(resp, "output_text", None)
                if isinstance(txt, str) and txt.strip():
                    return _extract_structured_json_from_response_text(txt)
                # Otherwise try scanning resp.output
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
                                        return _extract_structured_json_from_response_text(btext)
                # No structured output found:
                raise RuntimeError(f"No JSON extracted from responses API: {str(e)}")
            except Exception as e2:
                raise RuntimeError(f"New SDK attempt failed: {str(e)} | fallback failed: {str(e2)}")
    # Legacy SDK fallback (openai module)
    else:
        try:
            # client here is the openai module
            # Use ChatCompletion to get text output and parse JSON from assistant message
            messages = [
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt_text},
            ]
            # prefer ChatCompletion (gpt-3.5/4 style) - model name should be set in OPENAI_MODEL
            # Use whichever call the module exposes
            if hasattr(client, "ChatCompletion"):
                completion = client.ChatCompletion.create(model=OPENAI_MODEL, messages=messages, max_tokens=1500, temperature=0.0)
                # extract text
                choices = completion.get("choices") if isinstance(completion, dict) else getattr(completion, "choices", None)
                if choices:
                    first = choices[0]
                    # choice may be a dict or object
                    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
                    if isinstance(message, dict):
                        txt = message.get("content", "")
                    else:
                        # older SDKs may have .text
                        txt = first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")
                    return _extract_structured_json_from_response_text(txt)
                raise RuntimeError("No choices in ChatCompletion response")
            elif hasattr(client, "Completion"):
                # older completions api - generate text and parse json
                prompt_for_completion = prompt_text + "\n\nPlease output only valid JSON now."
                completion = client.Completion.create(engine=OPENAI_MODEL, prompt=prompt_for_completion, max_tokens=1500, temperature=0.0)
                txt = ""
                if isinstance(completion, dict):
                    txt = completion.get("choices", [{}])[0].get("text", "")
                else:
                    txt = getattr(completion.choices[0], "text", "")
                return _extract_structured_json_from_response_text(txt)
            else:
                raise RuntimeError("Legacy openai module found but no ChatCompletion/Completion API present")
        except Exception as e:
            raise RuntimeError(f"Legacy SDK fallback failed: {str(e)}")


# --------------------------
# Core analysis implementation
# --------------------------
async def _analyse_impl(left: UploadFile = None, right: UploadFile = None, rear: UploadFile = None):
    left_bytes = await left.read() if left else None
    right_bytes = await right.read() if right else None
    rear_bytes = await rear.read() if rear else None

    if not left_bytes and not right_bytes and not rear_bytes:
        return JSONResponse(_default_payload("No images provided."), status_code=200)

    left_url = _file_to_data_url(left_bytes, left.filename) if left_bytes else None
    right_url = _file_to_data_url(right_bytes, right.filename) if right_bytes else None
    rear_url = _file_to_data_url(rear_bytes, rear.filename) if rear_bytes else None

    # Build content list for model
    content = [{"type": "input_text", "text": "Begin analysis"},]
    if left_url:
        content.append({"type": "input_text", "text": "LEFT SOLE IMAGE:"})
        content.append({"type": "input_image", "image_url": left_url})
    if right_url:
        content.append({"type": "input_text", "text": "RIGHT SOLE IMAGE:"})
        content.append({"type": "input_image", "image_url": right_url})
    if rear_url:
        content.append({"type": "input_text", "text": "REAR HEEL IMAGE (both shoes):"})
        content.append({"type": "input_image", "image_url": rear_url})

    # Get client and info
    client_tuple = _get_openai_client()

    if client_tuple is None or client_tuple[0] is None:
        return JSONResponse(_default_payload("OpenAI client not available on server."), status_code=200)

    try:
        data = _call_model_for_schema(client_tuple, content)
        # Ensure required keys exist
        for k in PRONATION_SCHEMA["schema"]["required"]:
            if k not in data:
                data[k] = None

        # Force presence if images supplied
        if left_bytes:
            data["has_left_visual"] = True
        if right_bytes:
            data["has_right_visual"] = True

        data["ok"] = True
        data["confidence"] = _clamp01(data.get("confidence", 0.35))

        # Generate overlays
        left_overlay = _make_overlay_png(left_bytes, data.get("left_poly_points"), data.get("left_heat_points")) if left_bytes else None
        right_overlay = _make_overlay_png(right_bytes, data.get("right_poly_points"), data.get("right_heat_points")) if right_bytes else None

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

    except Exception as e:
        return JSONResponse(_default_payload(str(e)), status_code=200)


# --------------------------
# Routes
# --------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "checkmyrun-api",
        "message": "API is running. Use /health or POST images to /api/analyse (multipart/form-data).",
        "endpoints": {
            "health": "/health",
            "analyse": ["/api/analyse", "/analyse", "/api/analyze", "/analyze"],
        },
    }


@app.get("/health")
def health():
    return {"ok": True}


# Diagnostic route to inspect installed openai package / client shape
@app.get("/_diag_openai")
def diag_openai():
    client, info = _get_openai_client()
    # If client exists and is new-style, add a bit of introspection
    diag = {"python_version": sys.version, "OPENAI_API_KEY_set": bool(OPENAI_API_KEY)}
    diag.update(info)
    try:
        if client is not None:
            # show sample dir (trimmed)
            diag["client_dir_sample"] = sorted([k for k in dir(client) if not k.startswith("_")])[:80]
    except Exception as e:
        diag["client_dir_error"] = str(e)
    return diag


# Analyse endpoints (aliases)
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
