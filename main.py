# main.py
import os
import json
import base64
import io
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

# ---- OpenAI (new SDK) ----
# requirements.txt must include: openai==1.x and httpx pinned (e.g. httpx==0.27.2)
# Import may raise if package missing; we lazy-create the client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # will check later

# ---- Pillow for heatmap overlays (optional) ----
try:
    from PIL import Image, ImageDraw, ImageFilter
except Exception:
    Image = None
    ImageDraw = None
    ImageFilter = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # you can set via env

app = FastAPI(title="CheckMyRun API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# JSON schema for INTERNAL STRUCTURE (kept similar to your previous schema)
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
# Small embedded frontend (no extra files needed)
# --------------------------
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>CheckMyRun</title>
  <style>
    body{font-family:system-ui,Segoe UI,Helvetica,Arial;margin:28px;background:#fff;color:#111}
    .card{max-width:900px;margin:0 auto;padding:18px;border-radius:12px;border:1px solid #eee;background:#fafafa}
    .grid{display:flex;gap:12px;flex-wrap:wrap}
    label.drop{display:inline-block;padding:12px;border:2px dashed #ddd;border-radius:10px;cursor:pointer}
    input[type=file]{display:block;margin-top:8px}
    button{padding:10px 14px;border-radius:8px;border:1px solid #111;background:#fff;cursor:pointer}
    pre{background:#f4f6f8;padding:12px;border-radius:8px;overflow:auto;max-height:360px}
  </style>
</head>
<body>
  <div class="card">
    <h1>CheckMyRun</h1>
    <p>Upload clear photos of both soles and a rear-heel photo (optional). The first request may be slower if the server is waking up.</p>

    <form id="form" enctype="multipart/form-data">
      <div class="grid">
        <label class="drop">Left sole<br><input type="file" name="left" accept="image/*" required></label>
        <label class="drop">Right sole<br><input type="file" name="right" accept="image/*" required></label>
        <label class="drop">Rear heel (both shoes)<br><input type="file" name="rear" accept="image/*"></label>
      </div>

      <div style="margin-top:12px">
        <button id="btn">Analyse</button>
        <span id="status" style="margin-left:12px;color:#666"></span>
      </div>
    </form>

    <div id="result" style="margin-top:18px;display:none">
      <h2>Result</h2>
      <div id="human" style="margin-bottom:8px;color:#111"></div>
      <div id="images" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px"></div>
      <pre id="json"></pre>
    </div>

    <footer style="margin-top:14px;font-size:13px;color:#666">
      Informational only — not medical advice. Links may earn a commission at no extra cost to you.
    </footer>
  </div>

<script>
const API = "/api/analyse";
const form = document.getElementById('form');
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const result = document.getElementById('result');
const jsonOut = document.getElementById('json');
const human = document.getElementById('human');
const images = document.getElementById('images');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true;
  status.textContent = 'Uploading...';
  result.style.display = 'none';
  human.textContent = '';
  images.innerHTML = '';
  try {
    const fd = new FormData(form);
    const res = await fetch(API, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Server error: ' + res.status);
    const j = await res.json();

    // Human summary
    human.innerHTML = `<strong>${j.analysis_text || 'Analysis complete.'}</strong>`;

    // show overlays if present
    if (j.left_overlay_data_url) {
      const img = document.createElement('img');
      img.src = j.left_overlay_data_url;
      img.style.maxWidth = '300px';
      img.style.border = '1px solid #ddd';
      images.appendChild(img);
    }
    if (j.right_overlay_data_url) {
      const img = document.createElement('img');
      img.src = j.right_overlay_data_url;
      img.style.maxWidth = '300px';
      img.style.border = '1px solid #ddd';
      images.appendChild(img);
    }

    jsonOut.textContent = JSON.stringify(j, null, 2);
    result.style.display = 'block';
    status.textContent = 'Done ✅';
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""

# --------------------------
# Helpers (same approach you used earlier)
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
        x = float(x)
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return x
    except Exception:
        return 0.0


# Keep the overlay function close to your original (no heatmap overhaul for now)
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
    if not base_img_bytes or Image is None:
        return None

    try:
        base = Image.open(io.BytesIO(base_img_bytes)).convert("RGBA")
    except Exception:
        return None

    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # 1) Heat points as soft blobs (red-ish), intensity controls alpha
    if heat_points:
        blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bdraw = ImageDraw.Draw(blobs, "RGBA")

        for p in heat_points:
            try:
                x = float(p.get("x", 0))
                y = float(p.get("y", 0))
                inten = _clamp01(p.get("intensity", 0.2))
            except Exception:
                continue

            # attempt to detect normalized coords (0..1)
            if 0 <= x <= 1 and 0 <= y <= 1:
                x_px = int(round(x * w))
                y_px = int(round(y * h))
            else:
                x_px = int(round(x))
                y_px = int(round(y))

            r = point_radius
            a = int(overall_alpha * (0.25 + 0.75 * inten))  # never invisible
            bdraw.ellipse((x_px - r, y_px - r, x_px + r, y_px + r), fill=(255, 60, 0, a))

        blobs = blobs.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        overlay = Image.alpha_composite(overlay, blobs)

    # 2) Polygon outline (green-ish) and faint fill
    if poly_points and isinstance(poly_points, list) and len(poly_points) >= 3:
        try:
            pts = []
            for x, y in poly_points:
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append((int(round(x * w)), int(round(y * h))))
                else:
                    pts.append((int(round(x)), int(round(y))))
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


def _get_openai_client() -> OpenAI:
    """
    Lazy client creation so the app can boot even if OpenAI package/env are temporarily missing.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set on the server.")
    if OpenAI is None:
        raise RuntimeError("OpenAI package is not installed in the environment.")
    return OpenAI(api_key=OPENAI_API_KEY)


async def _analyse_impl(left: UploadFile = None, right: UploadFile = None, rear: UploadFile = None):
    left_bytes = await left.read() if left else None
    right_bytes = await right.read() if right else None
    rear_bytes = await rear.read() if rear else None

    if not left_bytes and not right_bytes and not rear_bytes:
        return JSONResponse(_default_payload("No images provided."), status_code=200)

    left_url = _file_to_data_url(left_bytes, left.filename) if left_bytes else None
    right_url = _file_to_data_url(right_bytes, right.filename) if right_bytes else None
    rear_url = _file_to_data_url(rear_bytes, rear.filename) if rear_bytes else None

    prompt = (
        "You analyse running shoe outsole wear patterns from photos.\n"
        "You MUST return JSON that matches the provided schema exactly where possible.\n\n"
        "Rules:\n"
        "- Always make a BEST-EFFORT estimate if the sole is visible.\n"
        "- Only return null for a side if that side's image is missing OR totally unusable.\n"
        "- If wear looks mild/symmetrical, still return a broad polygon + low-intensity heat points.\n"
        "- Set has_left_visual/has_right_visual TRUE whenever that sole image is present and visible.\n"
        "- Heat point intensity must be between 0 and 1.\n\n"
        "Output fields:\n"
        "- left_poly_points/right_poly_points: polygon around most worn region as [[x,y],...], at least 3 points.\n"
        "- left_heat_points/right_heat_points: 10–40 points {x,y,intensity} over the worn region.\n"
        "- analysis_text: concise user-facing explanation (hedged where needed).\n"
        "- confidence: 0..1 indicating how confident the model is about left/right bias.\n"
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

    # If OpenAI not configured, return a simple deterministic fallback to keep UI functional
    if not OPENAI_API_KEY or OpenAI is None:
        # Basic heuristic fallback: simple pixel-based proxies could be added here.
        # For now return a cautious neutral response so UI can display.
        left_overlay = _make_overlay_png(left_bytes, None, None) if left_bytes else None
        right_overlay = _make_overlay_png(right_bytes, None, None) if right_bytes else None
        return JSONResponse(
            {
                "ok": True,
                "analysis_text": "OpenAI not configured — returned fallback neutral estimate.",
                "confidence": 0.35,
                "notes": "No model key configured on server; this is a fallback.",
                "left_overlay_data_url": left_overlay,
                "right_overlay_data_url": right_overlay,
                "debug_internal": {},
            },
            status_code=200,
        )

    # Call OpenAI (Responses API) with json_schema response format (best-effort)
    try:
        client = _get_openai_client()

        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": PRONATION_SCHEMA},
            max_output_tokens=800,
        )

        data = _extract_structured_json(resp)

        # Ensure keys exist
        for k in PRONATION_SCHEMA["schema"]["required"]:
            if k not in data:
                data[k] = None

        # Force has_* true when images provided
        if left_bytes:
            data["has_left_visual"] = True
        if right_bytes:
            data["has_right_visual"] = True

        data["ok"] = True
        data["confidence"] = _clamp01(data.get("confidence", 0.35))

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
        return JSONResponse(_default_payload(f"Analyse failed: {str(e)}"), status_code=200)


# --------------------------
# Routes
# --------------------------

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    # Serve the embedded index so no extra frontend files are needed right now
    return HTMLResponse(content=INDEX_HTML, status_code=200)


@app.get("/health")
def health():
    return {"ok": True}


# Analyse endpoints (aliases to avoid 404s)
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
