import os
import json
import base64
import io
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageFilter

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

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

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>CheckMyRun</title>
  <style>
    body{
      font-family:system-ui,Segoe UI,Helvetica,Arial;
      margin:28px;
      background:#fff;
      color:#111;
    }
    .card{
      max-width:1280px;
      margin:0 auto;
      padding:18px;
      border-radius:12px;
      border:1px solid #eee;
      background:#fafafa;
    }
    .uploadGrid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
      gap:16px;
    }
    .uploadCard{
      background:#fff;
      border:1px solid #e5e5e5;
      border-radius:12px;
      padding:14px;
    }
    .uploadCard h3{
      margin:0 0 10px 0;
      font-size:18px;
    }
    .previewBox{
      margin-top:12px;
      min-height:220px;
      border:2px dashed #ddd;
      border-radius:10px;
      display:flex;
      align-items:center;
      justify-content:center;
      overflow:hidden;
      background:#fcfcfc;
    }
    .previewBox img{
      max-width:100%;
      max-height:320px;
      display:block;
    }
    .placeholder{
      color:#888;
      font-size:14px;
      text-align:center;
      padding:20px;
    }
    input[type=file]{
      display:block;
      margin-top:8px;
      width:100%;
    }
    button{
      padding:10px 14px;
      border-radius:8px;
      border:1px solid #111;
      background:#fff;
      cursor:pointer;
      font-weight:600;
    }
    .result-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
      gap:14px;
      margin-top:12px;
    }
    .result-card{
      background:#fff;
      border:1px solid #e5e5e5;
      border-radius:10px;
      padding:14px;
    }
    .heatmap-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
      gap:16px;
      margin-top:16px;
    }
    .heatmap-card{
      background:#fff;
      border:1px solid #e5e5e5;
      border-radius:10px;
      padding:14px;
    }
    .heatmap-card img{
      max-width:100%;
      border:1px solid #ddd;
      border-radius:8px;
      display:block;
      margin-top:10px;
    }
    .muted{
      color:#666;
    }
    pre{
      background:#f4f6f8;
      padding:12px;
      border-radius:8px;
      overflow:auto;
      max-height:360px;
      white-space:pre-wrap;
    }
    .sectionTitle{
      margin-top:28px;
      margin-bottom:10px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>CheckMyRun</h1>
    <p>Upload clear photos of both soles and a rear photo. You’ll get a pronation estimate, shoe category suggestion, and wear heatmaps underneath.</p>

    <form id="form" enctype="multipart/form-data">
      <div class="uploadGrid">
        <div class="uploadCard">
          <h3>Left sole</h3>
          <input id="leftInput" type="file" name="left" accept="image/*" required>
          <div class="previewBox">
            <img id="leftPreview" style="display:none" alt="Left preview">
            <div id="leftPlaceholder" class="placeholder">Left sole image preview will appear here</div>
          </div>
        </div>

        <div class="uploadCard">
          <h3>Right sole</h3>
          <input id="rightInput" type="file" name="right" accept="image/*" required>
          <div class="previewBox">
            <img id="rightPreview" style="display:none" alt="Right preview">
            <div id="rightPlaceholder" class="placeholder">Right sole image preview will appear here</div>
          </div>
        </div>

        <div class="uploadCard">
          <h3>Rear photo</h3>
          <input id="rearInput" type="file" name="rear" accept="image/*">
          <div class="previewBox">
            <img id="rearPreview" style="display:none" alt="Rear preview">
            <div id="rearPlaceholder" class="placeholder">Rear photo preview will appear here</div>
          </div>
        </div>
      </div>

      <div style="margin-top:16px">
        <button id="btn" type="submit">Analyse</button>
        <span id="status" style="margin-left:12px;color:#666"></span>
      </div>
    </form>

    <div id="result" style="margin-top:22px;display:none">
      <h2 class="sectionTitle">Result</h2>
      <div id="summary"></div>

      <div class="result-grid">
        <div class="result-card">
          <h3>Left</h3>
          <div id="leftResult"></div>
        </div>

        <div class="result-card">
          <h3>Right</h3>
          <div id="rightResult"></div>
        </div>

        <div class="result-card">
          <h3>Overall</h3>
          <div id="overallResult"></div>
        </div>

        <div class="result-card">
          <h3>Photo quality</h3>
          <div id="qualityResult"></div>
        </div>
      </div>

      <h2 class="sectionTitle">Heatmaps</h2>
      <div class="heatmap-grid">
        <div class="heatmap-card">
          <h3>Left heatmap</h3>
          <div id="leftHeatmapWrap" class="muted">No heatmap returned yet.</div>
        </div>

        <div class="heatmap-card">
          <h3>Right heatmap</h3>
          <div id="rightHeatmapWrap" class="muted">No heatmap returned yet.</div>
        </div>
      </div>

      <details style="margin-top:18px">
        <summary>Raw JSON</summary>
        <pre id="json"></pre>
      </details>

      <footer style="margin-top:14px;font-size:13px;color:#666">
        Informational only — not medical advice.
      </footer>
    </div>
  </div>

<script>
const API = "/analyze";

const form = document.getElementById("form");
const btn = document.getElementById("btn");
const status = document.getElementById("status");
const result = document.getElementById("result");
const summary = document.getElementById("summary");
const leftResult = document.getElementById("leftResult");
const rightResult = document.getElementById("rightResult");
const overallResult = document.getElementById("overallResult");
const qualityResult = document.getElementById("qualityResult");
const leftHeatmapWrap = document.getElementById("leftHeatmapWrap");
const rightHeatmapWrap = document.getElementById("rightHeatmapWrap");
const jsonOut = document.getElementById("json");

const leftInput = document.getElementById("leftInput");
const rightInput = document.getElementById("rightInput");
const rearInput = document.getElementById("rearInput");

const leftPreview = document.getElementById("leftPreview");
const rightPreview = document.getElementById("rightPreview");
const rearPreview = document.getElementById("rearPreview");

const leftPlaceholder = document.getElementById("leftPlaceholder");
const rightPlaceholder = document.getElementById("rightPlaceholder");
const rearPlaceholder = document.getElementById("rearPlaceholder");

function prettyLabel(v) {
  if (!v) return "—";
  return String(v).replace(/-/g, " ");
}

function showPreview(input, imgEl, placeholderEl) {
  const file = input.files && input.files[0];
  if (!file) {
    imgEl.style.display = "none";
    imgEl.src = "";
    placeholderEl.style.display = "block";
    return;
  }
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.style.display = "block";
  placeholderEl.style.display = "none";
}

leftInput.addEventListener("change", () => showPreview(leftInput, leftPreview, leftPlaceholder));
rightInput.addEventListener("change", () => showPreview(rightInput, rightPreview, rightPlaceholder));
rearInput.addEventListener("change", () => showPreview(rearInput, rearPreview, rearPlaceholder));

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  btn.disabled = true;
  status.textContent = "Uploading...";
  result.style.display = "none";
  summary.innerHTML = "";
  leftResult.innerHTML = "";
  rightResult.innerHTML = "";
  overallResult.innerHTML = "";
  qualityResult.innerHTML = "";
  leftHeatmapWrap.innerHTML = '<span class="muted">No heatmap returned yet.</span>';
  rightHeatmapWrap.innerHTML = '<span class="muted">No heatmap returned yet.</span>';
  jsonOut.textContent = "";

  try {
    const fd = new FormData(form);
    const res = await fetch(API, {
      method: "POST",
      body: fd
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || JSON.stringify(data));
    }

    summary.innerHTML = `
      <p><strong>Overall pronation:</strong> ${prettyLabel(data.overall?.pronation)}</p>
      <p><strong>Shoe category:</strong> ${prettyLabel(data.overall?.shoe_category)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.overall?.confidence || 0) * 100)}%</p>
    `;

    leftResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.left?.pronation)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.left?.confidence || 0) * 100)}%</p>
      <p>${data.left?.notes || ""}</p>
    `;

    rightResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.right?.pronation)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.right?.confidence || 0) * 100)}%</p>
      <p>${data.right?.notes || ""}</p>
    `;

    overallResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.overall?.pronation)}</p>
      <p><strong>Shoe category:</strong> ${prettyLabel(data.overall?.shoe_category)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.overall?.confidence || 0) * 100)}%</p>
    `;

    qualityResult.innerHTML = `
      <p><strong>OK:</strong> ${data.photo_quality?.ok ? "Yes" : "No"}</p>
      <p><strong>Issues:</strong></p>
      <ul>${(data.photo_quality?.issues || []).map(i => `<li>${i}</li>`).join("") || "<li>None</li>"}</ul>
    `;

    if (data.left_heatmap_data_url) {
      leftHeatmapWrap.innerHTML = `<img src="${data.left_heatmap_data_url}" alt="Left heatmap">`;
    }

    if (data.right_heatmap_data_url) {
      rightHeatmapWrap.innerHTML = `<img src="${data.right_heatmap_data_url}" alt="Right heatmap">`;
    }

    jsonOut.textContent = JSON.stringify(data, null, 2);
    result.style.display = "block";
    status.textContent = "Done ✅";
  } catch (err) {
    status.textContent = "Error";
    summary.innerHTML = `<p style="color:#b00020"><strong>${err.message}</strong></p>`;
    result.style.display = "block";
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=INDEX_HTML, status_code=200)

@app.get("/health")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V4-POINTS", "model": MODEL}

def upload_to_data_url(upload: UploadFile) -> tuple[str, bytes]:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")

    name = (upload.filename or "").lower()
    if name.endswith(".png"):
        mime = "image/png"
    elif name.endswith(".webp"):
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}", b

def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\\n".join(out).strip()

def clamp01(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x

def point_is_reasonable(px: float, py: float) -> bool:
    # reject image edges / obvious background
    return 0.12 <= px <= 0.88 and 0.08 <= py <= 0.96

def sanitise_heat_points(points):
    if not isinstance(points, list):
        return []

    clean = []
    for p in points:
        if not isinstance(p, dict):
            continue
        x = clamp01(p.get("x"))
        y = clamp01(p.get("y"))
        intensity = clamp01(p.get("intensity", 0.5))

        if not point_is_reasonable(x, y):
            continue

        clean.append({
            "x": x,
            "y": y,
            "intensity": max(0.2, intensity)
        })

    # dedupe points that are almost identical
    deduped = []
    for p in clean:
        too_close = False
        for q in deduped:
            if abs(p["x"] - q["x"]) < 0.03 and abs(p["y"] - q["y"]) < 0.03:
                too_close = True
                break
        if not too_close:
            deduped.append(p)

    return deduped[:8]

def make_heatmap_overlay(base_bytes: bytes, heat_points):
    if not base_bytes:
        return None

    points = sanitise_heat_points(heat_points)
    if len(points) < 2:
        return None

    try:
        base = Image.open(io.BytesIO(base_bytes)).convert("RGBA")
    except Exception:
        return None

    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(blobs, "RGBA")

    for p in points:
        x = p["x"] * w
        y = p["y"] * h
        intensity = p["intensity"]

        radius = max(28, int(min(w, h) * (0.045 + 0.08 * intensity)))
        alpha = int(90 + 110 * intensity)

        bdraw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 35, 0, alpha),
        )

    blur = max(10, int(min(w, h) * 0.02))
    blobs = blobs.filter(ImageFilter.GaussianBlur(radius=blur))
    overlay = Image.alpha_composite(overlay, blobs)

    combined = Image.alpha_composite(base, overlay)
    out = io.BytesIO()
    combined.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("utf-8")

@app.post("/analyze")
@app.post("/analyse")
@app.post("/api/analyze")
@app.post("/api/analyse")
async def analyze(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    rear: UploadFile = File(None),
):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url, left_bytes = upload_to_data_url(left)
        right_url, right_bytes = upload_to_data_url(right)
        rear_url = None
        if rear is not None and rear.filename:
            rear_url, _rear_bytes = upload_to_data_url(rear)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    response_schema = {
        "name": "checkmyrun_pronation_heatpoints",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "left": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "notes": {"type": "string"},
                        "heat_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "intensity": {"type": "number"}
                                },
                                "required": ["x", "y", "intensity"]
                            }
                        }
                    },
                    "required": ["pronation", "confidence", "notes", "heat_points"],
                },
                "right": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "notes": {"type": "string"},
                        "heat_points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "intensity": {"type": "number"}
                                },
                                "required": ["x", "y", "intensity"]
                            }
                        }
                    },
                    "required": ["pronation", "confidence", "notes", "heat_points"],
                },
                "overall": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                        },
                        "shoe_category": {
                            "type": "string",
                            "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]
                        },
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
                }
            },
            "required": ["left", "right", "overall", "photo_quality"],
        },
    }

    instruction = (
        "You are a running shoe fitting assistant. "
        "You will be given LEFT sole, RIGHT sole, and optionally a REAR shoe photo. "
        "Infer pronation style from visible wear patterns. "
        "Be conservative: if wear is unclear, output 'unclear' with low confidence. "
        "No medical advice. Notes must be short (1–2 sentences). "
        "Also assess photo quality and list issues. "
        "For each sole, return 4 to 8 heat_points only where visible outsole wear appears strongest. "
        "Use normalized coordinates from 0 to 1. "
        "Do not include background, hand, floor, walls, shadows, or photo edges. "
        "Do not trace the whole sole. "
        "If unclear, return an empty heat_points array. "
        "Return ONLY valid JSON matching the schema."
    )

    content = [
        {"type": "input_text", "text": instruction},
        {"type": "input_text", "text": "LEFT SOLE:"},
        {"type": "input_image", "image_url": left_url},
        {"type": "input_text", "text": "RIGHT SOLE:"},
        {"type": "input_image", "image_url": right_url},
    ]

    if rear_url:
        content.extend([
            {"type": "input_text", "text": "REAR PHOTO:"},
            {"type": "input_image", "image_url": rear_url},
        ])

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": content,
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
        "max_output_tokens": 900,
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

    left_heatmap = make_heatmap_overlay(
        left_bytes,
        data.get("left", {}).get("heat_points"),
    )
    right_heatmap = make_heatmap_overlay(
        right_bytes,
        data.get("right", {}).get("heat_points"),
    )

    data["left_heatmap_data_url"] = left_heatmap
    data["right_heatmap_data_url"] = right_heatmap

    return data
