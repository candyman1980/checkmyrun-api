import os
import json
import base64
import io
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageFilter, ImageStat

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
    <p>Upload clear photos of both soles and a rear photo. You’ll get a pronation estimate, shoe category suggestion, and heatmaps underneath.</p>

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
  return String(v).replace(/_/g, " ").replace(/-/g, " ");
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
      <p><strong>Wear zones:</strong> ${(data.left?.wear_zones || []).map(prettyLabel).join(", ") || "None obvious"}</p>
    `;

    rightResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.right?.pronation)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.right?.confidence || 0) * 100)}%</p>
      <p>${data.right?.notes || ""}</p>
      <p><strong>Wear zones:</strong> ${(data.right?.wear_zones || []).map(prettyLabel).join(", ") || "None obvious"}</p>
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
    return {"ok": True, "service": "checkmyrun-api", "marker": "ZONE-V1", "model": MODEL}

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

def img_to_data_url(img: Image.Image) -> str:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("utf-8")

def compute_zone_wear_scores(base_bytes: bytes):
    """
    Local wear scoring from the sole image itself.
    We use fixed zones and score them by low saturation + darker/greyer appearance.
    This is crude but far more stable than arbitrary model points.
    """
    img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
    w, h = img.size

    # Ignore very outer margins and very bottom hand region as much as possible.
    x0 = int(w * 0.12)
    x1 = int(w * 0.88)
    y0 = int(h * 0.06)
    y1 = int(h * 0.90)

    # Zone rectangles in normalized sole space (relative to cropped working area)
    zones = {
        "lateral_forefoot": (0.00, 0.00, 0.42, 0.34),
        "central_forefoot": (0.29, 0.14, 0.71, 0.42),
        "medial_forefoot": (0.58, 0.00, 1.00, 0.34),
        "lateral_midfoot": (0.02, 0.38, 0.35, 0.62),
        "medial_midfoot": (0.65, 0.38, 0.98, 0.62),
        "lateral_heel": (0.00, 0.68, 0.45, 1.00),
        "central_heel": (0.28, 0.72, 0.72, 1.00),
        "medial_heel": (0.55, 0.68, 1.00, 1.00),
    }

    scores = {}

    for name, (rx0, ry0, rx1, ry1) in zones.items():
        zx0 = x0 + int((x1 - x0) * rx0)
        zy0 = y0 + int((y1 - y0) * ry0)
        zx1 = x0 + int((x1 - x0) * rx1)
        zy1 = y0 + int((y1 - y0) * ry1)

        crop = img.crop((zx0, zy0, zx1, zy1)).convert("HSV")
        stat = ImageStat.Stat(crop)
        h_mean, s_mean, v_mean = stat.mean

        # Lower saturation + lower brightness tends to correlate with worn greyed rubber.
        sat_score = max(0.0, 1.0 - (s_mean / 255.0))
        dark_score = max(0.0, 1.0 - (v_mean / 255.0))

        # Weighted wear score.
        wear = (0.62 * sat_score) + (0.38 * dark_score)
        scores[name] = max(0.0, min(1.0, wear))

    # Normalize relative to this shoe's own zone distribution.
    vals = list(scores.values())
    vmin = min(vals)
    vmax = max(vals)
    norm_scores = {}
    for k, v in scores.items():
        if vmax - vmin < 0.05:
            norm_scores[k] = 0.2
        else:
            norm_scores[k] = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))

    return norm_scores

def top_wear_zones(scores: dict, threshold: float = 0.58):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, score in ranked if score >= threshold][:3]

def make_zone_heatmap_overlay(base_bytes: bytes, zone_scores: dict):
    if not base_bytes or not zone_scores:
        return None

    base = Image.open(io.BytesIO(base_bytes)).convert("RGBA")
    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    x0 = int(w * 0.12)
    x1 = int(w * 0.88)
    y0 = int(h * 0.06)
    y1 = int(h * 0.90)

    zones = {
        "lateral_forefoot": (0.00, 0.00, 0.42, 0.34),
        "central_forefoot": (0.29, 0.14, 0.71, 0.42),
        "medial_forefoot": (0.58, 0.00, 1.00, 0.34),
        "lateral_midfoot": (0.02, 0.38, 0.35, 0.62),
        "medial_midfoot": (0.65, 0.38, 0.98, 0.62),
        "lateral_heel": (0.00, 0.68, 0.45, 1.00),
        "central_heel": (0.28, 0.72, 0.72, 1.00),
        "medial_heel": (0.55, 0.68, 1.00, 1.00),
    }

    blobs = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(blobs, "RGBA")

    for name, score in zone_scores.items():
        if score < 0.35:
            continue

        rx0, ry0, rx1, ry1 = zones[name]
        zx0 = x0 + int((x1 - x0) * rx0)
        zy0 = y0 + int((y1 - y0) * ry0)
        zx1 = x0 + int((x1 - x0) * rx1)
        zy1 = y0 + int((y1 - y0) * ry1)

        cx = (zx0 + zx1) / 2
        cy = (zy0 + zy1) / 2
        radius_x = max(36, int((zx1 - zx0) * 0.55))
        radius_y = max(36, int((zy1 - zy0) * 0.55))
        alpha = int(70 + score * 120)

        bdraw.ellipse(
            (cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y),
            fill=(255, 45, 0, alpha),
        )

    blobs = blobs.filter(ImageFilter.GaussianBlur(radius=max(12, int(min(w, h) * 0.02))))
    overlay = Image.alpha_composite(overlay, blobs)

    combined = Image.alpha_composite(base, overlay)
    return img_to_data_url(combined)

def build_model_summary_prompt(left_scores, right_scores, rear_present: bool):
    return (
        "You are a running shoe fitting assistant. "
        "You are given zone wear scores derived from outsole photos. "
        "Use them conservatively to infer likely pronation. "
        "Do not invent certainty. "
        "Return JSON only.\n\n"
        f"Left zone scores: {json.dumps(left_scores)}\n"
        f"Right zone scores: {json.dumps(right_scores)}\n"
        f"Rear photo present: {rear_present}\n"
    )

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
        rear_present = bool(rear is not None and rear.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    left_scores = compute_zone_wear_scores(left_bytes)
    right_scores = compute_zone_wear_scores(right_bytes)

    left_wear_zones = top_wear_zones(left_scores)
    right_wear_zones = top_wear_zones(right_scores)

    # Ask model to interpret zone scores, not raw image coordinates.
    response_schema = {
        "name": "checkmyrun_zone_summary",
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
                    },
                    "required": ["pronation", "confidence", "notes"],
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
                    },
                    "required": ["pronation", "confidence", "notes"],
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

    instruction = build_model_summary_prompt(left_scores, right_scores, rear_present)

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction}
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
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Model returned non-JSON unexpectedly", "raw": text}

    data["left"]["wear_zones"] = left_wear_zones
    data["right"]["wear_zones"] = right_wear_zones
    data["left_zone_scores"] = left_scores
    data["right_zone_scores"] = right_scores

    data["left_heatmap_data_url"] = make_zone_heatmap_overlay(left_bytes, left_scores)
    data["right_heatmap_data_url"] = make_zone_heatmap_overlay(right_bytes, right_scores)

    return data
