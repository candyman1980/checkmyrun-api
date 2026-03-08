import base64
import io

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

cv2.setNumThreads(1)

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
    <p>Upload clear photos of both soles and a rear photo. You’ll get a local wear analysis and wear overlays underneath.</p>

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

      <h2 class="sectionTitle">Wear overlays</h2>
      <div class="heatmap-grid">
        <div class="heatmap-card">
          <h3>Left overlay</h3>
          <div id="leftHeatmapWrap" class="muted">No overlay returned yet.</div>
        </div>

        <div class="heatmap-card">
          <h3>Right overlay</h3>
          <div id="rightHeatmapWrap" class="muted">No overlay returned yet.</div>
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

function safeSetOverlay(container, dataUrl, altText) {
  container.innerHTML = "";
  if (!dataUrl || typeof dataUrl !== "string") {
    container.innerHTML = '<span class="muted">No overlay returned.</span>';
    return;
  }
  const trimmed = dataUrl.trim();
  if (!trimmed.startsWith("data:image/")) {
    container.innerHTML = '<span class="muted">Overlay returned in unexpected format.</span>';
    return;
  }
  try {
    const img = document.createElement("img");
    img.alt = altText;
    img.src = trimmed;
    container.appendChild(img);
  } catch (e) {
    container.innerHTML = '<span class="muted">Could not display overlay.</span>';
  }
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
  leftHeatmapWrap.innerHTML = '<span class="muted">No overlay returned yet.</span>';
  rightHeatmapWrap.innerHTML = '<span class="muted">No overlay returned yet.</span>';
  jsonOut.textContent = "";

  try {
    const fd = new FormData(form);
    const res = await fetch(API, {
      method: "POST",
      body: fd
    });

    const data = await res.json();
    jsonOut.textContent = JSON.stringify(data, null, 2);
    result.style.display = "block";

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

    safeSetOverlay(leftHeatmapWrap, data.left_heatmap_data_url, "Left overlay");
    safeSetOverlay(rightHeatmapWrap, data.right_heatmap_data_url, "Right overlay");

    status.textContent = "Done ✅";
  } catch (err) {
    status.textContent = "Error";
    summary.innerHTML = `<p style="color:#b00020"><strong>${err.message || String(err)}</strong></p>`;
    if (!jsonOut.textContent) {
      jsonOut.textContent = String(err);
    }
    result.style.display = "block";
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""

ZONE_RECTS = {
    "lateral_forefoot": (0.00, 0.00, 0.42, 0.34),
    "central_forefoot": (0.29, 0.14, 0.71, 0.42),
    "medial_forefoot": (0.58, 0.00, 1.00, 0.34),
    "lateral_midfoot": (0.02, 0.38, 0.35, 0.62),
    "medial_midfoot": (0.65, 0.38, 0.98, 0.62),
    "lateral_heel": (0.00, 0.68, 0.45, 1.00),
    "central_heel": (0.28, 0.72, 0.72, 1.00),
    "medial_heel": (0.55, 0.68, 1.00, 1.00),
}

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=INDEX_HTML, status_code=200)

@app.get("/health")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "LOCAL-WEAR-V3"}

def upload_to_bytes(upload: UploadFile) -> bytes:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")
    return b

def img_to_data_url_pil(img: Image.Image) -> str:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("utf-8")

def resize_for_processing(img_bgr: np.ndarray, max_size: int = 900) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr

def extract_sole_mask(base_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    arr = np.frombuffer(base_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_bgr = resize_for_processing(img_bgr, max_size=900)
    h, w = img_bgr.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    rect = (
        int(w * 0.10),
        int(h * 0.04),
        int(w * 0.80),
        int(h * 0.88),
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        gc_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype("uint8")
    except Exception:
        gc_mask = np.zeros((h, w), np.uint8)
        gc_mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255

    gc_mask[int(h * 0.96):, :] = 0

    contours, _ = cv2.findContours(gc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        sole_mask = np.zeros_like(gc_mask)
        cv2.drawContours(sole_mask, [contours[0]], -1, 255, thickness=cv2.FILLED)
    else:
        sole_mask = gc_mask

    kernel = np.ones((7, 7), np.uint8)
    sole_mask = cv2.morphologyEx(sole_mask, cv2.MORPH_CLOSE, kernel)
    sole_mask = cv2.morphologyEx(sole_mask, cv2.MORPH_OPEN, kernel)

    return img_bgr, sole_mask

def compute_wear_mask(base_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_bgr, sole_mask = extract_sole_mask(base_bytes)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # True-wear signal: low edge density + relatively brighter smooth rubber.
    edges = cv2.Canny(blur, 40, 120)
    edge_density = cv2.blur(edges, (21, 21))
    low_edges = cv2.inRange(edge_density, 0, 12)
    bright = cv2.inRange(gray, 110, 255)

    wear_candidate = cv2.bitwise_and(low_edges, bright)
    wear_candidate = cv2.bitwise_and(wear_candidate, sole_mask)

    kernel = np.ones((5, 5), np.uint8)
    wear_candidate = cv2.morphologyEx(wear_candidate, cv2.MORPH_OPEN, kernel)
    wear_candidate = cv2.morphologyEx(wear_candidate, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(wear_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wear_mask = np.zeros_like(wear_candidate)
    min_area = max(80, int(img_bgr.shape[0] * img_bgr.shape[1] * 0.0008))
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            cv2.drawContours(wear_mask, [c], -1, 255, thickness=cv2.FILLED)

    return img_bgr, sole_mask, wear_mask

def zone_score_from_mask(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)

def compute_zone_wear_scores_from_masks(sole_mask: np.ndarray, wear_mask: np.ndarray) -> dict:
    ys, xs = np.where(sole_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Could not isolate sole")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    scores = {}
    for name, (rx0, ry0, rx1, ry1) in ZONE_RECTS.items():
        zx0 = x0 + int((x1 - x0) * rx0)
        zy0 = y0 + int((y1 - y0) * ry0)
        zx1 = x0 + int((x1 - x0) * rx1)
        zy1 = y0 + int((y1 - y0) * ry1)
        scores[name] = zone_score_from_mask(wear_mask, zx0, zy0, zx1, zy1)

    vals = list(scores.values())
    vmin = min(vals)
    vmax = max(vals)

    norm_scores = {}
    for k, v in scores.items():
        if vmax - vmin < 1e-6:
            norm_scores[k] = 0.0
        else:
            norm_scores[k] = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))

    return norm_scores

def mirror_right_scores_to_left_frame(right_scores: dict) -> dict:
    mapping = {
        "lateral_forefoot": "medial_forefoot",
        "central_forefoot": "central_forefoot",
        "medial_forefoot": "lateral_forefoot",
        "lateral_midfoot": "medial_midfoot",
        "medial_midfoot": "lateral_midfoot",
        "lateral_heel": "medial_heel",
        "central_heel": "central_heel",
        "medial_heel": "lateral_heel",
    }
    return {mapping[k]: v for k, v in right_scores.items()}

def asymmetry_adjust_scores(left_scores: dict, right_scores: dict) -> tuple[dict, dict]:
    mirrored_right = mirror_right_scores_to_left_frame(right_scores)

    left_adj = left_scores.copy()
    right_adj = right_scores.copy()

    for left_zone, left_val in left_scores.items():
        right_val = mirrored_right.get(left_zone, 0.0)
        diff = left_val - right_val

        if abs(diff) > 0.10:
            if diff > 0:
                left_adj[left_zone] = min(1.0, left_val + 0.10)
            else:
                mirrored_zone = {
                    "lateral_forefoot": "medial_forefoot",
                    "central_forefoot": "central_forefoot",
                    "medial_forefoot": "lateral_forefoot",
                    "lateral_midfoot": "medial_midfoot",
                    "medial_midfoot": "lateral_midfoot",
                    "lateral_heel": "medial_heel",
                    "central_heel": "central_heel",
                    "medial_heel": "lateral_heel",
                }[left_zone]
                right_adj[mirrored_zone] = min(1.0, right_scores.get(mirrored_zone, 0.0) + 0.10)

    return left_adj, right_adj

def top_wear_zones(scores: dict, threshold: float = 0.40):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, score in ranked if score >= threshold][:4]

def make_mask_overlay(img_bgr: np.ndarray, sole_mask: np.ndarray, wear_mask: np.ndarray) -> str:
    base_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
    overlay = np.zeros_like(base_rgba, dtype=np.uint8)

    overlay[..., 0] = 255
    overlay[..., 1] = 45
    overlay[..., 2] = 0
    overlay[..., 3] = np.where(wear_mask > 0, 130, 0).astype(np.uint8)

    alpha = overlay[..., 3]
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=4, sigmaY=4)
    overlay[..., 3] = alpha

    contours, _ = cv2.findContours(sole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 120, 180), thickness=2)

    base_img = Image.fromarray(base_rgba, mode="RGBA")
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    combined = Image.alpha_composite(base_img, overlay_img)
    return img_to_data_url_pil(combined)

def infer_pronation_from_scores(left_scores: dict, right_scores: dict):
    def side_result(scores: dict):
        medial = max(scores.get("medial_forefoot", 0), scores.get("medial_heel", 0), scores.get("medial_midfoot", 0))
        lateral = max(scores.get("lateral_forefoot", 0), scores.get("lateral_heel", 0), scores.get("lateral_midfoot", 0))
        center = max(scores.get("central_forefoot", 0), scores.get("central_heel", 0))

        diff = medial - lateral
        strength = max(medial, lateral, center)

        if strength < 0.18:
            pronation = "unclear"
            confidence = 0.25
            notes = "Wear pattern is weak or not clearly distinguishable."
        elif diff > 0.12:
            pronation = "overpronation"
            confidence = min(0.9, 0.45 + diff + strength * 0.25)
            notes = "More wear appears on medial zones, suggesting inward roll."
        elif diff < -0.12:
            pronation = "underpronation"
            confidence = min(0.9, 0.45 + abs(diff) + strength * 0.25)
            notes = "More wear appears on lateral zones, suggesting outward loading."
        else:
            pronation = "neutral"
            confidence = min(0.85, 0.45 + strength * 0.25)
            notes = "Wear appears relatively balanced across the sole."
        return pronation, round(confidence, 2), notes

    lp, lc, ln = side_result(left_scores)
    rp, rc, rn = side_result(right_scores)

    votes = [lp, rp]
    if votes.count("overpronation") >= 2:
        overall = "overpronation"
        shoe_category = "stability"
    elif votes.count("underpronation") >= 2:
        overall = "underpronation"
        shoe_category = "cushioned-neutral"
    elif votes.count("neutral") >= 1:
        overall = "neutral"
        shoe_category = "neutral"
    else:
        overall = "unclear"
        shoe_category = "unclear"

    overall_conf = round((lc + rc) / 2, 2)

    return {
        "left": {"pronation": lp, "confidence": lc, "notes": ln},
        "right": {"pronation": rp, "confidence": rc, "notes": rn},
        "overall": {
            "pronation": overall,
            "shoe_category": shoe_category,
            "confidence": overall_conf,
        },
        "photo_quality": {
            "ok": True,
            "issues": []
        }
    }

@app.post("/analyze")
@app.post("/analyse")
@app.post("/api/analyze")
@app.post("/api/analyse")
async def analyze(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    rear: UploadFile = File(None),
):
    try:
        left_bytes = upload_to_bytes(left)
        right_bytes = upload_to_bytes(right)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    try:
        left_img, left_sole_mask, left_wear_mask = compute_wear_mask(left_bytes)
        right_img, right_sole_mask, right_wear_mask = compute_wear_mask(right_bytes)

        left_scores = compute_zone_wear_scores_from_masks(left_sole_mask, left_wear_mask)
        right_scores = compute_zone_wear_scores_from_masks(right_sole_mask, right_wear_mask)

        left_scores, right_scores = asymmetry_adjust_scores(left_scores, right_scores)

        left_overlay = make_mask_overlay(left_img, left_sole_mask, left_wear_mask)
        right_overlay = make_mask_overlay(right_img, right_sole_mask, right_wear_mask)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not analyse sole images: {e}")

    data = infer_pronation_from_scores(left_scores, right_scores)

    data["left"]["wear_zones"] = top_wear_zones(left_scores)
    data["right"]["wear_zones"] = top_wear_zones(right_scores)
    data["left_zone_scores"] = left_scores
    data["right_zone_scores"] = right_scores
    data["left_heatmap_data_url"] = left_overlay
    data["right_heatmap_data_url"] = right_overlay

    return data
