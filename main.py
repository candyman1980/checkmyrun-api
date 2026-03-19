import cv2
cv2.setNumThreads(1)

import base64
import io
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from ultralytics import YOLOWorld


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Reduced grid so the model can reliably complete JSON
GRID_W = 10
GRID_H = 16

app = FastAPI(title="CheckMyRun")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>CheckMyRun</title>
  <style>
    :root{
      --bg:#fafafa;
      --card:#ffffff;
      --line:#e6e6e6;
      --muted:#666;
      --text:#111;
      --warn:#b00020;
    }
    *{box-sizing:border-box}
    body{
      font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;
      margin:24px;
      background:var(--bg);
      color:var(--text);
    }
    .wrap{
      max-width:1200px;
      margin:0 auto;
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      padding:20px;
    }
    h1{margin:0 0 10px 0}
    .hint{
      background:#f7f7f7;
      border:1px solid var(--line);
      border-radius:12px;
      padding:14px 16px;
      color:#444;
      line-height:1.45;
      margin-bottom:18px;
    }
    .uploadGrid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
      gap:16px;
    }
    .uploadCard{
      background:#fff;
      border:1px solid var(--line);
      border-radius:14px;
      padding:14px;
    }
    .uploadCard h3{
      margin:0 0 10px 0;
      font-size:20px;
    }
    .frameLabel{
      display:block;
      cursor:pointer;
    }
    .previewBox{
      min-height:220px;
      border:2px dashed #d5d5d5;
      border-radius:12px;
      overflow:hidden;
      background:#fcfcfc;
      display:flex;
      align-items:center;
      justify-content:center;
    }
    .previewBox:hover{
      border-color:#999;
      background:#f7f7f7;
    }
    .previewBox img{
      width:100%;
      height:auto;
      display:block;
      object-fit:contain;
      max-height:520px;
    }
    .placeholder{
      color:#888;
      font-size:14px;
      text-align:center;
      padding:20px;
      line-height:1.45;
    }
    .hiddenFileInput{display:none}
    .fileMeta{
      margin-top:10px;
      font-size:13px;
      color:var(--muted);
      min-height:18px;
      word-break:break-word;
    }
    button{
      padding:11px 16px;
      border-radius:10px;
      border:1px solid #111;
      background:#fff;
      cursor:pointer;
      font-weight:700;
    }
    .result-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
      gap:14px;
      margin-top:12px;
    }
    .result-card, .heatmap-card{
      background:#fff;
      border:1px solid var(--line);
      border-radius:12px;
      padding:14px;
    }
    .heatmap-grid{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
      gap:16px;
      margin-top:16px;
    }
    .heatmap-card img{
      max-width:100%;
      border:1px solid #ddd;
      border-radius:8px;
      display:block;
      margin-top:10px;
    }
    .muted{color:var(--muted)}
    pre{
      background:#f4f6f8;
      padding:12px;
      border-radius:8px;
      overflow:auto;
      max-height:420px;
      white-space:pre-wrap;
    }
    .sectionTitle{
      margin-top:26px;
      margin-bottom:10px;
    }
    .errorText{
      color:var(--warn);
      font-weight:700;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CheckMyRun</h1>

    <div class="hint">
      Upload left and right sole photos. The app crops each sole, analyses wear, and returns a dense heatmap plus interpretation.
    </div>

    <form id="form" enctype="multipart/form-data">
      <div class="uploadGrid">
        <div class="uploadCard">
          <h3>Left sole</h3>
          <label class="frameLabel" for="leftInput">
            <div class="previewBox">
              <img id="leftPreview" style="display:none" alt="Left preview">
              <div id="leftPlaceholder" class="placeholder">Tap here to upload the left sole photo</div>
            </div>
          </label>
          <input id="leftInput" class="hiddenFileInput" type="file" name="left" accept="image/*" required>
          <div id="leftMeta" class="fileMeta"></div>
        </div>

        <div class="uploadCard">
          <h3>Right sole</h3>
          <label class="frameLabel" for="rightInput">
            <div class="previewBox">
              <img id="rightPreview" style="display:none" alt="Right preview">
              <div id="rightPlaceholder" class="placeholder">Tap here to upload the right sole photo</div>
            </div>
          </label>
          <input id="rightInput" class="hiddenFileInput" type="file" name="right" accept="image/*" required>
          <div id="rightMeta" class="fileMeta"></div>
        </div>

        <div class="uploadCard">
          <h3>Rear photo</h3>
          <label class="frameLabel" for="rearInput">
            <div class="previewBox">
              <img id="rearPreview" style="display:none" alt="Rear preview">
              <div id="rearPlaceholder" class="placeholder">Tap here to upload the rear photo (optional)</div>
            </div>
          </label>
          <input id="rearInput" class="hiddenFileInput" type="file" name="rear" accept="image/*">
          <div id="rearMeta" class="fileMeta"></div>
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

const leftMeta = document.getElementById("leftMeta");
const rightMeta = document.getElementById("rightMeta");
const rearMeta = document.getElementById("rearMeta");

function prettyLabel(v) {
  if (!v) return "—";
  return String(v).replace(/_/g, " ").replace(/-/g, " ");
}

function showPreview(input, imgEl, placeholderEl, metaEl) {
  const file = input.files && input.files[0];
  if (!file) {
    imgEl.style.display = "none";
    imgEl.src = "";
    placeholderEl.style.display = "block";
    metaEl.textContent = "";
    return;
  }
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.style.display = "block";
  placeholderEl.style.display = "none";
  metaEl.textContent = file.name;
}

function safeSetImage(container, dataUrl, altText) {
  container.innerHTML = "";
  if (!dataUrl || typeof dataUrl !== "string") {
    container.innerHTML = '<span class="muted">No image returned.</span>';
    return;
  }
  if (!dataUrl.trim().startsWith("data:image/")) {
    container.innerHTML = '<span class="muted">Image returned in unexpected format.</span>';
    return;
  }
  const img = document.createElement("img");
  img.alt = altText;
  img.src = dataUrl;
  container.appendChild(img);
}

leftInput.addEventListener("change", () => showPreview(leftInput, leftPreview, leftPlaceholder, leftMeta));
rightInput.addEventListener("change", () => showPreview(rightInput, rightPreview, rightPlaceholder, rightMeta));
rearInput.addEventListener("change", () => showPreview(rearInput, rearPreview, rearPlaceholder, rearMeta));

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  btn.disabled = true;
  status.textContent = "Uploading...";
  result.style.display = "none";
  summary.innerHTML = "";
  leftResult.innerHTML = "";
  rightResult.innerHTML = "";
  overallResult.innerHTML = "";
  leftHeatmapWrap.innerHTML = '<span class="muted">No heatmap returned yet.</span>';
  rightHeatmapWrap.innerHTML = '<span class="muted">No heatmap returned yet.</span>';
  jsonOut.textContent = "";

  try {
    const fd = new FormData(form);
    const res = await fetch(API, { method: "POST", body: fd });

    const rawText = await res.text();
    jsonOut.textContent = rawText;
    result.style.display = "block";

    let data;
    try {
      data = JSON.parse(rawText);
    } catch {
      throw new Error(rawText || "Server did not return valid JSON.");
    }

    if (!res.ok) {
      throw new Error(data.detail || JSON.stringify(data));
    }

    summary.innerHTML = `
      <p><strong>Overall:</strong> ${data.analysis_text || "Analysis complete."}</p>
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

    safeSetImage(leftHeatmapWrap, data.left_heatmap_data_url, "Left heatmap");
    safeSetImage(rightHeatmapWrap, data.right_heatmap_data_url, "Right heatmap");

    status.textContent = "Done ✅";
  } catch (err) {
    status.textContent = "Error";
    summary.innerHTML = `<p class="errorText">${err.message || String(err)}</p>`;
    result.style.display = "block";
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""


@lru_cache(maxsize=1)
def get_yolo_world():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["shoe sole", "outsole", "running shoe sole", "hand"])
    return model


def _default_payload(error: str) -> Dict[str, Any]:
    return {"detail": error}


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


def decode_image(base_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(base_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")
    h, w = img_bgr.shape[:2]
    scale = 1280 / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def detect_world_boxes(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    model = get_yolo_world()
    results = model.predict(img_bgr, imgsz=640, conf=0.12, verbose=False)
    r = results[0]

    out: List[Dict[str, Any]] = []
    if r.boxes is None:
        return out

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names

    for box, conf, cls_id in zip(boxes, confs, clss):
        out.append({
            "xyxy": [float(v) for v in box.tolist()],
            "conf": float(conf),
            "label": str(names[int(cls_id)]).lower(),
        })
    return out


def score_sole_candidate(box: List[float], conf: float, img_shape: Tuple[int, int, int]) -> float:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area_ratio = (bw * bh) / float(h * w)
    aspect = bh / max(1.0, bw)
    cx = (x1 + x2) / 2.0
    center_score = 1.0 - abs(cx - (w / 2.0)) / (w / 2.0)

    if area_ratio < 0.04:
        return -1e9

    score = 0.0
    score += 1.6 * conf
    score += 2.0 * min(1.0, area_ratio / 0.24)
    score += 1.0 * max(0.0, min(1.0, center_score))
    score += 0.4 * max(0.0, min(2.0, aspect))
    return score


def choose_best_sole_box(detections: List[Dict[str, Any]], img_shape: Tuple[int, int, int]) -> List[float]:
    sole_like = [d for d in detections if d["label"] in {"shoe sole", "outsole", "running shoe sole"}]
    if not sole_like:
        raise ValueError("Could not find the sole cleanly. Retake the photo with the sole larger in frame.")
    return max(sole_like, key=lambda d: score_sole_candidate(d["xyxy"], d["conf"], img_shape))["xyxy"]


def expand_box(
    box: List[float],
    img_shape: Tuple[int, int, int],
    pad_x: float = 0.10,
    pad_y_top: float = 0.08,
    pad_y_bottom: float = 0.02,
) -> Tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    nx1 = max(0, int(round(x1 - bw * pad_x)))
    ny1 = max(0, int(round(y1 - bh * pad_y_top)))
    nx2 = min(w, int(round(x2 + bw * pad_x)))
    ny2 = min(h, int(round(y2 + bh * pad_y_bottom)))
    return nx1, ny1, nx2, ny2


def crop_from_yolo(base_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    img_bgr = decode_image(base_bytes)
    detections = detect_world_boxes(img_bgr)
    sole_box = choose_best_sole_box(detections, img_bgr.shape)
    crop_box = expand_box(sole_box, img_bgr.shape)

    x1, y1, x2, y2 = crop_box
    crop = img_bgr[y1:y2, x1:x2].copy()

    ok, enc = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Could not encode cropped sole image.")

    debug = {
        "detections": detections,
        "sole_box": sole_box,
        "crop_box": crop_box,
    }
    return enc.tobytes(), debug


def make_heat_grid_schema() -> Dict[str, Any]:
    row_schema = {
        "type": "array",
        "minItems": GRID_W,
        "maxItems": GRID_W,
        "items": {"type": "number"},
    }
    grid_schema = {
        "type": "array",
        "minItems": GRID_H,
        "maxItems": GRID_H,
        "items": row_schema,
    }

    return {
        "name": "checkmyrun_sole_analysis",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analysis_text": {"type": "string"},
                "left": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"],
                        },
                        "confidence": {"type": "number"},
                        "notes": {"type": "string"},
                        "wear_zones": {"type": "array", "items": {"type": "string"}},
                        "heat_grid": grid_schema,
                    },
                    "required": ["pronation", "confidence", "notes", "wear_zones", "heat_grid"],
                },
                "right": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"],
                        },
                        "confidence": {"type": "number"},
                        "notes": {"type": "string"},
                        "wear_zones": {"type": "array", "items": {"type": "string"}},
                        "heat_grid": grid_schema,
                    },
                    "required": ["pronation", "confidence", "notes", "wear_zones", "heat_grid"],
                },
                "overall": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"],
                        },
                        "shoe_category": {
                            "type": "string",
                            "enum": ["stability", "neutral", "cushioned-neutral", "unclear"],
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["pronation", "shoe_category", "confidence"],
                },
            },
            "required": ["analysis_text", "left", "right", "overall"],
        },
    }


VISION_SCHEMA = make_heat_grid_schema()

SYSTEM_PROMPT = f"""
You are analysing running shoe outsole wear from photos.

You will receive cropped sole images that have already been localized, but may still contain small amounts of hand or background near the heel.
You must identify only the outsole itself and ignore hands, wrists, sleeves, watches, floor, bags, chairs, and background clutter.

Return JSON only.

Rules:
- Be object-aware.
- Prefer useful insight over blandness.
- Lower confidence if the evidence is weak.
- Notes should mention asymmetry where relevant.
- wear_zones should use labels from this set when possible:
  lateral heel
  central heel
  medial heel
  lateral forefoot
  central forefoot
  medial forefoot
  lateral midfoot
  medial midfoot
- For each shoe, return a heat_grid of exactly {GRID_H} rows by {GRID_W} columns.
- Each cell is a number from 0.0 to 1.0 representing how worn that region looks.
- Higher values should cover visibly broad smoothed, polished, darkened, flattened, or abraded rubber.
- Be assertive rather than timid when wear is clearly visible.
- Broad obvious worn areas should occupy multiple neighbouring cells.
- Strong wear should use many 0.75 to 1.0 cells, not just mild mid-range values.
- Distinguish true hotspots from surrounding support regions.
- Do not light up background, hand, or untouched decorative tread.
- The grid is top-to-bottom and left-to-right over the cropped image.
"""

USER_PROMPT = f"""
Analyse the attached cropped sole photos.

Return:
1. left/right/overall pronation judgement
2. a useful written analysis
3. concise notes for each shoe
4. ordered wear_zones for each shoe from strongest to weaker
5. a dense {GRID_H}x{GRID_W} wear heat_grid for each shoe

Be more sensitive to broad visible wear, not just peak spots. Use higher values for clearly worn patches so the final heatmap shows convincing hotspots instead of only lukewarm regions.
"""


def _extract_json_from_response(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    def try_parse(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    output_text = resp_json.get("output_text")
    if isinstance(output_text, str):
        parsed = try_parse(output_text)
        if parsed is not None:
            return parsed

    for item in resp_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                txt = content.get("text", "")
                if isinstance(txt, str):
                    parsed = try_parse(txt)
                    if parsed is not None:
                        return parsed

                    match = re.search(r"\{[\s\S]*\}", txt)
                    if match:
                        parsed = try_parse(match.group(0))
                        if parsed is not None:
                            return parsed

    raise ValueError(f"Could not parse JSON from model response: {resp_json}")


def call_openai_vision(left_url: str, right_url: str, rear_url: Optional[str]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": USER_PROMPT},
        {"type": "input_text", "text": "LEFT CROPPED SOLE IMAGE"},
        {"type": "input_image", "image_url": left_url},
        {"type": "input_text", "text": "RIGHT CROPPED SOLE IMAGE"},
        {"type": "input_image", "image_url": right_url},
    ]
    if rear_url:
        content.append({"type": "input_text", "text": "OPTIONAL REAR IMAGE"})
        content.append({"type": "input_image", "image_url": rear_url})

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": VISION_SCHEMA["name"],
                "schema": VISION_SCHEMA["schema"],
                "strict": True,
            }
        },
        "max_output_tokens": 4000,
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if r.status_code != 200:
        raise ValueError(f"OpenAI error {r.status_code}: {r.text}")

    raw = r.json()

    if raw.get("status") == "incomplete":
        reason = raw.get("incomplete_details", {}).get("reason", "unknown")
        raise ValueError(f"Model response incomplete: {reason}")

    return _extract_json_from_response(raw)


def clamp01(x: Any) -> float:
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0


def normalise_grid(grid: Any) -> List[List[float]]:
    if not isinstance(grid, list) or len(grid) != GRID_H:
        return [[0.0 for _ in range(GRID_W)] for _ in range(GRID_H)]

    out: List[List[float]] = []
    for row in grid:
        if not isinstance(row, list) or len(row) != GRID_W:
            out.append([0.0 for _ in range(GRID_W)])
            continue
        cleaned = [clamp01(v) for v in row]
        cleaned = [0.0 if v < 0.08 else min(1.0, (v ** 0.72) * 1.22) for v in cleaned]
        out.append(cleaned)
    return out


def build_crop_mask(base_img_bytes: bytes) -> np.ndarray:
    base = Image.open(io.BytesIO(base_img_bytes)).convert("RGB")
    arr = np.array(base)
    h, w = arr.shape[:2]

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask1 = cv2.inRange(gray, 0, 248)
    mask2 = cv2.inRange(sat, 8, 255)
    mask3 = cv2.inRange(val, 15, 255)

    mask = cv2.bitwise_and(mask1, mask3)
    mask = cv2.bitwise_or(mask, mask2)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    mask[int(h * 0.975):, :] = 0
    return mask


def make_grid_heatmap(base_img_bytes: bytes, grid: List[List[float]]) -> str:
    base = Image.open(io.BytesIO(base_img_bytes)).convert("RGBA")
    w, h = base.size

    crop_mask = build_crop_mask(base_img_bytes).astype(np.float32) / 255.0
    crop_mask = cv2.GaussianBlur(crop_mask, (0, 0), sigmaX=max(3, int(min(w, h) * 0.006)))

    grid_arr = np.array(grid, dtype=np.float32)
    grid_arr = np.clip(grid_arr, 0.0, 1.0)

    grid_arr = np.where(grid_arr < 0.10, 0.0, grid_arr)
    grid_arr = np.clip((grid_arr ** 0.62) * 1.28, 0.0, 1.0)

    heat = cv2.resize(grid_arr, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=max(10, int(min(w, h) * 0.018)),
        sigmaY=max(10, int(min(w, h) * 0.018)),
    )

    heat = np.clip((heat ** 0.82) * 1.18, 0.0, 1.0)
    heat *= crop_mask
    heat = np.clip(heat, 0.0, 1.0)

    if float(heat.max()) <= 1e-6:
        out = io.BytesIO()
        base.save(out, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(out.getvalue()).decode('utf-8')}"

    heat = heat / float(heat.max())

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.clip(250 - heat * 250, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(85 - heat * 85, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip((heat ** 0.82) * 245, 0, 245).astype(np.uint8)

    overlay = Image.fromarray(rgba, "RGBA")
    combined = Image.alpha_composite(base, overlay)

    out = io.BytesIO()
    combined.save(out, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(out.getvalue()).decode('utf-8')}"


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=INDEX_HTML, status_code=200)


@app.head("/")
def head_root():
    return HTMLResponse(status_code=200)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "checkmyrun-api",
        "marker": "CHATGPT-GRID-HEATMAP-V3",
        "model": OPENAI_MODEL,
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
        left_bytes = await left.read()
        right_bytes = await right.read()
        rear_bytes = await rear.read() if rear else None

        if not left_bytes or not right_bytes:
            return JSONResponse(content=_default_payload("Left and right images are required."), status_code=400)

        left_crop_bytes, left_debug = crop_from_yolo(left_bytes)
        right_crop_bytes, right_debug = crop_from_yolo(right_bytes)

        left_url = _file_to_data_url(left_crop_bytes, "left_crop.png")
        right_url = _file_to_data_url(right_crop_bytes, "right_crop.png")
        rear_url = _file_to_data_url(rear_bytes, rear.filename or "rear.jpg") if rear_bytes else None

        data = call_openai_vision(left_url, right_url, rear_url)

        left_grid = normalise_grid(data.get("left", {}).get("heat_grid"))
        right_grid = normalise_grid(data.get("right", {}).get("heat_grid"))

        data["left_heatmap_data_url"] = make_grid_heatmap(left_crop_bytes, left_grid)
        data["right_heatmap_data_url"] = make_grid_heatmap(right_crop_bytes, right_grid)

        data["debug"] = {
            "left_yolo": left_debug,
            "right_yolo": right_debug,
            "rear_supplied": rear is not None,
        }

        return JSONResponse(content=data, status_code=200)

    except Exception as e:
        return JSONResponse(content=_default_payload(str(e)), status_code=500)
