import cv2
cv2.setNumThreads(1)

import base64
import io
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from ultralytics import YOLOWorld


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

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
      --accent:#22c55e;
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
      max-width:1280px;
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
    .hint ul{margin:8px 0 0 20px;padding:0}
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
      position:relative;
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
      position:relative;
      z-index:1;
    }
    .ghostGuide{
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      pointer-events:none;
      z-index:2;
    }
    .ghostGuide svg{
      width:74%;
      height:86%;
      opacity:.55;
      filter:drop-shadow(0 0 6px rgba(34,197,94,.18));
    }
    .ghostGuide .guideText{
      position:absolute;
      left:12px;
      right:12px;
      bottom:10px;
      font-size:13px;
      color:#555;
      text-align:center;
      background:rgba(255,255,255,.78);
      border:1px solid rgba(0,0,0,.06);
      border-radius:999px;
      padding:6px 10px;
      backdrop-filter: blur(4px);
    }
    .placeholder{
      color:#888;
      font-size:14px;
      text-align:center;
      padding:20px;
      line-height:1.45;
      position:relative;
      z-index:1;
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
      Best results:
      <ul>
        <li>one sole per frame, filling most of the image</li>
        <li>camera as straight-on as possible</li>
        <li>keep fingers low on the heel edge if you have to hold the shoe</li>
        <li>try to keep the full sole inside the green guide shape</li>
      </ul>
    </div>

    <form id="form" enctype="multipart/form-data">
      <div class="uploadGrid">
        <div class="uploadCard">
          <h3>Left sole</h3>
          <label class="frameLabel" for="leftInput">
            <div class="previewBox">
              <img id="leftPreview" style="display:none" alt="Left preview">
              <div class="ghostGuide" id="leftGuide">
                <svg viewBox="0 0 240 420" aria-hidden="true">
                  <path d="M122 10
                           C165 14, 203 46, 210 99
                           C215 142, 204 169, 188 205
                           C177 231, 173 259, 173 304
                           C173 360, 154 404, 121 410
                           C88 405, 68 360, 68 304
                           C68 259, 63 231, 52 205
                           C36 169, 25 142, 30 99
                           C37 46, 80 14, 122 10 Z"
                        fill="rgba(34,197,94,0.11)"
                        stroke="rgba(34,197,94,0.95)"
                        stroke-width="5"
                        stroke-dasharray="10 8"/>
                </svg>
                <div class="guideText">Centre the sole inside this guide</div>
              </div>
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
              <div class="ghostGuide" id="rightGuide">
                <svg viewBox="0 0 240 420" aria-hidden="true">
                  <path d="M122 10
                           C165 14, 203 46, 210 99
                           C215 142, 204 169, 188 205
                           C177 231, 173 259, 173 304
                           C173 360, 154 404, 121 410
                           C88 405, 68 360, 68 304
                           C68 259, 63 231, 52 205
                           C36 169, 25 142, 30 99
                           C37 46, 80 14, 122 10 Z"
                        fill="rgba(34,197,94,0.11)"
                        stroke="rgba(34,197,94,0.95)"
                        stroke-width="5"
                        stroke-dasharray="10 8"/>
                </svg>
                <div class="guideText">Centre the sole inside this guide</div>
              </div>
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
              <div class="ghostGuide" id="rearGuide">
                <svg viewBox="0 0 300 220" aria-hidden="true">
                  <path d="M85 145 C78 92, 98 58, 128 50 C157 58, 177 92, 170 145"
                        fill="none" stroke="rgba(34,197,94,0.95)" stroke-width="5" stroke-dasharray="10 8"/>
                  <path d="M130 145 C123 92, 143 58, 173 50 C202 58, 222 92, 215 145"
                        fill="none" stroke="rgba(34,197,94,0.95)" stroke-width="5" stroke-dasharray="10 8"/>
                </svg>
                <div class="guideText">Optional: heels level and centred</div>
              </div>
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

const leftGuide = document.getElementById("leftGuide");
const rightGuide = document.getElementById("rightGuide");
const rearGuide = document.getElementById("rearGuide");

function prettyLabel(v) {
  if (!v) return "—";
  return String(v).replace(/_/g, " ").replace(/-/g, " ");
}

function showPreview(input, imgEl, placeholderEl, metaEl, guideEl) {
  const file = input.files && input.files[0];
  if (!file) {
    imgEl.style.display = "none";
    imgEl.src = "";
    placeholderEl.style.display = "block";
    guideEl.style.display = "flex";
    metaEl.textContent = "";
    return;
  }
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.style.display = "block";
  placeholderEl.style.display = "none";
  guideEl.style.display = "flex";
  metaEl.textContent = file.name;
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

leftInput.addEventListener("change", () => showPreview(leftInput, leftPreview, leftPlaceholder, leftMeta, leftGuide));
rightInput.addEventListener("change", () => showPreview(rightInput, rightPreview, rightPlaceholder, rightMeta, rightGuide));
rearInput.addEventListener("change", () => showPreview(rearInput, rearPreview, rearPlaceholder, rearMeta, rearGuide));

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  btn.disabled = true;
  status.textContent = "Uploading...";
  result.style.display = "none";
  summary.innerHTML = "";
  leftResult.innerHTML = "";
  rightResult.innerHTML = "";
  overallResult.innerHTML = "";
  leftHeatmapWrap.innerHTML = '<span class="muted">No overlay returned yet.</span>';
  rightHeatmapWrap.innerHTML = '<span class="muted">No overlay returned yet.</span>';
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
      throw new Error("Server did not return valid JSON. Check Raw JSON below.");
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

    safeSetOverlay(leftHeatmapWrap, data.left_overlay_data_url, "Left overlay");
    safeSetOverlay(rightHeatmapWrap, data.right_overlay_data_url, "Right overlay");

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

VISION_SCHEMA: Dict[str, Any] = {
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
                        "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                    },
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"},
                    "wear_zones": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["pronation", "confidence", "notes", "wear_zones"]
            },
            "right": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {
                        "type": "string",
                        "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                    },
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"},
                    "wear_zones": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["pronation", "confidence", "notes", "wear_zones"]
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
                    "confidence": {"type": "number"}
                },
                "required": ["pronation", "shoe_category", "confidence"]
            },
            "left_outline": {
                "type": ["array", "null"],
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "right_outline": {
                "type": ["array", "null"],
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                }
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
                        "radius": {"type": "number"}
                    },
                    "required": ["x", "y", "intensity", "radius"]
                }
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
                        "radius": {"type": "number"}
                    },
                    "required": ["x", "y", "intensity", "radius"]
                }
            }
        },
        "required": [
            "analysis_text",
            "left",
            "right",
            "overall",
            "left_outline",
            "right_outline",
            "left_heat_points",
            "right_heat_points"
        ]
    }
}

SYSTEM_PROMPT = """
You are analysing running shoe outsole wear from photos.

You will receive cropped sole images that have already been localized, but may still contain small amounts of hand or background near the heel.
You must identify only the outsole itself and ignore hands, wrists, sleeves, watches, floor, bags, chairs, and background clutter.

Return JSON only.

Rules:
- Be object-aware. A human hand holding the heel is NOT part of the sole outline.
- The outline must trace the visible outsole boundary only.
- Outline points are normalized coordinates from 0 to 1 relative to the cropped image.
- Heat points must sit only on meaningful visible wear regions of the outsole.
- Do not spread heat across the whole sole.
- If evidence is weak, lower confidence and say so.
- Prefer useful insight over blandness.
- Notes should mention left/right asymmetry where relevant.

wear_zones examples:
- lateral heel
- central heel
- medial heel
- lateral forefoot
- central forefoot
- medial forefoot
- lateral midfoot
- medial midfoot
"""

USER_PROMPT = """
Analyse the attached cropped sole photos.

Return:
1. left/right/overall pronation judgement
2. a useful written analysis
3. normalized outsole outline for left and right
4. wear heat points for left and right

For the outline:
- use about 18 to 40 points if possible
- trace only the outsole edge
- do NOT include hand or wrist

For heat points:
- use around 8 to 25 points per sole
- each point must have x, y, intensity (0 to 1), radius (0.02 to 0.12)
- place points over genuine wear regions, not decorative tread unless clearly worn
"""


@lru_cache(maxsize=1)
def get_yolo_world():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["shoe sole", "outsole", "hand"])
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


def _clamp01(x: Any) -> float:
    try:
        x = float(x)
        return max(0.0, min(1.0, x))
    except Exception:
        return 0.0


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
    aspect = bh / bw
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
    sole_like = [d for d in detections if d["label"] in {"shoe sole", "outsole"}]
    if not sole_like:
        raise ValueError("Could not find the sole cleanly. Retake the photo with the sole larger in frame.")
    return max(sole_like, key=lambda d: score_sole_candidate(d["xyxy"], d["conf"], img_shape))["xyxy"]


def expand_box(box: List[float], img_shape: Tuple[int, int, int], pad_x: float = 0.10, pad_y: float = 0.08) -> Tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    nx1 = max(0, int(round(x1 - bw * pad_x)))
    ny1 = max(0, int(round(y1 - bh * pad_y)))
    nx2 = min(w, int(round(x2 + bw * pad_x)))
    ny2 = min(h, int(round(y2 + bh * pad_y)))
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


def _extract_json_from_response(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    output_text = resp_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return json.loads(output_text)

    for item in resp_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                txt = content["text"].strip()
                if txt:
                    return json.loads(txt)

    raise ValueError("Could not extract structured JSON from model response.")


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
        "max_output_tokens": 1800,
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

    return _extract_json_from_response(r.json())


def normalize_outline(points: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
    if not points or not isinstance(points, list):
        return None
    out: List[List[float]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        out.append([_clamp01(p[0]), _clamp01(p[1])])
    return out if len(out) >= 3 else None


def normalize_heat_points(points: Optional[List[Dict[str, Any]]]) -> List[Dict[str, float]]:
    if not points or not isinstance(points, list):
        return []
    out: List[Dict[str, float]] = []
    for p in points:
        if not isinstance(p, dict):
            continue
        out.append({
            "x": _clamp01(p.get("x", 0.0)),
            "y": _clamp01(p.get("y", 0.0)),
            "intensity": _clamp01(p.get("intensity", 0.4)),
            "radius": max(0.02, min(0.12, float(p.get("radius", 0.05)))),
        })
    return out


def make_overlay_png(base_img_bytes: bytes, outline_points: Optional[List[List[float]]], heat_points: List[Dict[str, float]]) -> str:
    base = Image.open(io.BytesIO(base_img_bytes)).convert("RGBA")
    w, h = base.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    if heat_points:
        heat_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_heat = ImageDraw.Draw(heat_layer, "RGBA")
        for hp in heat_points:
            x = hp["x"] * w
            y = hp["y"] * h
            r = hp["radius"] * min(w, h)
            intensity = hp["intensity"]
            alpha = int(110 + intensity * 90)
            draw_heat.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(255, 115, 35, alpha),
            )
        heat_layer = heat_layer.filter(ImageFilter.GaussianBlur(radius=max(6, min(w, h) * 0.012)))
        overlay = Image.alpha_composite(overlay, heat_layer)

    if outline_points and len(outline_points) >= 3:
        draw_outline = ImageDraw.Draw(overlay, "RGBA")
        pts = [(p[0] * w, p[1] * h) for p in outline_points]
        draw_outline.line(pts + [pts[0]], fill=(72, 255, 160, 230), width=max(3, int(min(w, h) * 0.005)))

    combined = Image.alpha_composite(base, overlay)
    out = io.BytesIO()
    combined.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
        "marker": "HYBRID-YOLO-OPENAI-V1",
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

        data["left_outline"] = normalize_outline(data.get("left_outline"))
        data["right_outline"] = normalize_outline(data.get("right_outline"))
        data["left_heat_points"] = normalize_heat_points(data.get("left_heat_points"))
        data["right_heat_points"] = normalize_heat_points(data.get("right_heat_points"))

        data["left_overlay_data_url"] = make_overlay_png(
            left_crop_bytes,
            data.get("left_outline"),
            data.get("left_heat_points", []),
        )
        data["right_overlay_data_url"] = make_overlay_png(
            right_crop_bytes,
            data.get("right_outline"),
            data.get("right_heat_points", []),
        )

        data["debug"] = {
            "left_yolo": left_debug,
            "right_yolo": right_debug,
            "rear_supplied": rear is not None,
        }

        return JSONResponse(content=data, status_code=200)

    except Exception as e:
        return JSONResponse(content=_default_payload(str(e)), status_code=500)
