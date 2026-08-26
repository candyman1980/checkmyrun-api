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

GRID_W = 10
GRID_H = 16

app = FastAPI(title="CheckMyRun")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ======= YOLO CROPPING =========
# ===============================

@lru_cache(maxsize=1)
def get_yolo_world():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["shoe sole", "outsole", "running shoe sole"])
    return model


def decode_image(base_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(base_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _encode_crop(img: np.ndarray, box: Tuple[int, int, int, int]):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(8, int((x2 - x1) * 0.04))
    pad_y = max(8, int((y2 - y1) * 0.04))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("The detected sole area was empty")
    _, enc = cv2.imencode(".png", crop)
    return enc.tobytes()


def _background_crop(img: np.ndarray):
    """Find the main object by comparing it with the image-border colour."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.concatenate((distance[0], distance[-1], distance[:, 0], distance[:, -1]))
    threshold = max(16.0, float(np.percentile(border_distance, 90)) + 8.0)
    mask = (distance > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= h * w * 0.08]
    if not candidates:
        return None
    x, y, cw, ch = cv2.boundingRect(max(candidates, key=cv2.contourArea))
    if cw < w * 0.18 or ch < h * 0.30:
        return None
    return (x, y, x + cw, y + ch)


def crop_sole(base_bytes: bytes):
    img = decode_image(base_bytes)
    h, w = img.shape[:2]
    if min(h, w) < 400:
        raise ValueError("Photo resolution is too low; use a photo at least 400 pixels wide and high")

    try:
        model = get_yolo_world()
        results = model.predict(img, imgsz=960, conf=0.08, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            best_index = int(np.argmax(confidences))
            box = tuple(map(int, boxes[best_index]))
            return _encode_crop(img, box), {
                "crop_method": "object_detector",
                "detection_confidence": round(float(confidences[best_index]), 3),
            }
    except Exception:
        # A detector/model failure must not discard an otherwise usable photo.
        pass

    fallback_box = _background_crop(img)
    if fallback_box is not None:
        return _encode_crop(img, fallback_box), {
            "crop_method": "background_segmentation",
            "detection_confidence": None,
        }

    # The upload guide already asks users to fill the frame with one sole. Using
    # the full image is safer than crashing or inventing a crop.
    return _encode_crop(img, (0, 0, w, h)), {
        "crop_method": "full_image_fallback",
        "detection_confidence": None,
    }


# ===============================
# ======= CV WEAR MAP ===========
# ===============================

def compute_cv_abrasion_map(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0

    lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F)
    texture = np.abs(lap)
    texture = texture / (texture.max() + 1e-6)

    smooth = 1 - texture

    g = arr[:, :, 1]
    r = arr[:, :, 0]
    b = arr[:, :, 2]

    non_green = 1 - np.clip(g - (r + b) / 2, 0, 1)

    wear = 0.6 * smooth + 0.4 * non_green
    wear = cv2.GaussianBlur(wear, (0, 0), sigmaX=5)

    return np.clip(wear, 0, 1)


def map_to_grid(score, w=10, h=16):
    H, W = score.shape
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            patch = score[
                int(y * H / h):int((y + 1) * H / h),
                int(x * W / w):int((x + 1) * W / w)
            ]
            row.append(float(np.mean(patch)))
        out.append(row)
    return out


# ===============================
# ======= OPENAI CALL ===========
# ===============================

SYSTEM_PROMPT = "Return JSON only with heat_grid."


def call_openai(left_url, right_url):
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_image", "image_url": right_url},
                ],
            }
        ],
        "max_output_tokens": 2000,
    }

    r = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json=payload,
        timeout=60,
    )

    return json.loads(r.text)


# ===============================
# ======= HEATMAP ===============
# ===============================

def make_heatmap(img_bytes, grid):
    base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = base.size

    g = np.array(grid)
    heat = cv2.resize(g, (w, h))
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=10)

    heat = heat / (heat.max() + 1e-6)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255 - heat * 255
    rgba[..., 3] = heat * 180

    overlay = Image.fromarray(rgba, "RGBA")
    out = Image.alpha_composite(base, overlay)

    buf = io.BytesIO()
    out.save(buf, format="PNG")

    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ===============================
# ======= API ===================
# ===============================

@app.get("/health")
def health():
    return {"ok": True, "marker": "HYBRID-V2"}


@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    try:
        left_bytes = await left.read()
        right_bytes = await right.read()

        left_crop, left_detection = crop_sole(left_bytes)
        right_crop, right_detection = crop_sole(right_bytes)

        left_cv = compute_cv_abrasion_map(left_crop)
        right_cv = compute_cv_abrasion_map(right_crop)

        left_grid = map_to_grid(left_cv)
        right_grid = map_to_grid(right_cv)

        return {
            "left_heatmap_data_url": make_heatmap(left_crop, left_grid),
            "right_heatmap_data_url": make_heatmap(right_crop, right_grid),
            "debug": {
                "left_grid": left_grid,
                "right_grid": right_grid,
            },
            "quality": {
                "left": left_detection,
                "right": right_detection,
            },
        }

    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
