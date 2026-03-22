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


def crop_from_yolo(base_bytes: bytes):
    img = decode_image(base_bytes)
    model = get_yolo_world()
    results = model.predict(img, imgsz=640, conf=0.2, verbose=False)[0]

    if results.boxes is None:
        raise ValueError("No sole detected")

    boxes = results.boxes.xyxy.cpu().numpy()
    best = boxes[0]
    x1, y1, x2, y2 = map(int, best)

    crop = img[y1:y2, x1:x2]
    _, enc = cv2.imencode(".png", crop)

    return enc.tobytes()


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

        left_crop = crop_from_yolo(left_bytes)
        right_crop = crop_from_yolo(right_bytes)

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
        }

    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
