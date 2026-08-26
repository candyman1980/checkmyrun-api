import base64
import io
import json
import os
from functools import lru_cache
from typing import Dict, Tuple

import cv2
cv2.setNumThreads(1)

import httpx
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLOWorld


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_IMAGE_SIDE = 1800

app = FastAPI(title="CheckMyRun")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_yolo_world():
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["shoe outsole", "shoe sole", "running shoe outsole", "trainer sole"])
    return model


def prepare_image(raw: bytes) -> Tuple[bytes, np.ndarray]:
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as exc:
        raise ValueError("The uploaded file is not a readable photograph") from exc
    if min(pil.size) < 400:
        raise ValueError("Photo resolution is too low; use an image at least 400 pixels wide and high")
    scale = min(1.0, MAX_IMAGE_SIDE / max(pil.size))
    if scale < 1:
        pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return buffer.getvalue(), bgr


def image_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def background_box(img: np.ndarray):
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


def locate_sole(img: np.ndarray):
    h, w = img.shape[:2]
    try:
        result = get_yolo_world().predict(img, imgsz=960, conf=0.06, verbose=False)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            index = int(np.argmax(confidence))
            x1, y1, x2, y2 = map(int, boxes[index])
            return (x1, y1, x2, y2), "object_detector", round(float(confidence[index]), 3)
    except Exception:
        pass
    fallback = background_box(img)
    if fallback:
        return fallback, "background_segmentation", None
    return (0, 0, w, h), "full_image_fallback", None


ZONE_KEYS = [
    "heel_lateral", "heel_central", "heel_medial",
    "midfoot_lateral", "midfoot_medial",
    "forefoot_lateral", "forefoot_central", "forefoot_medial", "toe",
]

ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["usable", "confidence", "left", "right", "comparison", "limitations"],
    "properties": {
        "usable": {"type": "boolean"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "left": {
            "type": "object", "additionalProperties": False,
            "required": ZONE_KEYS,
            "properties": {key: {"type": "integer", "minimum": 0, "maximum": 3} for key in ZONE_KEYS},
        },
        "right": {
            "type": "object", "additionalProperties": False,
            "required": ZONE_KEYS,
            "properties": {key: {"type": "integer", "minimum": 0, "maximum": 3} for key in ZONE_KEYS},
        },
        "comparison": {"type": "string"},
        "limitations": {"type": "string"},
    },
}

ASSESSMENT_PROMPT = """You are assessing visible running-shoe outsole wear from two photographs.
Image 1 is the LEFT shoe; Image 2 is the RIGHT shoe. The user was told to point both toes away from the camera, with heels nearest the camera.

Score only visually supported loss of tread depth, rounded tread edges, smoothing/polishing, abrasion, or clearly asymmetric material loss. Use this scale for every zone: 0=no visible wear, 1=light, 2=moderate, 3=heavy. Compare corresponding regions across both shoes and use intact nearby tread as an internal reference. Do not treat shadows, dirt, wetness, printed colour, tread design, or the edge of the shoe as wear.

Set usable=false and confidence below 35 if either full sole is not visible, orientation is unclear, blur/glare prevents tread inspection, or wear cannot be distinguished from the original tread design. Keep confidence conservative. The comparison must describe only visible wear evidence. Do not diagnose gait, pronation, supination, injury risk, or a medical condition."""


def assess_zones(left_url: str, right_url: str) -> Dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("The visual assessment service is not configured")
    payload = {
        "model": OPENAI_MODEL,
        "store": False,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": ASSESSMENT_PROMPT + "\n\nIMAGE 1 — LEFT SHOE"},
                {"type": "input_image", "image_url": left_url, "detail": "high"},
                {"type": "input_text", "text": "IMAGE 2 — RIGHT SHOE"},
                {"type": "input_image", "image_url": right_url, "detail": "high"},
            ],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "shoe_wear_zones",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            }
        },
        "max_output_tokens": 3000,
    }
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json=payload,
        timeout=90,
    )
    if response.status_code >= 400:
        detail = response.json().get("error", {}).get("message", "Visual assessment request failed")
        raise RuntimeError(detail)
    body = response.json()
    output_text = body.get("output_text")
    if not output_text:
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    output_text = content.get("text")
                    break
    if not output_text:
        raise RuntimeError("The visual assessment returned no structured result")
    return json.loads(output_text)


def sole_mask(img: np.ndarray, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), np.uint8)
    if x1 > 1 or y1 > 1 or x2 < w - 1 or y2 < h - 1:
        grab = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img, grab, (x1, y1, max(2, x2 - x1), max(2, y2 - y1)), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
            mask = np.where((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            mask[y1:y2, x1:x2] = 255
    else:
        mask[y1:y2, x1:x2] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def zone_value(scores: Dict, y: float, x: float, side: str):
    medial = x > 0.5 if side == "left" else x < 0.5
    if y >= 0.82:
        return scores["toe"]
    if y >= 0.57:
        if x < 0.34 or x > 0.66:
            return scores["forefoot_medial" if medial else "forefoot_lateral"]
        return scores["forefoot_central"]
    if y >= 0.32:
        return scores["midfoot_medial" if medial else "midfoot_lateral"]
    if x < 0.34 or x > 0.66:
        return scores["heel_medial" if medial else "heel_lateral"]
    return scores["heel_central"]


def overlay_heatmap(jpeg: bytes, img: np.ndarray, box, scores: Dict, side: str):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    heat = np.zeros((h, w), np.float32)
    for py in range(max(0, y1), min(h, y2)):
        # The guide places the heel nearest the camera (top after display rotation is not guaranteed),
        # so normalized y is reversed: heel=0, toe=1.
        ny = (y2 - py) / box_h
        for px in range(max(0, x1), min(w, x2)):
            nx = (px - x1) / box_w
            heat[py, px] = zone_value(scores, ny, nx, side) / 3.0
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(7, box_w * 0.035))
    heat *= sole_mask(img, box).astype(np.float32) / 255.0
    colour = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_TURBO)
    alpha = np.clip(heat * 0.68, 0, 0.68)[..., None]
    composed = (img.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha).astype(np.uint8)
    output = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(output)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"ok": True, "marker": "ZONAL-V3"}


@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    try:
        left_jpeg, left_img = prepare_image(await left.read())
        right_jpeg, right_img = prepare_image(await right.read())
        left_box, left_method, left_detection = locate_sole(left_img)
        right_box, right_method, right_detection = locate_sole(right_img)
        assessment = assess_zones(image_data_url(left_jpeg), image_data_url(right_jpeg))
        if not assessment["usable"]:
            return JSONResponse({
                "detail": "These photographs do not show enough reliable tread detail for an assessment.",
                "assessment": assessment,
            }, status_code=422)
        return {
            "left_heatmap_data_url": overlay_heatmap(left_jpeg, left_img, left_box, assessment["left"], "left"),
            "right_heatmap_data_url": overlay_heatmap(right_jpeg, right_img, right_box, assessment["right"], "right"),
            "assessment": assessment,
            "quality": {
                "left": {"crop_method": left_method, "detection_confidence": left_detection},
                "right": {"crop_method": right_method, "detection_confidence": right_detection},
            },
        }
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
