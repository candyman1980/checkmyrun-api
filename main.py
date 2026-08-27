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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
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
    "required": ["usable", "confidence", "left", "right", "left_patches", "right_patches", "comparison", "limitations"],
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
        "left_patches": {
            "type": "array", "maxItems": 14,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["x", "y", "width", "height", "intensity", "evidence"],
                "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 100},
                    "y": {"type": "integer", "minimum": 0, "maximum": 100},
                    "width": {"type": "integer", "minimum": 3, "maximum": 50},
                    "height": {"type": "integer", "minimum": 3, "maximum": 50},
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "evidence": {"type": "string", "enum": ["smoothing", "rounded_edges", "tread_loss", "abrasion"]},
                },
            },
        },
        "right_patches": {
            "type": "array", "maxItems": 14,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["x", "y", "width", "height", "intensity", "evidence"],
                "properties": {
                    "x": {"type": "integer", "minimum": 0, "maximum": 100},
                    "y": {"type": "integer", "minimum": 0, "maximum": 100},
                    "width": {"type": "integer", "minimum": 3, "maximum": 50},
                    "height": {"type": "integer", "minimum": 3, "maximum": 50},
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "evidence": {"type": "string", "enum": ["smoothing", "rounded_edges", "tread_loss", "abrasion"]},
                },
            },
        },
        "comparison": {"type": "string"},
        "limitations": {"type": "string"},
    },
}

ASSESSMENT_PROMPT = """Assess visible running-shoe outsole wear. You receive a clean crop and a coordinate-grid copy for each sole. The grid copy is ONLY for locating evidence; inspect tread detail in the clean crop. Coordinates are percentages within each crop: x=0 left edge, x=100 right edge, y=0 top, y=100 bottom.

First compare repeated tread blocks within the same sole and corresponding areas across both soles. Detect only local patches where the rubber is visibly smoother or polished, tread edges are rounded, grooves/lugs have visibly lost depth, or abrasion/material loss is clear. Smooth factory rubber, recessed channels, shadows, dirt, glare, printed colour and naturally different tread compounds are not wear. Do not mark a broad zone when only a small patch is supported.

For each supported patch, return its centre x/y and a tight elliptical width/height. Intensity: 1=slight but visible smoothing/rounding, 2=clear flattening or tread-depth reduction, 3=pronounced material/tread loss. Return no patch when wear is not visually distinguishable. The nine zone scores must summarize these same patches using 0=no visible wear, 1=light, 2=moderate, 3=heavy.

Set usable=false and confidence below 35 if either sole is incomplete, perspective is strongly oblique, blur/glare hides tread, or original design cannot be distinguished from wear. Confidence measures evidence quality, not certainty about gait. Do not diagnose gait, pronation, supination, injury risk or a medical condition."""


def analysis_crop_data_url(img: np.ndarray, box, with_grid: bool = False) -> str:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = img.copy()
    if with_grid:
        ch, cw = crop.shape[:2]
        for index in range(1, 10):
            x, y = round(cw * index / 10), round(ch * index / 10)
            cv2.line(crop, (x, 0), (x, ch), (255, 255, 255), max(1, cw // 500))
            cv2.line(crop, (0, y), (cw, y), (255, 255, 255), max(1, cw // 500))
        for row in range(10):
            for col in range(10):
                label = f"{col}{row}"
                cv2.putText(crop, label, (round(cw * col / 10) + 3, round(ch * row / 10) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(crop, label, (round(cw * col / 10) + 3, round(ch * row / 10) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError("Could not prepare the sole crop")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


def assess_zones(left_clean: str, left_grid: str, right_clean: str, right_grid: str) -> Dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("The visual assessment service is not configured")
    payload = {
        "model": OPENAI_MODEL,
        "store": False,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": ASSESSMENT_PROMPT + "\n\nLEFT SHOE — CLEAN CROP"},
                {"type": "input_image", "image_url": left_clean, "detail": "high"},
                {"type": "input_text", "text": "LEFT SHOE — COORDINATE GUIDE"},
                {"type": "input_image", "image_url": left_grid, "detail": "high"},
                {"type": "input_text", "text": "RIGHT SHOE — CLEAN CROP"},
                {"type": "input_image", "image_url": right_clean, "detail": "high"},
                {"type": "input_text", "text": "RIGHT SHOE — COORDINATE GUIDE"},
                {"type": "input_image", "image_url": right_grid, "detail": "high"},
            ],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "shoe_wear_patches",
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


def overlay_heatmap(img: np.ndarray, box, patches):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    heat = np.zeros((h, w), np.float32)
    for patch in patches:
        centre = (round(x1 + box_w * patch["x"] / 100), round(y1 + box_h * patch["y"] / 100))
        axes = (max(3, round(box_w * patch["width"] / 200)), max(3, round(box_h * patch["height"] / 200)))
        layer = np.zeros((h, w), np.float32)
        cv2.ellipse(layer, centre, axes, 0, 0, 360, float(patch["intensity"]) / 3.0, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(3, min(axes) * 0.18))
        heat = np.maximum(heat, layer)
    heat *= sole_mask(img, box).astype(np.float32) / 255.0
    colour = np.zeros_like(img)
    colour[:, :] = (24, 24, 238)
    alpha = np.clip(heat * 0.62, 0, 0.62)[..., None]
    composed = (img.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha).astype(np.uint8)
    output = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(output)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"ok": True, "marker": "PATCH-RED-V4"}


@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    try:
        left_jpeg, left_img = prepare_image(await left.read())
        right_jpeg, right_img = prepare_image(await right.read())
        left_box, left_method, left_detection = locate_sole(left_img)
        right_box, right_method, right_detection = locate_sole(right_img)
        assessment = assess_zones(
            analysis_crop_data_url(left_img, left_box),
            analysis_crop_data_url(left_img, left_box, True),
            analysis_crop_data_url(right_img, right_box),
            analysis_crop_data_url(right_img, right_box, True),
        )
        if not assessment["usable"]:
            return JSONResponse({
                "detail": "These photographs do not show enough reliable tread detail for an assessment.",
                "assessment": assessment,
            }, status_code=422)
        return {
            "left_heatmap_data_url": overlay_heatmap(left_img, left_box, assessment["left_patches"]),
            "right_heatmap_data_url": overlay_heatmap(right_img, right_box, assessment["right_patches"]),
            "assessment": assessment,
            "quality": {
                "left": {"crop_method": left_method, "detection_confidence": left_detection},
                "right": {"crop_method": right_method, "detection_confidence": right_detection},
            },
        }
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
