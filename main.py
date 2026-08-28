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
GRID_COLS = 14
GRID_ROWS = 28
GRID_CELLS = GRID_COLS * GRID_ROWS
GAVIOTA_5_REFERENCE_URL = "https://media.au.hoka.com/cdn-cgi/image/fit%3Dscale-down%2Cf%3Dauto%2Cw%3D1280/products/7f6b704b-e124-447f-a3e0-76de84263d5f/7ada0c6d/1134235-hmrg_hmrg_08.jpg"

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
    "required": ["usable", "confidence", "left", "right", "left_grid", "right_grid", "comparison", "limitations"],
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
        "left_grid": {"type": "array", "minItems": GRID_CELLS, "maxItems": GRID_CELLS,
                      "items": {"type": "integer", "minimum": 0, "maximum": 3}},
        "right_grid": {"type": "array", "minItems": GRID_CELLS, "maxItems": GRID_CELLS,
                       "items": {"type": "integer", "minimum": 0, "maximum": 3}},
        "comparison": {"type": "string"},
        "limitations": {"type": "string"},
    },
}

ASSESSMENT_PROMPT = f"""Map visible Hoka Gaviota 5 outsole wear at high spatial precision. The first image is an UNWORN GAVIOTA 5 STRUCTURAL REFERENCE. Its colourway, lighting, scale and rotation are irrelevant. Use only its manufactured geometry: rubber-pad outlines, grooves, ribs, moulded lines, cut-outs and channels. After it, you receive each user's untouched full photograph followed by a tight {GRID_COLS}-column by {GRID_ROWS}-row location guide. Read the grid arrays row-major: all {GRID_COLS} cells of row 0 left-to-right, then row 1, through row {GRID_ROWS - 1}. In the required capture position the rounded TOE is the far/top end of the guide (approximately rows 0-6) and the HEEL is the near/bottom end, usually nearest the hand (approximately rows 21-27). Verify this from the sole anatomy, and never place a heel finding in forefoot rows or a toe finding in heel rows.

First geometrically align the reference outsole to each user sole using pad outlines, channels and cut-outs—not colour. Then compare corresponding manufactured details. The governing rule is CONTINUITY OF MANUFACTURED TEXTURE. Trace the reference's man-made lines, ribs, contours, stippling and fine mould texture through every corresponding rubber pad. Where expected reference detail becomes faint, interrupted or absent in the user photo without an intentional boundary, the smooth gap is wear and must be highlighted. Confirm with neighbouring texture and the matching shoe. Inspect the TOE PAD and HEEL PAD separately; these high-contact areas must not be skipped. Mark the full smooth interruption, not merely its boundary.

CRITICAL DISTINCTION: visible man-made lines and contours are the reference pattern, not wear. Preserve them, but highlight the adjacent smooth rubber wherever the pattern that should continue no longer exists. A few surviving large contours do not make the surrounding rubber unworn when finer lines have disappeared. Default an unexplained smooth gap inside a patterned rubber pad to wear, not factory-smooth. Call it factory-smooth only when a crisp intentional boundary or repeated identical examples prove that design. Exclude recessed foam/channels, dirt, shadows, glare, colour changes and photographic blur.

For every grid cell return 0 when outside rubber, uncertain, intentionally factory-smooth, or fine manufactured texture remains substantially intact; 1 when any meaningful part has credible local texture loss; 2 for clearly smooth/flattened rubber replacing expected texture; 3 for pronounced polished smoothness or material loss. A cell may be nonzero when only part is worn. Prefer sensitivity over omission once adjacent evidence establishes what texture should continue. Before finishing, perform a mandatory second visual sweep of the outer toe edge, central toe pad, outer heel edge and central heel pad and add every supported smooth interruption. The nine zone scores summarize the same evidence: 0 none, 1 light, 2 moderate, 3 heavy.

Set usable=false and confidence below 35 if either sole is incomplete, strongly oblique, blurred, glared, or the original design cannot be inferred. Confidence measures photographic evidence only. Do not diagnose gait, pronation, supination, injury risk or a medical condition."""


def analysis_crop_data_url(img: np.ndarray, box, with_grid: bool = False) -> str:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = img.copy()
    if with_grid:
        ch, cw = crop.shape[:2]
        for index in range(1, GRID_COLS):
            x = round(cw * index / GRID_COLS)
            cv2.line(crop, (x, 0), (x, ch), (255, 255, 255), max(1, cw // 500))
        for index in range(1, GRID_ROWS):
            y = round(ch * index / GRID_ROWS)
            cv2.line(crop, (0, y), (cw, y), (255, 255, 255), max(1, cw // 500))
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                label = f"{row:02d},{col}"
                origin = (round(cw * col / GRID_COLS) + 2, round(ch * row / GRID_ROWS) + 12)
                cv2.putText(crop, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(crop, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError("Could not prepare the sole crop")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


def request_assessment(content) -> Dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("The visual assessment service is not configured")
    payload = {
        "model": OPENAI_MODEL,
        "store": False,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "shoe_wear_patches",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            }
        },
        "max_output_tokens": 8000,
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


def assess_zones(left_original: str, left_grid: str, right_original: str, right_grid: str) -> Dict:
    images = [
        {"type": "input_text", "text": ASSESSMENT_PROMPT + "\n\nUNWORN HOKA GAVIOTA 5 — STRUCTURAL REFERENCE"},
        {"type": "input_image", "image_url": GAVIOTA_5_REFERENCE_URL, "detail": "high"},
        {"type": "input_text", "text": "LEFT SHOE — UNTOUCHED FULL PHOTOGRAPH"},
        {"type": "input_image", "image_url": left_original, "detail": "high"},
        {"type": "input_text", "text": "LEFT SHOE — LOCATION GUIDE ONLY"},
        {"type": "input_image", "image_url": left_grid, "detail": "high"},
        {"type": "input_text", "text": "RIGHT SHOE — UNTOUCHED FULL PHOTOGRAPH"},
        {"type": "input_image", "image_url": right_original, "detail": "high"},
        {"type": "input_text", "text": "RIGHT SHOE — LOCATION GUIDE ONLY"},
        {"type": "input_image", "image_url": right_grid, "detail": "high"},
    ]
    return request_assessment(images)


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


def overlay_heatmap(img: np.ndarray, box, grid_values, zones, side):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    coarse = np.asarray(grid_values, dtype=np.float32).reshape(GRID_ROWS, GRID_COLS) / 3.0
    heel_score = max(zones["heel_lateral"], zones["heel_central"], zones["heel_medial"])
    if heel_score > 0:
        # The required photo position fixes heel at the bottom. Reinforce the
        # anatomical outer heel when the zonal verdict and dense map disagree.
        lateral_cols = range(0, 4) if side == "left" else range(GRID_COLS - 4, GRID_COLS)
        target = float(heel_score) / 3.0
        for row in range(GRID_ROWS - 4, GRID_ROWS - 1):
            for col in lateral_cols:
                distance = abs(row - (GRID_ROWS - 3)) / 2.2 + abs(col - (1.5 if side == "left" else GRID_COLS - 2.5)) / 2.8
                if distance < 1.0:
                    coarse[row, col] = max(coarse[row, col], target * (1.0 - distance * 0.45))
    cell_w, cell_h = box_w / GRID_COLS, box_h / GRID_ROWS

    # Render each positive cell as a soft evidence centre. Adjacent centres merge
    # naturally into irregular wear zones instead of exposing the analysis grid.
    mapped = np.zeros((box_h, box_w), np.float32)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            strength = float(coarse[row, col])
            if strength <= 0:
                continue
            centre = (round((col + 0.5) * cell_w), round((row + 0.5) * cell_h))
            axes = (max(3, round(cell_w * 0.85)), max(3, round(cell_h * 0.85)))
            cv2.ellipse(mapped, centre, axes, 0, 0, 360, strength, -1, cv2.LINE_AA)
    mapped = cv2.GaussianBlur(
        mapped, (0, 0),
        sigmaX=max(3, cell_w * 0.72),
        sigmaY=max(3, cell_h * 0.72),
    )

    # Visible ribs/grooves mean tread is still present. Attenuate model evidence
    # over locally edge-rich rubber while retaining it over smooth polished areas.
    crop_gray = cv2.cvtColor(img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)], cv2.COLOR_BGR2GRAY)
    if crop_gray.shape == mapped.shape:
        gx = cv2.Sobel(crop_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_density = cv2.GaussianBlur(
            cv2.magnitude(gx, gy), (0, 0),
            sigmaX=max(2, cell_w * 0.38),
            sigmaY=max(2, cell_h * 0.38),
        )
        low, high = np.percentile(edge_density, (25, 78))
        if high > low + 0.01:
            smoothness = 1.0 - np.clip((edge_density - low) / (high - low), 0, 1)
            mapped *= 0.06 + 0.94 * np.power(smoothness, 1.55)
    heat = np.zeros((h, w), np.float32)
    heat[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = np.clip(mapped, 0, 1)[:min(h, y2)-max(0, y1), :min(w, x2)-max(0, x1)]
    heat *= sole_mask(img, box).astype(np.float32) / 255.0
    colour = np.zeros_like(img)
    colour[:, :] = (24, 24, 238)
    # Continuous opacity avoids rectangular threshold edges. Only negligible haze
    # is removed; credible evidence remains clearly visible.
    alpha = np.clip(heat * 1.58, 0, 0.76)
    alpha[alpha < 0.035] = 0
    alpha = alpha[..., None]
    composed = (img.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha).astype(np.uint8)
    output = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(output)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"ok": True, "marker": "GAVIOTA5-REFERENCE-V15"}


@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    try:
        left_jpeg, left_img = prepare_image(await left.read())
        right_jpeg, right_img = prepare_image(await right.read())
        left_box, left_method, left_detection = locate_sole(left_img)
        right_box, right_method, right_detection = locate_sole(right_img)
        assessment = assess_zones(
            image_data_url(left_jpeg),
            analysis_crop_data_url(left_img, left_box, True),
            image_data_url(right_jpeg),
            analysis_crop_data_url(right_img, right_box, True),
        )
        if not assessment["usable"]:
            return JSONResponse({
                "detail": "These photographs do not show enough reliable tread detail for an assessment.",
                "assessment": assessment,
            }, status_code=422)
        return {
            "left_heatmap_data_url": overlay_heatmap(left_img, left_box, assessment["left_grid"], assessment["left"], "left"),
            "right_heatmap_data_url": overlay_heatmap(right_img, right_box, assessment["right_grid"], assessment["right"], "right"),
            "assessment": assessment,
            "quality": {
                "left": {"crop_method": left_method, "detection_confidence": left_detection},
                "right": {"crop_method": right_method, "detection_confidence": right_detection},
            },
        }
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
