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
    "required": ["usable", "confidence", "left", "right", "left_regions", "right_regions", "comparison", "limitations"],
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
        "left_regions": {
            "type": "array", "maxItems": 14,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["intensity", "points"],
                "properties": {
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "points": {
                        "type": "array", "minItems": 3, "maxItems": 14,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["x", "y"],
                            "properties": {
                                "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                                "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                            },
                        },
                    },
                },
            },
        },
        "right_regions": {
            "type": "array", "maxItems": 14,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["intensity", "points"],
                "properties": {
                    "intensity": {"type": "integer", "minimum": 1, "maximum": 3},
                    "points": {
                        "type": "array", "minItems": 3, "maxItems": 14,
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["x", "y"],
                            "properties": {
                                "x": {"type": "integer", "minimum": 0, "maximum": 1000},
                                "y": {"type": "integer", "minimum": 0, "maximum": 1000},
                            },
                        },
                    },
                },
            },
        },
        "comparison": {"type": "string"},
        "limitations": {"type": "string"},
    },
}

ASSESSMENT_PROMPT = """Map visible Hoka Gaviota 5 outsole wear at high spatial precision. The first image is an UNWORN GAVIOTA 5 STRUCTURAL REFERENCE. Its colourway, lighting, scale and rotation are irrelevant. Use only its manufactured geometry: rubber-pad outlines, grooves, ribs, moulded lines, cut-outs and channels. After it, you receive each user's untouched full photograph followed by a tight sole crop with a light coordinate guide. Coordinates run 0..1000 within that crop: x from left to right and y from top to bottom. The rounded TOE is normally at the top and the HEEL nearest the hand at the bottom. Verify anatomy from pad geometry.

First geometrically align the reference outsole to each user sole using pad outlines, channels and cut-outs—not colour. Then compare corresponding manufactured details. The governing rule is CONTINUITY OF MANUFACTURED TEXTURE. Trace the reference's man-made lines, ribs, contours, stippling and fine mould texture through every corresponding rubber pad. Where expected reference detail becomes faint, interrupted or absent in the user photo without an intentional boundary, the smooth gap is wear and must be highlighted. Confirm with neighbouring texture and the matching shoe. Inspect the TOE PAD and HEEL PAD separately; these high-contact areas must not be skipped. Mark the full smooth interruption, not merely its boundary.

CRITICAL DISTINCTION: visible man-made lines and contours are the reference pattern, not wear. Preserve them, but highlight the adjacent smooth rubber wherever the pattern that should continue no longer exists. A few surviving large contours do not make the surrounding rubber unworn when finer lines have disappeared. Default an unexplained smooth gap inside a patterned rubber pad to wear, not factory-smooth. Call it factory-smooth only when a crisp intentional boundary or repeated identical examples prove that design.

POLYGON EXCLUSION RULE: every polygon must remain on the raised, ground-contacting face of an outsole pad. Never cross a pad outline. Never cover exposed midsole foam, recessed channels, flex grooves, cut-outs, holes, dirt-filled trenches, pad sidewalls or the central sculpted spine. These features are deliberately manufactured geometry, not worn contact surfaces. Before returning each polygon, verify that every vertex and the area between vertices lies on one continuous contact pad. If a suspected patch crosses a channel, split it into separate polygons and omit the channel. Ignore dirt, shadows, glare, colour changes and photographic blur.

Return irregular polygon regions that closely trace the visible boundaries of smooth worn rubber. Use enough points to follow each natural patch; never return rectangles, grid cells or an entire pad when only part is worn. Polygon coordinates are 0..1000 in the tight coordinate-guide crop. Intensity 1 means light texture loss, 2 clearly smooth/flattened rubber, and 3 pronounced polishing or material loss. Adjacent wear with the same intensity should be one organic region. Prefer sensitivity over omission once surrounding or reference detail proves what texture should continue.

MANDATORY HEEL CHECK: identify the lowest broad ground-contacting heel pad in each user crop and its matching unworn reference pad. Trace every area where the reference's fine parallel lines or mould texture has disappeared. If the zonal heel score is nonzero, at least one returned heel polygon must cover that same evidence on the bottom heel pad—not a midfoot pad or a channel above it. Repeat this check for the outer and central toe contact pads. Finally reject or trim any polygon that touches recessed foam or crosses a manufactured pad boundary. The nine zone scores summarize the same evidence: 0 none, 1 light, 2 moderate, 3 heavy.

Set usable=false and confidence below 35 if either sole is incomplete, strongly oblique, blurred, glared, or the original design cannot be inferred. Confidence measures photographic evidence only. Do not diagnose gait, pronation, supination, injury risk or a medical condition."""

GENERIC_ASSESSMENT_PROMPT = """Inspect the two running-shoe outsole photographs for visible rubber wear. This must work for any brand, model and colourway. Never reject a clear photograph merely because the shoe model is unfamiliar.

Wear means local loss of manufactured tread detail: ribs, grooves, stippling, mould texture or sharp lug edges become smoother, shallower, rounded or absent. Infer the intended pattern from repeated neighbouring elements, continuity across each rubber pad, and comparison between the left and right shoe. Existing crisp man-made lines are NOT wear. Do not confuse dirt, shadows, glare, colour, recessed channels, exposed midsole or deliberately smooth panels with wear.

Return organic polygon regions around only visually supported worn rubber. Coordinates are 0..1000 in each tight coordinate-guide crop: x from left to right and y from top/toe to bottom/heel. Every polygon must stay on one raised ground-contacting rubber pad. Never cross pad outlines or include background, hand, foam, channels, grooves, holes, trenches or pad sidewalls. Split disconnected worn areas into separate polygons. Intensity 1 means subtle smoothing, 2 clear loss of texture, and 3 severe flattening or material loss.

Inspect the entire outer and central toe pads and the entire outer and central heel pads twice before finishing; these high-contact areas are commonly missed. Prefer sensitivity once neighbouring detail proves that manufactured texture should continue, but do not paint intact patterned rubber. The nine zone scores summarize the same evidence from 0 none to 3 heavy.

Set usable=false and confidence below 35 only if either sole is incomplete, strongly oblique, badly blurred or obscured by glare. Confidence measures photographic evidence only. Do not diagnose gait, pronation, supination, injury risk or a medical condition."""


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
            x = round(cw * index / 10)
            y = round(ch * index / 10)
            cv2.line(crop, (x, 0), (x, ch), (255, 255, 255), 1)
            cv2.line(crop, (0, y), (cw, y), (255, 255, 255), 1)
            cv2.putText(crop, str(index * 100), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(crop, str(index * 100), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 2, cv2.LINE_AA)
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
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "shoe_wear_patches",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            }
        },
        "max_output_tokens": 12000,
    }
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json=payload,
        timeout=150,
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
        {"type": "input_text", "text": GENERIC_ASSESSMENT_PROMPT},
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


def overlay_heatmap(img: np.ndarray, box, regions):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    heat = np.zeros((h, w), np.float32)

    # Each model-drawn polygon becomes a feathered patch. The maximum operation
    # preserves severity where regions overlap without exposing artificial grids.
    for region in regions:
        points = region.get("points", [])
        if len(points) < 3:
            continue
        polygon = np.asarray([
            [x1 + round(float(point["x"]) * box_w / 1000),
             y1 + round(float(point["y"]) * box_h / 1000)]
            for point in points
        ], dtype=np.int32)
        region_mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(region_mask, [polygon], 1.0, cv2.LINE_AA)
        feather = max(3.0, min(box_w, box_h) * 0.009)
        region_mask = cv2.GaussianBlur(region_mask, (0, 0), feather)
        strength = {1: 0.42, 2: 0.70, 3: 1.0}.get(int(region.get("intensity", 1)), 0.42)
        heat = np.maximum(heat, region_mask * strength)

    heat *= sole_mask(img, box).astype(np.float32) / 255.0
    colour = np.zeros_like(img)
    colour[:, :] = (24, 24, 238)
    # Continuous opacity avoids rectangular threshold edges. Only negligible haze
    # is removed; credible evidence remains clearly visible.
    alpha = np.clip(heat * 0.82, 0, 0.78)
    alpha[alpha < 0.025] = 0
    alpha = alpha[..., None]
    composed = (img.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha).astype(np.uint8)
    output = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(output)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"ok": True, "marker": "GENERIC-CONTOURS-V17"}


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
            "left_heatmap_data_url": overlay_heatmap(left_img, left_box, assessment["left_regions"]),
            "right_heatmap_data_url": overlay_heatmap(right_img, right_box, assessment["right_regions"]),
            "assessment": assessment,
            "quality": {
                "left": {"crop_method": left_method, "detection_confidence": left_detection},
                "right": {"crop_method": right_method, "detection_confidence": right_detection},
            },
        }
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
