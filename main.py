import os
import json
import base64
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
import requests

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------
# Config
# ---------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEBUG_UPLOADS = os.environ.get("DEBUG_UPLOADS", "0").strip() in ("1", "true", "True", "yes", "YES")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("checkmyrun")

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


# ---------------------------
# Helpers: uploads
# ---------------------------
def read_upload_bytes(upload: UploadFile) -> bytes:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")
    return b


def mime_from_filename(name: str) -> str:
    name = (name or "").lower()
    if name.endswith(".png"):
        return "image/png"
    return "image/jpeg"


def bytes_to_data_url(b: bytes, mime: str) -> str:
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def img_bytes_to_bgr(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


# ---------------------------
# Level 2: outsole segmentation + wear heatmap
# ---------------------------
def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[0]


def segment_outsole_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Heuristic segmentation (no ML):
    - Use GrabCut initialized around center
    - Then refine with morphology
    - Keep largest connected region
    Returns uint8 mask {0,255}.
    """
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()

    # Downscale for stability/speed
    scale = 900.0 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    hh, ww = img.shape[:2]

    # GrabCut rectangle roughly central (shoe likely central even if held)
    rect_margin_x = int(ww * 0.08)
    rect_margin_y = int(hh * 0.08)
    rect = (rect_margin_x, rect_margin_y, ww - 2 * rect_margin_x, hh - 2 * rect_margin_y)

    mask = np.zeros((hh, ww), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        # 4-5 iterations usually enough
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # If GrabCut fails, fallback to “everything”
        fallback = np.full((hh, ww), 255, np.uint8)
        return cv2.resize(fallback, (w, h), interpolation=cv2.INTER_NEAREST)

    # Convert GrabCut mask to binary foreground
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Cleanup
    k = max(5, int(min(hh, ww) * 0.01) // 2 * 2 + 1)  # odd kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest region
    cnt = largest_contour(fg)
    if cnt is None or cv2.contourArea(cnt) < (hh * ww * 0.05):
        # Too small => segmentation likely failed; return blank so UI draws nothing
        blank = np.zeros((hh, ww), np.uint8)
        return cv2.resize(blank, (w, h), interpolation=cv2.INTER_NEAREST)

    clean = np.zeros_like(fg)
    cv2.drawContours(clean, [cnt], -1, 255, thickness=cv2.FILLED)

    # Resize back to original
    clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_NEAREST)
    return clean


def crop_to_mask(img_bgr: np.ndarray, mask: np.ndarray, pad: float = 0.08) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crops img+mask to the bounding box of mask, with padding.
    Also turns background to white (so OpenAI stops reading the floor).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_bgr, mask

    h, w = img_bgr.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    px = int((x1 - x0 + 1) * pad)
    py = int((y1 - y0 + 1) * pad)

    x0 = max(0, x0 - px)
    x1 = min(w - 1, x1 + px)
    y0 = max(0, y0 - py)
    y1 = min(h - 1, y1 + py)

    img_c = img_bgr[y0:y1 + 1, x0:x1 + 1].copy()
    mask_c = mask[y0:y1 + 1, x0:x1 + 1].copy()

    # White background
    bg = np.full_like(img_c, 255)
    m3 = cv2.cvtColor(mask_c, cv2.COLOR_GRAY2BGR)
    img_c = np.where(m3 > 0, img_c, bg)

    return img_c, mask_c


def contour_polygon_norm(mask: np.ndarray, max_points: int = 18) -> List[Dict[str, float]]:
    """
    Approximates outsole contour as a polygon, normalized to 0..1 coords.
    """
    cnt = largest_contour(mask)
    if cnt is None:
        return []

    peri = cv2.arcLength(cnt, True)
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)

    pts = approx.reshape(-1, 2).astype(float)
    h, w = mask.shape[:2]

    # Downsample if too many points
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]

    poly = []
    for x, y in pts:
        poly.append({"x": float(x / max(1, w - 1)), "y": float(y / max(1, h - 1))})
    return poly


def wear_score_map(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Wear proxy: worn areas tend to be smoother / less textured => lower edge energy.
    We compute edge energy and invert it inside mask.
    Returns float map 0..1.
    """
    # Grayscale + contrast normalisation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur a bit to suppress noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge energy (Laplacian magnitude)
    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)
    edge = np.abs(lap)

    # Normalise edge within mask
    m = (mask > 0)
    if not np.any(m):
        return np.zeros_like(gray, dtype=np.float32)

    e = edge[m]
    # Robust scaling (percentiles)
    lo = np.percentile(e, 10)
    hi = np.percentile(e, 95)
    denom = max(1e-6, hi - lo)
    edge_n = (edge - lo) / denom
    edge_n = np.clip(edge_n, 0.0, 1.0)

    # Wear score = inverse of edge energy
    wear = 1.0 - edge_n

    # Also encourage “smooth + bright” (often scuffed rubber shows brighter)
    g = gray.astype(np.float32) / 255.0
    wear = (0.75 * wear + 0.25 * g)

    # Mask outside
    wear = wear * m.astype(np.float32)

    # Smooth wear map so it makes nice blobs
    wear = cv2.GaussianBlur(wear, (0, 0), sigmaX=6, sigmaY=6)
    wear = np.clip(wear, 0.0, 1.0)

    return wear.astype(np.float32)


def heatmap_points_from_wear(wear: np.ndarray, mask: np.ndarray, n: int = 6) -> List[Dict[str, float]]:
    """
    Picks hotspots from wear map and returns a list of points:
    {x,y,radius,intensity} normalized to 0..1
    """
    h, w = wear.shape[:2]
    m = (mask > 0)
    if not np.any(m):
        return []

    # Focus on top wear values inside mask
    wear_in = wear.copy()
    wear_in[~m] = 0.0

    # Threshold to candidates
    thr = float(np.percentile(wear_in[m], 85))
    cand = (wear_in >= thr).astype(np.uint8) * 255

    # If too sparse, lower threshold
    if cand.sum() < 500:
        thr = float(np.percentile(wear_in[m], 75))
        cand = (wear_in >= thr).astype(np.uint8) * 255

    # Find connected hotspots
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[: max(1, n * 2)]

    pts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 120:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # intensity from wear value at centroid
        ix = int(np.clip(round(cx), 0, w - 1))
        iy = int(np.clip(round(cy), 0, h - 1))
        intensity = float(np.clip(wear_in[iy, ix], 0.2, 1.0))

        # radius from hotspot size
        radius = float(np.clip(np.sqrt(area / np.pi) / max(1.0, w), 0.06, 0.18))

        pts.append({
            "x": float(cx / max(1, w - 1)),
            "y": float(cy / max(1, h - 1)),
            "radius": radius,
            "intensity": intensity,
        })

        if len(pts) >= n:
            break

    # If we somehow got none, place a gentle center point (but low intensity)
    if not pts:
        pts = [{"x": 0.5, "y": 0.6, "radius": 0.12, "intensity": 0.25}]

    return pts


def bgr_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode jpeg")
    return buf.tobytes()


# ---------------------------
# OpenAI response parsing
# ---------------------------
def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(out).strip()


# ---------------------------
# API
# ---------------------------
@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V3-HEATMAP", "model": MODEL}


# Strict JSON schema response (includes visuals)
RESPONSE_SCHEMA = {
    "name": "checkmyrun_pronation_v3",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "left": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "notes": {"type": "string"},
                },
                "required": ["pronation", "confidence", "notes"],
            },
            "right": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "notes": {"type": "string"},
                },
                "required": ["pronation", "confidence", "notes"],
            },
            "overall": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "shoe_category": {"type": "string", "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]},
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
            },
            "left_visual": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outsole_polygon": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "heatmap": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "radius": {"type": "number", "minimum": 0, "maximum": 1},
                                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y", "radius", "intensity"],
                        },
                    },
                },
                "required": ["outsole_polygon", "heatmap"],
            },
            "right_visual": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outsole_polygon": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "heatmap": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "radius": {"type": "number", "minimum": 0, "maximum": 1},
                                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y", "radius", "intensity"],
                        },
                    },
                },
                "required": ["outsole_polygon", "heatmap"],
            },
        },
        "required": ["left", "right", "overall", "photo_quality", "left_visual", "right_visual"],
    },
}


def build_instruction() -> str:
    return (
        "You are a running shoe fitting assistant.\n"
        "You will be given two OUTSOLE PHOTOS (LEFT and RIGHT) that have been cropped to the outsole only "
        "(background removed/whitened), plus a rear heel photo showing both shoes.\n\n"
        "Task:\n"
        "1) Identify visible wear hotspots (heel: lateral/central/medial; forefoot: lateral/central/medial; toe-off).\n"
        "2) Infer likely gait category from wear patterns and heel alignment: overpronation / underpronation / neutral / unclear.\n"
        "3) Recommend shoe category: stability / neutral / cushioned-neutral / unclear.\n\n"
        "Rules:\n"
        "- Base your decision ONLY on what is visible in the photos.\n"
        "- Use the rear heel photo to assess heel alignment/inward collapse indicators and left-right asymmetry.\n"
        "- If photos are too new/dirty/blurry/heavily shadowed/angled, output 'unclear' and list issues.\n"
        "- If left and right differ, reflect that and set overall to the more supportive recommendation.\n"
        "- Keep notes concise and specific.\n"
        "- Informational only; no medical claims.\n"
        "Return ONLY valid JSON matching the schema.\n"
    )


def call_openai(left_data_url: str, right_data_url: str, rear_data_url: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_instruction()},
                    {"type": "input_text", "text": "LEFT OUTSOLE (CROPPED):"},
                    {"type": "input_image", "image_url": left_data_url},
                    {"type": "input_text", "text": "RIGHT OUTSOLE (CROPPED):"},
                    {"type": "input_image", "image_url": right_data_url},
                    {"type": "input_text", "text": "REAR HEEL VIEW (BOTH SHOES):"},
                    {"type": "input_image", "image_url": rear_data_url},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": RESPONSE_SCHEMA["name"],
                "schema": RESPONSE_SCHEMA["schema"],
                "strict": True,
            }
        },
        "max_output_tokens": 650,
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
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Model returned non-JSON unexpectedly: {text[:4000]}")


def compute_visuals(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Returns:
      cropped_img_bgr, cropped_mask, outsole_polygon(norm), heatmap_points(norm)
    """
    mask = segment_outsole_mask(img_bgr)
    cropped_img, cropped_mask = crop_to_mask(img_bgr, mask, pad=0.08)

    # If segmentation failed => no polygon/heat
    if np.count_nonzero(cropped_mask) < 500:
        return cropped_img, cropped_mask, [], []

    poly = contour_polygon_norm(cropped_mask, max_points=18)

    wear = wear_score_map(cropped_img, cropped_mask)
    heat = heatmap_points_from_wear(wear, cropped_mask, n=6)

    return cropped_img, cropped_mask, poly, heat


@app.post("/analyse")
async def analyse(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    rear: UploadFile = File(...),
):
    # Read bytes (so we can log sizes)
    try:
        left_b = read_upload_bytes(left)
        right_b = read_upload_bytes(right)
        rear_b = read_upload_bytes(rear)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    if DEBUG_UPLOADS:
        log.info(f"DEBUG_UPLOADS: LEFT filename={left.filename} size={len(left_b)} bytes")
        log.info(f"DEBUG_UPLOADS: RIGHT filename={right.filename} size={len(right_b)} bytes")
        log.info(f"DEBUG_UPLOADS: REAR filename={rear.filename} size={len(rear_b)} bytes")

    # Decode images
    try:
        left_img = img_bytes_to_bgr(left_b)
        right_img = img_bytes_to_bgr(right_b)
        rear_img = img_bytes_to_bgr(rear_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    # Compute CV visuals on outsole shots
    left_crop, left_mask, left_poly, left_heat = compute_visuals(left_img)
    right_crop, right_mask, right_poly, right_heat = compute_visuals(right_img)

    # Convert cropped outsole images to data URLs (background removed => model stops reading the floor)
    left_crop_bytes = bgr_to_jpeg_bytes(left_crop, quality=88)
    right_crop_bytes = bgr_to_jpeg_bytes(right_crop, quality=88)
    rear_mime = mime_from_filename(rear.filename or "")
    rear_url = bytes_to_data_url(rear_b, rear_mime)

    left_url = bytes_to_data_url(left_crop_bytes, "image/jpeg")
    right_url = bytes_to_data_url(right_crop_bytes, "image/jpeg")

    # Call OpenAI for pronation + notes + quality
    result = call_openai(left_url, right_url, rear_url)

    # Inject visuals computed from pixels (NOT from the model)
    result["left_visual"] = {"outsole_polygon": left_poly, "heatmap": left_heat}
    result["right_visual"] = {"outsole_polygon": right_poly, "heatmap": right_heat}

    # If segmentation failed, add a photo-quality issue so the UI can explain why heatmap may be empty
    issues = result.get("photo_quality", {}).get("issues", [])
    if not left_poly:
        issues.append("Left outsole could not be isolated clearly (background/angle/hand obstruction).")
    if not right_poly:
        issues.append("Right outsole could not be isolated clearly (background/angle/hand obstruction).")

    # de-dup issues
    dedup = []
    for it in issues:
        if it not in dedup:
            dedup.append(it)
    if "photo_quality" in result and isinstance(result["photo_quality"], dict):
        result["photo_quality"]["issues"] = dedup
        # ok is only true if there are no issues
        result["photo_quality"]["ok"] = (len(dedup) == 0)

    return result


# Alias for older clients
@app.post("/analyze")
async def analyze(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    rear: UploadFile = File(...),
):
    return await analyse(left=left, right=right, rear=rear)
