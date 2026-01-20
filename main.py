import os
import json
import base64
import requests
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEBUG_UPLOADS = os.environ.get("DEBUG_UPLOADS", "").strip().lower() in ("1", "true", "yes", "on")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://checkmyrun.com", "https://www.checkmyrun.com", "http://checkmyrun.com", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V3-HEATMAP", "model": MODEL}


def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\n".join(out).strip()


def read_upload_bytes(upload: UploadFile) -> bytes:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")
    if DEBUG_UPLOADS:
        try:
            print(f"DEBUG_UPLOADS: filename={upload.filename} size={len(b)} bytes")
        except Exception:
            pass
    return b


def to_data_url_from_bytes(b: bytes, filename: str | None) -> str:
    name = (filename or "").lower()
    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def decode_image_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def sole_polygon_from_image(img_bgr: np.ndarray) -> tuple[list[dict] | None, list[str]]:
    """
    Returns (polygon_points_normalised, issues)
    polygon points are [{x:0-1,y:0-1}, ...]
    """
    issues: list[str] = []

    h0, w0 = img_bgr.shape[:2]
    if h0 < 200 or w0 < 200:
        return None, ["image too small"]

    # Resize for stability/speed
    target = 900
    scale = target / max(h0, w0)
    if scale < 1.0:
        img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)))
    else:
        img = img_bgr.copy()

    h, w = img.shape[:2]

    # Slight blur reduces texture noise / floor patterns
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # GrabCut initial rectangle (central region where sole usually is)
    rect = (
        int(0.10 * w),
        int(0.08 * h),
        int(0.80 * w),
        int(0.84 * h),
    )

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_blur, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None, ["segmentation failed"]

    # Foreground mask
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")

    # Clean up mask
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    # Keep largest connected component (usually the sole)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return None, ["could not isolate sole"]

    # stats: [label, x, y, w, h, area]
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])

    area_ratio = best_area / float(w * h)
    if area_ratio < 0.08:
        return None, ["sole region too small (background too dominant)"]
    if area_ratio > 0.75:
        return None, ["sole region too large (background likely included)"]

    sole = (labels == best_idx).astype("uint8") * 255

    # Find contour and hull
    contours, _ = cv2.findContours(sole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, ["could not find sole outline"]

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    ar = bw / float(bh + 1e-6)

    # Shoe-like aspect ratio guard
    if ar > 0.95 or ar < 0.18:
        return None, ["sole outline aspect ratio looks wrong (try a closer/straighter photo)"]

    hull = cv2.convexHull(cnt)

    # Simplify polygon
    peri = cv2.arcLength(hull, True)
    eps = 0.01 * peri
    approx = cv2.approxPolyDP(hull, eps, True)

    if len(approx) < 6:
        # fall back to hull points
        approx = hull

    pts = []
    for p in approx.reshape(-1, 2):
        px = float(p[0]) / float(w)
        py = float(p[1]) / float(h)
        pts.append({"x": max(0.0, min(1.0, px)), "y": max(0.0, min(1.0, py))})

    # Extra sanity: bbox coverage of polygon
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if box_area < 0.10:
        return None, ["sole outline too small/uncertain"]

    return pts, issues


# Response schema (includes heatmap + outsole polygon)
response_schema = {
    "name": "checkmyrun_pronation_heatmap",
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
                    "notes": {"type": "string"},
                },
                "required": ["pronation", "notes"],
            },
            "right": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "notes": {"type": "string"},
                },
                "required": ["pronation", "notes"],
            },
            "overall": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pronation": {"type": "string", "enum": ["overpronation", "underpronation", "neutral", "unclear"]},
                    "shoe_category": {"type": "string", "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]},
                },
                "required": ["pronation", "shoe_category"],
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
                    "heatmap": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "radius": {"type": "number", "minimum": 0.02, "maximum": 0.35},
                                "intensity": {"type": "number", "minimum": 0.1, "maximum": 1},
                            },
                            "required": ["x", "y", "radius", "intensity"],
                        },
                    }
                },
                "required": ["heatmap"],
            },
            "right_visual": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "heatmap": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "radius": {"type": "number", "minimum": 0.02, "maximum": 0.35},
                                "intensity": {"type": "number", "minimum": 0.1, "maximum": 1},
                            },
                            "required": ["x", "y", "radius", "intensity"],
                        },
                    }
                },
                "required": ["heatmap"],
            },
        },
        "required": ["left", "right", "overall", "photo_quality", "left_visual", "right_visual"],
    },
}


def build_instruction() -> str:
    return (
        "You are a running shoe fitting assistant.\n"
        "Inputs:\n"
        "- LEFT outsole photo\n"
        "- RIGHT outsole photo\n"
        "- Rear heel photo showing BOTH shoes\n\n"
        "Task:\n"
        "1) Identify visible wear hotspots (heel: lateral/central/medial; forefoot: lateral/central/medial; toe-off).\n"
        "2) Infer gait category: overpronation / underpronation / neutral / unclear.\n"
        "3) Recommend shoe category: stability / neutral / cushioned-neutral / unclear.\n"
        "4) Provide a simple wear heatmap for LEFT and RIGHT as points (x,y,radius,intensity) where wear appears strongest.\n\n"
        "Rules:\n"
        "- Be CONSERVATIVE. If evidence is weak or mixed, prefer 'neutral' or 'unclear' rather than over-calling.\n"
        "- Base decisions ONLY on what is visible.\n"
        "- Use the rear heel photo to assess inward collapse/heel alignment and left-right asymmetry.\n"
        "- If photos are blurry, shadowed, too new/dirty, poor angle, or rear heel unclear: set overall to 'unclear' and list issues.\n"
        "- Keep notes concise and specific.\n"
        "- Informational only; no medical claims.\n"
        "Return ONLY valid JSON matching the schema.\n"
    )


def call_openai(left_url: str, right_url: str, rear_url: str) -> dict:
    instruction = build_instruction()

    payload = {
        "model": MODEL,
        "temperature": 0,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": "LEFT OUTSOLE:"},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_text", "text": "RIGHT OUTSOLE:"},
                    {"type": "input_image", "image_url": right_url},
                    {"type": "input_text", "text": "REAR HEEL (BOTH SHOES):"},
                    {"type": "input_image", "image_url": rear_url},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": response_schema["name"],
                "schema": response_schema["schema"],
                "strict": True,
            }
        },
        "max_output_tokens": 700,
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {r.text}")

    resp_json = r.json()
    text = extract_output_text(resp_json)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Model returned non-JSON unexpectedly: {text[:400]}")


def postprocess_visuals(result: dict, left_img: np.ndarray, right_img: np.ndarray) -> dict:
    # Compute outsole polygons locally (this is the key fix for background bleed)
    left_poly, left_issues = sole_polygon_from_image(left_img)
    right_poly, right_issues = sole_polygon_from_image(right_img)

    issues = list(result.get("photo_quality", {}).get("issues", []))

    # Attach polygons if available; if not, suppress heatmap by emptying it (avoid misleading background heat)
    if left_poly is None:
        issues.extend([f"left outsole outline uncertain: {msg}" for msg in left_issues] or ["left outsole outline uncertain"])
        result["left_visual"]["heatmap"] = []
        result["left_visual"]["outsole_polygon"] = None
    else:
        result["left_visual"]["outsole_polygon"] = left_poly

    if right_poly is None:
        issues.extend([f"right outsole outline uncertain: {msg}" for msg in right_issues] or ["right outsole outline uncertain"])
        result["right_visual"]["heatmap"] = []
        result["right_visual"]["outsole_polygon"] = None
    else:
        result["right_visual"]["outsole_polygon"] = right_poly

    # Normalise photo_quality
    result.setdefault("photo_quality", {"ok": True, "issues": []})
    # Mark not-ok if we had segmentation problems
    if left_poly is None or right_poly is None:
        result["photo_quality"]["ok"] = False

    # De-dup issues, keep order
    seen = set()
    deduped = []
    for it in issues:
        if it and it not in seen:
            seen.add(it)
            deduped.append(it)
    result["photo_quality"]["issues"] = deduped

    # Optional: if quality is not ok, be conservative overall
    if not result["photo_quality"]["ok"]:
        # If the model overcalled, soften to unclear
        if result.get("overall", {}).get("pronation") in ("overpronation", "underpronation"):
            result["overall"]["pronation"] = "unclear"
            result["overall"]["shoe_category"] = "unclear"

    return result


async def _analyse_impl(left: UploadFile, right: UploadFile, rear: UploadFile):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_b = read_upload_bytes(left)
        right_b = read_upload_bytes(right)
        rear_b = read_upload_bytes(rear)

        left_img = decode_image_bytes(left_b)
        right_img = decode_image_bytes(right_b)

        left_url = to_data_url_from_bytes(left_b, left.filename)
        right_url = to_data_url_from_bytes(right_b, right.filename)
        rear_url = to_data_url_from_bytes(rear_b, rear.filename)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    result = call_openai(left_url, right_url, rear_url)
    result = postprocess_visuals(result, left_img, right_img)
    return result


# UK spelling endpoint (your frontend uses this)
@app.post("/analyse")
async def analyse(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await _analyse_impl(left, right, rear)


# US spelling alias (won’t hurt to keep)
@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...), rear: UploadFile = File(...)):
    return await _analyse_impl(left, right, rear)
