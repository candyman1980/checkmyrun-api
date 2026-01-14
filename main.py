from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://checkmyrun.com", "https://www.checkmyrun.com", "http://checkmyrun.com", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api"}

def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def wear_bias_from_image(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Make the sole white
    if np.mean(th) > 127:
        th = 255 - th

    kernel = np.ones((9, 9), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"bias": "unknown", "confidence": 0.0}

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        return {"bias": "unknown", "confidence": 0.0}

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tex = np.abs(lap)

    tex_masked = tex[mask == 255]
    p95 = np.percentile(tex_masked, 95) if tex_masked.size else 1.0
    tex_norm = np.clip(tex / (p95 + 1e-6), 0, 1)

    wear_map = 1.0 - tex_norm  # higher = smoother = more worn (proxy)

    H, W = wear_map.shape[:2]
    left_half = wear_map[:, :W//2]
    right_half = wear_map[:, W//2:]
    left_mask = mask[:, :W//2]
    right_mask = mask[:, W//2:]

    def mean_wear(half, half_mask):
        vals = half[half_mask == 255]
        return float(np.mean(vals)) if vals.size else 0.0

    wear_L = mean_wear(left_half, left_mask)
    wear_R = mean_wear(right_half, right_mask)

    ratio = (wear_L + 1e-6) / (wear_R + 1e-6)

    if ratio > 1.12:
        bias = "left-side heavier wear"
    elif ratio < 0.88:
        bias = "right-side heavier wear"
    else:
        bias = "balanced wear"

    confidence = float(min(abs(np.log(ratio)) / 0.5, 1.0))
    return {"bias": bias, "confidence": round(confidence, 2), "details": {"wear_left": round(wear_L, 3), "wear_right": round(wear_R, 3), "ratio": round(ratio, 3)}}

@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    left_img = read_image(left)
    right_img = read_image(right)

    left_res = wear_bias_from_image(left_img)
    right_res = wear_bias_from_image(right_img)

    return {
        "left": left_res,
        "right": right_res,
        "note": "Prototype wear-pattern proxy. For best results: bright light, camera straight above, minimal shadows."
    }
