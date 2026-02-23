from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import base64

# ------------------------
# CREATE APP FIRST (CRITICAL)
# ------------------------
app = FastAPI(title="CheckMyRun API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# BASIC ROUTES
# ------------------------

@app.get("/")
def home():
    return {
        "ok": True,
        "service": "checkmyrun-api",
        "message": "API is running. Use /health or POST images to /api/analyse",
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------
# HELPER
# ------------------------

def to_data_url(file_bytes: bytes, filename: str):
    if not file_bytes:
        return None
    mime = "image/jpeg"
    if filename.lower().endswith(".png"):
        mime = "image/png"
    encoded = base64.b64encode(file_bytes).decode()
    return f"data:{mime};base64,{encoded}"


# ------------------------
# ANALYSIS ENDPOINT (SAFE MODE)
# ------------------------

@app.post("/api/analyse")
async def analyse(
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    rear: Optional[UploadFile] = File(None),
):
    try:
        left_bytes = await left.read() if left else None
        right_bytes = await right.read() if right else None
        rear_bytes = await rear.read() if rear else None

        if not left_bytes and not right_bytes:
            return {"ok": False, "analysis_text": "No images uploaded", "confidence": 0}

        return JSONResponse({
            "ok": True,
            "analysis_text": "Server recovered successfully. AI analysis temporarily disabled.",
            "confidence": 0.25,
            "left_overlay_data_url": to_data_url(left_bytes, left.filename) if left else None,
            "right_overlay_data_url": to_data_url(right_bytes, right.filename) if right else None,
            "notes": "Recovery mode"
        })

    except Exception as e:
        return JSONResponse({
            "ok": False,
            "analysis_text": f"Server error: {str(e)}",
            "confidence": 0
        })
