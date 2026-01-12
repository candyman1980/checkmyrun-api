from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"ok": True, "service": "checkmyrun-api"}

@app.post("/analyze")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    return {
        "wear_bias": "neutral",
        "shoe_type": "neutral trainer",
        "confidence": 0.55,
        "message": "Prototype result"
    }
