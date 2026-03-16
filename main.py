import os
import io
import cv2
import base64
import numpy as np
import httpx
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

cv2.setNumThreads(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load YOLO once
# -----------------------------

model = YOLO("yolov8n.pt")

# -----------------------------
# Simple UI
# -----------------------------

INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CheckMyRun</title>
<style>
body{font-family:system-ui;margin:30px;background:#fafafa}
.card{max-width:1100px;margin:auto;background:white;padding:20px;border-radius:12px;border:1px solid #ddd}
.upload{border:2px dashed #ccc;padding:30px;text-align:center;border-radius:10px;margin-bottom:20px;cursor:pointer}
.upload img{max-width:100%;margin-top:10px}
.result img{max-width:100%;border:1px solid #ddd;margin-top:10px}
button{padding:10px 18px;font-weight:bold}
</style>
</head>
<body>

<div class="card">

<h1>CheckMyRun</h1>

<form id="form">

<div class="upload">
<input type="file" name="left" id="left" accept="image/*" required>
<p>Upload LEFT sole</p>
<img id="leftPreview">
</div>

<div class="upload">
<input type="file" name="right" id="right" accept="image/*" required>
<p>Upload RIGHT sole</p>
<img id="rightPreview">
</div>

<button>Analyse</button>

</form>

<div id="result"></div>

</div>

<script>

function preview(input,img){

input.onchange=()=>{
const f=input.files[0]
if(!f)return
img.src=URL.createObjectURL(f)
}

}

preview(
document.getElementById("left"),
document.getElementById("leftPreview")
)

preview(
document.getElementById("right"),
document.getElementById("rightPreview")
)

document.getElementById("form").onsubmit=async(e)=>{

e.preventDefault()

const fd=new FormData(e.target)

const r=await fetch("/analyze",{method:"POST",body:fd})

const j=await r.json()

document.getElementById("result").innerHTML=`
<h2>Analysis</h2>
<p>${j.analysis}</p>

<h3>Left heatmap</h3>
<img src="${j.left_heatmap}">

<h3>Right heatmap</h3>
<img src="${j.right_heatmap}">
`
}

</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML

# -----------------------------
# Image utilities
# -----------------------------

def decode(bytes):

    arr=np.frombuffer(bytes,np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)

    return img

def encode(img):

    _,buf=cv2.imencode(".png",img)

    return buf.tobytes()

# -----------------------------
# YOLO sole detection
# -----------------------------

def crop_sole(img):

    res=model.predict(img,conf=0.25,verbose=False)[0]

    if res.boxes is None:
        raise Exception("No sole detected")

    boxes=res.boxes.xyxy.cpu().numpy()

    best=None
    best_area=0

    for b in boxes:

        x1,y1,x2,y2=b

        area=(x2-x1)*(y2-y1)

        if area>best_area:
            best=b
            best_area=area

    x1,y1,x2,y2=map(int,best)

    return img[y1:y2,x1:x2]

# -----------------------------
# Sole mask (OpenCV segmentation)
# -----------------------------

def mask_sole(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,th=cv2.threshold(gray,220,255,cv2.THRESH_BINARY_INV)

    kernel=np.ones((7,7),np.uint8)

    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel)

    cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return np.ones(gray.shape,np.uint8)

    largest=max(cnts,key=cv2.contourArea)

    mask=np.zeros(gray.shape,np.uint8)

    cv2.drawContours(mask,[largest],-1,255,-1)

    return mask

# -----------------------------
# Wear heatmap
# -----------------------------

def heatmap(img):

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    s=hsv[:,:,1]

    blur=cv2.GaussianBlur(s,(0,0),15)

    norm=cv2.normalize(blur,None,0,255,cv2.NORM_MINMAX)

    heat=cv2.applyColorMap(norm,cv2.COLORMAP_JET)

    overlay=cv2.addWeighted(img,0.6,heat,0.4,0)

    return overlay

# -----------------------------
# OpenAI analysis
# -----------------------------

def analyze_ai(left,right):

    l="data:image/png;base64,"+base64.b64encode(left).decode()
    r="data:image/png;base64,"+base64.b64encode(right).decode()

    payload={
    "model":MODEL,
    "input":[{
        "role":"user",
        "content":[
        {"type":"input_text","text":"Analyse running shoe wear and pronation."},
        {"type":"input_image","image_url":l},
        {"type":"input_image","image_url":r}
        ]
    }]
    }

    resp=httpx.post(
    "https://api.openai.com/v1/responses",
    headers={
    "Authorization":f"Bearer {OPENAI_API_KEY}"
    },
    json=payload,
    timeout=60
    )

    data=resp.json()

    return data["output_text"]

# -----------------------------
# API endpoint
# -----------------------------

@app.post("/analyze")
async def analyze(left:UploadFile=File(...),right:UploadFile=File(...)):

    left_bytes=await left.read()
    right_bytes=await right.read()

    left_img=decode(left_bytes)
    right_img=decode(right_bytes)

    left_crop=crop_sole(left_img)
    right_crop=crop_sole(right_img)

    left_mask=mask_sole(left_crop)
    right_mask=mask_sole(right_crop)

    left_heat=heatmap(left_crop)
    right_heat=heatmap(right_crop)

    left_heat_bytes=encode(left_heat)
    right_heat_bytes=encode(right_heat)

    analysis=analyze_ai(left_heat_bytes,right_heat_bytes)

    return {
    "analysis":analysis,
    "left_heatmap":"data:image/png;base64,"+base64.b64encode(left_heat_bytes).decode(),
    "right_heatmap":"data:image/png;base64,"+base64.b64encode(right_heat_bytes).decode()
    }
