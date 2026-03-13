import cv2
cv2.setNumThreads(1)

import base64
import io
from functools import lru_cache

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from rembg import remove, new_session

cv2.setNumThreads(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# HTML UI
# ----------------------------------------------------

INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>CheckMyRun</title>
<style>
body{font-family:system-ui;margin:30px;background:#fafafa}
.card{max-width:1100px;margin:auto;background:#fff;padding:20px;border-radius:10px;border:1px solid #eee}
.preview{border:2px dashed #ccc;padding:20px;text-align:center;border-radius:8px;cursor:pointer}
img{max-width:100%;margin-top:10px}
</style>
</head>

<body>
<div class="card">

<h2>CheckMyRun</h2>

<form id="form">

<label class="preview">
<input type="file" name="left" id="left" hidden required>
Left sole
<img id="leftPreview">
</label>

<br>

<label class="preview">
<input type="file" name="right" id="right" hidden required>
Right sole
<img id="rightPreview">
</label>

<br><br>

<button>Analyse</button>

</form>

<div id="result"></div>

</div>

<script>

function preview(input,id){
const file=input.files[0]
if(!file)return
const url=URL.createObjectURL(file)
document.getElementById(id).src=url
}

left.onchange=()=>preview(left,"leftPreview")
right.onchange=()=>preview(right,"rightPreview")

form.onsubmit=async e=>{
e.preventDefault()

const fd=new FormData(form)

const r=await fetch("/analyze",{method:"POST",body:fd})
const t=await r.text()

let data
try{data=JSON.parse(t)}catch{alert("Server error");return}

result.innerHTML=`

<h3>Overall: ${data.overall.pronation}</h3>

<h4>Left</h4>
<img src="${data.left_heatmap_data_url}">

<h4>Right</h4>
<img src="${data.right_heatmap_data_url}">

`

}

</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return INDEX_HTML


# ----------------------------------------------------
# Model session
# ----------------------------------------------------

@lru_cache(maxsize=1)
def get_session():
    return new_session("u2netp")


# ----------------------------------------------------
# Image helpers
# ----------------------------------------------------

def upload_bytes(upload: UploadFile):
    b=upload.file.read()
    if not b:
        raise ValueError("Empty upload")
    return b


def data_url(img):

    out=io.BytesIO()
    img.save(out,"PNG")

    return "data:image/png;base64,"+base64.b64encode(out.getvalue()).decode()


# ----------------------------------------------------
# Sole segmentation
# ----------------------------------------------------

def segment_sole(image_bytes):

    session=get_session()

    cut=remove(image_bytes,session=session)

    rgba=Image.open(io.BytesIO(cut)).convert("RGBA")
    arr=np.array(rgba)

    alpha=arr[:,:,3]

    mask=cv2.inRange(alpha,30,255)

    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))

    return mask


# ----------------------------------------------------
# Wear detection
# ----------------------------------------------------

def detect_wear(img,mask):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur=cv2.GaussianBlur(gray,(5,5),0)

    lap=cv2.Laplacian(blur,cv2.CV_32F)
    lap=cv2.convertScaleAbs(lap)

    texture=cv2.blur(lap,(15,15))

    wear=cv2.inRange(texture,0,25)

    wear=cv2.bitwise_and(wear,mask)

    wear=cv2.morphologyEx(wear,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))

    return wear


# ----------------------------------------------------
# Heatmap overlay
# ----------------------------------------------------

def make_overlay(img,sole_mask,wear_mask):

    rgba=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)

    overlay=np.zeros_like(rgba)

    overlay[:,:,0]=255
    overlay[:,:,1]=120
    overlay[:,:,2]=0
    overlay[:,:,3]=np.where(wear_mask>0,180,0)

    overlay[:,:,3]=cv2.GaussianBlur(overlay[:,:,3],(0,0),5)

    contours,_=cv2.findContours(sole_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay,contours,-1,(0,255,120,200),2)

    base=Image.fromarray(rgba)
    over=Image.fromarray(overlay)

    return data_url(Image.alpha_composite(base,over))


# ----------------------------------------------------
# Analysis
# ----------------------------------------------------

def analyze_shoe(bytes_img):

    arr=np.frombuffer(bytes_img,np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)

    mask=segment_sole(bytes_img)

    wear=detect_wear(img,mask)

    overlay=make_overlay(img,mask,wear)

    return overlay


# ----------------------------------------------------
# API
# ----------------------------------------------------

@app.post("/analyze")
async def analyze(
left:UploadFile=File(...),
right:UploadFile=File(...)
):

    try:

        left_bytes=upload_bytes(left)
        right_bytes=upload_bytes(right)

        left_overlay=analyze_shoe(left_bytes)
        right_overlay=analyze_shoe(right_bytes)

        return {

            "left_heatmap_data_url":left_overlay,
            "right_heatmap_data_url":right_overlay,

            "left":{"pronation":"neutral"},
            "right":{"pronation":"neutral"},

            "overall":{
                "pronation":"neutral"
            }

        }

    except Exception as e:

        return JSONResponse({"detail":str(e)},500)
