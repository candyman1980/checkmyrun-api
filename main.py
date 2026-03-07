import os
import json
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # can set to gpt-4o-mini for cheaper tests

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

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>CheckMyRun</title>
  <style>
    body{font-family:system-ui,Segoe UI,Helvetica,Arial;margin:28px;background:#fff;color:#111}
    .card{max-width:980px;margin:0 auto;padding:18px;border-radius:12px;border:1px solid #eee;background:#fafafa}
    .grid{display:flex;gap:12px;flex-wrap:wrap}
    label.drop{display:inline-block;padding:12px;border:2px dashed #ddd;border-radius:10px;cursor:pointer;min-width:220px}
    input[type=file]{display:block;margin-top:8px}
    button{padding:10px 14px;border-radius:8px;border:1px solid #111;background:#fff;cursor:pointer}
    pre{background:#f4f6f8;padding:12px;border-radius:8px;overflow:auto;max-height:360px;white-space:pre-wrap}
    .result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:12px}
    .result-card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:12px}
    .muted{color:#666}
  </style>
</head>
<body>
  <div class="card">
    <h1>CheckMyRun</h1>
    <p>Upload clear photos of both soles. You'll get a pronation estimate and shoe category suggestion.</p>

    <form id="form" enctype="multipart/form-data">
      <div class="grid">
        <label class="drop">
          Left sole
          <input type="file" name="left" accept="image/*" required>
        </label>

        <label class="drop">
          Right sole
          <input type="file" name="right" accept="image/*" required>
        </label>
      </div>

      <div style="margin-top:12px">
        <button id="btn" type="submit">Analyse</button>
        <span id="status" style="margin-left:12px;color:#666"></span>
      </div>
    </form>

    <div id="result" style="margin-top:18px;display:none">
      <h2>Result</h2>
      <div id="summary"></div>
      <div class="result-grid">
        <div class="result-card">
          <h3>Left</h3>
          <div id="leftResult"></div>
        </div>
        <div class="result-card">
          <h3>Right</h3>
          <div id="rightResult"></div>
        </div>
        <div class="result-card">
          <h3>Overall</h3>
          <div id="overallResult"></div>
        </div>
        <div class="result-card">
          <h3>Photo quality</h3>
          <div id="qualityResult"></div>
        </div>
      </div>

      <details style="margin-top:14px">
        <summary>Raw JSON</summary>
        <pre id="json"></pre>
      </details>
    </div>

    <footer style="margin-top:14px;font-size:13px;color:#666">
      Informational only — not medical advice.
    </footer>
  </div>

<script>
const API = "/analyze";
const form = document.getElementById("form");
const btn = document.getElementById("btn");
const status = document.getElementById("status");
const result = document.getElementById("result");
const summary = document.getElementById("summary");
const leftResult = document.getElementById("leftResult");
const rightResult = document.getElementById("rightResult");
const overallResult = document.getElementById("overallResult");
const qualityResult = document.getElementById("qualityResult");
const jsonOut = document.getElementById("json");

function prettyLabel(v) {
  if (!v) return "—";
  return String(v).replace(/-/g, " ");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  btn.disabled = true;
  status.textContent = "Uploading...";
  result.style.display = "none";
  summary.innerHTML = "";
  leftResult.innerHTML = "";
  rightResult.innerHTML = "";
  overallResult.innerHTML = "";
  qualityResult.innerHTML = "";
  jsonOut.textContent = "";

  try {
    const fd = new FormData(form);
    const res = await fetch(API, {
      method: "POST",
      body: fd
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || JSON.stringify(data));
    }

    summary.innerHTML = `
      <p><strong>Overall pronation:</strong> ${prettyLabel(data.overall?.pronation)}</p>
      <p><strong>Shoe category:</strong> ${prettyLabel(data.overall?.shoe_category)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.overall?.confidence || 0) * 100)}%</p>
    `;

    leftResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.left?.pronation)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.left?.confidence || 0) * 100)}%</p>
      <p>${data.left?.notes || ""}</p>
    `;

    rightResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.right?.pronation)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.right?.confidence || 0) * 100)}%</p>
      <p>${data.right?.notes || ""}</p>
    `;

    overallResult.innerHTML = `
      <p><strong>Pronation:</strong> ${prettyLabel(data.overall?.pronation)}</p>
      <p><strong>Shoe category:</strong> ${prettyLabel(data.overall?.shoe_category)}</p>
      <p><strong>Confidence:</strong> ${Math.round((data.overall?.confidence || 0) * 100)}%</p>
    `;

    qualityResult.innerHTML = `
      <p><strong>OK:</strong> ${data.photo_quality?.ok ? "Yes" : "No"}</p>
      <p><strong>Issues:</strong></p>
      <ul>${(data.photo_quality?.issues || []).map(i => `<li>${i}</li>`).join("") || "<li>None</li>"}</ul>
    `;

    jsonOut.textContent = JSON.stringify(data, null, 2);
    result.style.display = "block";
    status.textContent = "Done ✅";
  } catch (err) {
    status.textContent = "Error";
    summary.innerHTML = `<p style="color:#b00020"><strong>${err.message}</strong></p>`;
    result.style.display = "block";
  } finally {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=INDEX_HTML, status_code=200)

@app.get("/health")
def health():
    return {"ok": True, "service": "checkmyrun-api", "marker": "OPENAI-V2", "model": MODEL}

def to_data_url(upload: UploadFile) -> str:
    b = upload.file.read()
    if not b:
        raise ValueError("Empty upload")

    name = (upload.filename or "").lower()
    if name.endswith(".png"):
        mime = "image/png"
    else:
        mime = "image/jpeg"

    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_output_text(resp_json: dict) -> str:
    out = []
    for item in resp_json.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "\\n".join(out).strip()

@app.post("/analyze")
@app.post("/analyse")
@app.post("/api/analyze")
@app.post("/api/analyse")
async def analyze(left: UploadFile = File(...), right: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in Render env vars for this service")

    try:
        left_url = to_data_url(left)
        right_url = to_data_url(right)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad upload: {e}")

    response_schema = {
        "name": "checkmyrun_pronation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "left": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "notes": {"type": "string"},
                    },
                    "required": ["pronation", "confidence", "notes"],
                },
                "right": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "notes": {"type": "string"},
                    },
                    "required": ["pronation", "confidence", "notes"],
                },
                "overall": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pronation": {
                            "type": "string",
                            "enum": ["overpronation", "underpronation", "neutral", "unclear"]
                        },
                        "shoe_category": {
                            "type": "string",
                            "enum": ["stability", "neutral", "cushioned-neutral", "unclear"]
                        },
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
            },
            "required": ["left", "right", "overall", "photo_quality"],
        },
    }

    instruction = (
        "You are a running shoe fitting assistant. "
        "You will be given two outsole (sole) photos: LEFT and RIGHT shoe. "
        "Infer pronation style from wear patterns. "
        "Be conservative: if wear is unclear, output 'unclear' with low confidence. "
        "No medical advice. Notes must be short (1–2 sentences). "
        "Also assess photo quality and list issues. "
        "Return ONLY valid JSON matching the schema."
    )

    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": "LEFT SOLE:"},
                    {"type": "input_image", "image_url": left_url},
                    {"type": "input_text", "text": "RIGHT SOLE:"},
                    {"type": "input_image", "image_url": right_url},
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
        "max_output_tokens": 500,
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
        return {"error": "Model returned non-JSON unexpectedly", "raw": text}
