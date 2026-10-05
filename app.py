"""
app.py  –  Equipment Label Reader
----------------------------------
Upload a nameplate photo → OpenAI vision reads it → editable fields appear.

Routes
------
GET  /                  home / upload page
POST /read              process image or sample label
POST /export/csv        download edited fields as CSV
POST /export/json       download edited fields as JSON
POST /api/read-label    JSON API  { fields, raw_text }
POST /api/speak         JSON API  → audio/wav (Azure TTS, optional)
GET  /health            health-check JSON

EXTRACTOR in .env:
    openai (default)  needs OPENAI_API_KEY
    azure             needs AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT
"""

import base64
import csv
import io
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── resolve project root ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# Support running from the repo root OR from inside label_reader/
LABEL_READER_DIR = os.path.join(BASE, "label_reader")

def _find(rel, *fallbacks):
    """Return first existing path from candidates, else first candidate."""
    for p in (os.path.join(BASE, rel), *(os.path.join(d, rel) for d in fallbacks)):
        if os.path.exists(p):
            return p
    return os.path.join(BASE, rel)

TEMPLATES_DIR = _find("templates", LABEL_READER_DIR)
STATIC_DIR    = _find("static",    LABEL_READER_DIR)
DEMO_DIR      = _find("demo",      LABEL_READER_DIR)

# Make extractors importable
sys.path.insert(0, BASE)

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.json.sort_keys = False

# ── extractors ────────────────────────────────────────────────────────────────
from extractors.openai_extractor import OpenAIExtractor
from extractors.azure_extractor  import AzureFoundryExtractor

def build_extractor():
    mode = os.getenv("EXTRACTOR", "").lower()
    azure_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT") or os.getenv("VISION_MODEL_DEPLOYMENT")

    # Clean endpoint for AzureOpenAI client (needs base URL like https://name.openai.azure.com/)
    if azure_endpoint and "/openai/v1" in azure_endpoint:
        azure_endpoint = azure_endpoint.replace("/openai/v1", "").rstrip("/") + "/"

    if mode == "azure" or (not mode and azure_key) or (mode != "openai" and azure_key):
        return AzureFoundryExtractor(
            endpoint   = azure_endpoint,
            api_key    = azure_key,
            deployment = azure_deployment or "gpt-4.1-mini",
            api_version= os.getenv("AZURE_API_VERSION", "2024-10-21"),
        )
    return OpenAIExtractor(
        api_key = os.getenv("OPENAI_API_KEY"),
        model   = os.getenv("OPENAI_MODEL", "gpt-4o"),
    )

extractor = build_extractor()

# ── speech (optional) ─────────────────────────────────────────────────────────
from speech import fields_to_speech_text
from speech import is_available  as speech_available
from speech import synthesise

# ── constants ─────────────────────────────────────────────────────────────────
FIELD_ORDER = ["manufacturer","model","serial_number",
               "voltage","frequency","power","manufacture_date"]
LABELS = {
    "manufacturer":    "Manufacturer",
    "model":           "Model",
    "serial_number":   "Serial number",
    "voltage":         "Voltage",
    "frequency":       "Frequency",
    "power":           "Power",
    "manufacture_date":"Manufacture date",
}
SAMPLES  = {"motor":"Motor","pump":"Pump","transformer":"Transformer"}
ALLOWED  = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif", "image/tiff", "image/pjpeg", "image/x-png", "application/octet-stream"}

# ── helpers ───────────────────────────────────────────────────────────────────
def engine_name():
    if isinstance(extractor, AzureFoundryExtractor):
        return "Azure AI Foundry"
    return f"OpenAI {getattr(extractor,'model','vision')}"


def read_fields(image_bytes: bytes):
    result  = extractor.extract(image_bytes)
    fields  = result.get("fields", {})
    ordered = {k: fields.get(k) for k in FIELD_ORDER}
    for k, v in fields.items():
        if k not in ordered and not isinstance(v, (dict, list)):
            ordered[k] = v
    return ordered, result.get("raw_text")


def page(**ctx):
    return render_template(
        "index.html",
        engine           = engine_name(),
        samples          = SAMPLES,
        labels           = LABELS,
        speech_available = speech_available(),
        **ctx,
    )


def edited_fields():
    return {k[2:]: v.strip() for k, v in request.form.items() if k.startswith("f_")}


# ── web routes ────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return page(result=None, error=None)


@app.post("/read")
def read_page():
    sample = request.form.get("sample")
    if sample:
        if sample not in SAMPLES:
            return page(result=None, error="Unknown sample."), 400
        with open(os.path.join(DEMO_DIR, f"{sample}.png"), "rb") as fh:
            data, mimetype = fh.read(), "image/png"
    else:
        f = request.files.get("image")
        if f is None or not f.filename:
            return page(result=None, error="Choose an image photo first."), 400
        data, mimetype = f.read(), f.mimetype

    try:
        fields, raw = read_fields(data)
    except ValueError as e:
        return page(result=None, error=str(e)), 422
    except Exception:
        app.logger.exception("Extraction failed")
        return page(result=None, error="Reading the label failed. Try a clearer photo."), 500

    result = {
        "fields":    fields,
        "raw_text":  raw,
        "found":     sum(1 for v in fields.values() if v),
        "total":     len(fields),
        "image":     f"data:{mimetype};base64,{base64.b64encode(data).decode()}",
        "originals": json.dumps(fields),
    }
    return page(result=result, error=None)


@app.errorhandler(413)
def too_large(_):
    return page(result=None, error="Image is larger than 10 MB. Use a smaller photo."), 413


@app.post("/export/csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    data = edited_fields()
    w.writerow(data.keys())
    w.writerow(data.values())
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=label.csv"})


@app.post("/export/json")
def export_json():
    return Response(json.dumps(edited_fields(), indent=2), mimetype="application/json",
                    headers={"Content-Disposition":"attachment; filename=label.json"})


@app.get("/demo/<path:name>")
def demo_file(name):
    return send_from_directory(DEMO_DIR, name)


# ── JSON API ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify(status="ok", extractor=type(extractor).__name__,
                   engine=engine_name(), speech=speech_available())


@app.post("/api/read-label")
def api_read_label():
    """
    POST multipart/form-data with field 'image' (JPEG or PNG).
    Returns: { fields: {...}, raw_text: null }
    """
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image received. Attach a JPEG or PNG as 'image'."), 400
    if f.mimetype not in ALLOWED:
        return jsonify(error="Only JPEG and PNG photos are supported."), 415
    try:
        fields, raw = read_fields(f.read())
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception:
        app.logger.exception("Extraction failed")
        return jsonify(error="Reading the label failed. Try a clearer photo."), 500
    return jsonify(fields=fields, raw_text=raw)


@app.post("/api/speak")
def api_speak():
    """
    POST JSON { "fields": {...}, "voice": "en-US-JennyNeural" }
    Returns: audio/wav bytes
    Requires AZURE_SPEECH_KEY + AZURE_SPEECH_REGION in .env
    """
    if not speech_available():
        return jsonify(error="Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env to enable TTS."), 503
    body   = request.get_json(silent=True) or {}
    fields = body.get("fields", {})
    if not fields:
        return jsonify(error="Provide 'fields' in the JSON body."), 400
    try:
        audio = synthesise(fields_to_speech_text(fields), voice=body.get("voice"))
    except (ImportError, RuntimeError) as e:
        return jsonify(error=str(e)), 503
    except Exception:
        app.logger.exception("Speech synthesis failed")
        return jsonify(error="Speech synthesis failed."), 500
    return Response(audio, mimetype="audio/wav",
                    headers={"Content-Disposition":"inline; filename=label.wav"})


@app.post("/api/chat")
def api_chat():
    """
    POST JSON { "message": "...", "fields": {...}, "raw_text": "..." }
    Returns: JSON { "reply": "..." }
    """
    body = request.get_json(silent=True) or {}
    user_msg = body.get("message", "").strip()
    fields = body.get("fields", {})
    raw_text = body.get("raw_text")

    if not user_msg:
        return jsonify(error="Message cannot be empty."), 400

    context_str = json.dumps(fields, indent=2)
    system_prompt = (
        "You are an expert industrial equipment and electrical engineering AI assistant. "
        "The user is viewing an equipment nameplate label with the following extracted specifications:\n\n"
        f"EXTRACTED FIELDS:\n{context_str}\n\n"
        f"RAW OCR TEXT (if available):\n{raw_text or 'None'}\n\n"
        "Answer the user's questions clearly, accurately, and concisely based on these specs, "
        "electrical standards (IEC, NEMA, IEEE, IP ratings, voltage/frequency standards), "
        "maintenance best practices, and installation/safety guidelines. Use readable markdown formatting."
    )

    try:
        if isinstance(extractor, AzureFoundryExtractor):
            res = extractor.client.chat.completions.create(
                model=extractor.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=600
            )
        else:
            res = extractor.client.chat.completions.create(
                model=getattr(extractor, "model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=600
            )
        reply = res.choices[0].message.content
        return jsonify(reply=reply)
    except Exception as e:
        app.logger.exception("Chat API error")
        return jsonify(error=f"Chat query failed: {str(e)}"), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
