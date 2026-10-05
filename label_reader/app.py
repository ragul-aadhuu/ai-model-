import base64
import csv
import io
import json
import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from extractors.azure_extractor import AzureFoundryExtractor
from extractors.local_extractor import LocalExtractor

load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
app.json.sort_keys = False  # keep fields in label order

FIELD_ORDER = [
    "manufacturer", "model", "serial_number",
    "voltage", "frequency", "power", "manufacture_date",
]
LABELS = {
    "manufacturer": "Manufacturer", "model": "Model", "serial_number": "Serial number",
    "voltage": "Voltage", "frequency": "Frequency", "power": "Power",
    "manufacture_date": "Manufacture date",
}
SAMPLES = {"motor": "Motor", "pump": "Pump", "transformer": "Transformer"}
ALLOWED = {"image/jpeg", "image/png"}


def build_extractor():
    if os.getenv("EXTRACTOR", "local").lower() == "azure":
        return AzureFoundryExtractor(
            endpoint=os.environ["AZURE_ENDPOINT"],
            api_key=os.environ["AZURE_API_KEY"],
            deployment=os.environ["AZURE_DEPLOYMENT"],
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
        )
    return LocalExtractor()


extractor = build_extractor()


def engine_name():
    return "Azure AI Foundry" if isinstance(extractor, AzureFoundryExtractor) else "this computer"


def read_fields(image_bytes):
    """Run the extractor and return (ordered_fields, raw_text)."""
    result = extractor.extract(image_bytes)
    fields = result.get("fields", {})
    ordered = {k: fields.get(k) for k in FIELD_ORDER}
    for k, v in fields.items():
        if k not in ordered and not isinstance(v, (dict, list)):
            ordered[k] = v
    return ordered, result.get("raw_text")


def page(**ctx):
    return render_template("index.html", engine=engine_name(), samples=SAMPLES,
                           labels=LABELS, **ctx)


# ---------- web pages (server-rendered) ----------

@app.get("/")
def index():
    return page(result=None, error=None)


@app.post("/read")
def read_page():
    sample = request.form.get("sample")
    if sample:
        if sample not in SAMPLES:
            return page(result=None, error="Unknown sample."), 400
        with open(os.path.join(BASE, "demo", f"{sample}.png"), "rb") as fh:
            data, mimetype = fh.read(), "image/png"
    else:
        f = request.files.get("image")
        if f is None or not f.filename:
            return page(result=None, error="Choose a JPEG or PNG photo first."), 400
        if f.mimetype not in ALLOWED:
            return page(result=None, error="Only JPEG and PNG photos are supported."), 415
        data, mimetype = f.read(), f.mimetype
    try:
        fields, raw = read_fields(data)
    except ValueError as e:
        return page(result=None, error=str(e)), 422
    except Exception:
        app.logger.exception("Extraction failed")
        return page(result=None, error="Reading the label failed. Try a sharper photo."), 500
    result = {
        "fields": fields,
        "raw_text": raw,
        "found": sum(1 for v in fields.values() if v),
        "total": len(fields),
        "image": f"data:{mimetype};base64,{base64.b64encode(data).decode()}",
        "originals": json.dumps(fields),
    }
    return page(result=result, error=None)


@app.errorhandler(413)
def too_large(_):
    return page(result=None, error="Image is larger than 10 MB. Use a smaller photo."), 413


def edited_fields():
    return {k[2:]: v.strip() for k, v in request.form.items() if k.startswith("f_")}


@app.post("/export/csv")
def export_csv():
    data = edited_fields()
    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL)
    w.writerow(data.keys())
    w.writerow(data.values())
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=label.csv"})


@app.post("/export/json")
def export_json():
    return Response(json.dumps(edited_fields(), indent=2), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=label.json"})


@app.get("/demo/<path:name>")
def demo_file(name):
    return send_from_directory(os.path.join(BASE, "demo"), name)


# ---------- JSON API (for other apps) ----------

@app.get("/health")
def health():
    return jsonify(status="ok", extractor=type(extractor).__name__)


@app.post("/api/read-label")
def api_read_label():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image received. Choose a JPEG or PNG photo."), 400
    if f.mimetype not in ALLOWED:
        return jsonify(error="Only JPEG and PNG photos are supported."), 415
    try:
        fields, raw = read_fields(f.read())
    except ValueError as e:
        return jsonify(error=str(e)), 422
    except Exception:
        app.logger.exception("Extraction failed")
        return jsonify(error="Reading the label failed. Try a sharper photo."), 500
    return jsonify(fields=fields, raw_text=raw)


if __name__ == "__main__":
    app.run(debug=True)
