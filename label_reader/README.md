# Equipment Label Reader

Upload a photo of an equipment nameplate and get its details (manufacturer, model,
serial number, voltage, frequency, power, manufacture date) as editable fields.
Pure Python Flask: server-rendered Jinja templates, no JavaScript. Reads labels locally with Tesseract OCR today;
switches to an Azure AI Foundry vision model later with one setting.

## Run it
1. Install the Tesseract program (Ubuntu: `sudo apt install tesseract-ocr`, macOS: `brew install tesseract`, Windows: UB-Mannheim installer).
2. `pip install -r requirements.txt`
3. `cp .env.example .env`
4. `python app.py`, then open http://127.0.0.1:5000
5. Click a sample (Motor, Pump, Transformer) or drop in your own photo.

API: `curl -F "image=@label.jpg" http://127.0.0.1:5000/api/read-label`

## Tests
`python -m unittest discover -s tests -v`

## Docker
`docker build -t label-reader . && docker run -p 8000:8000 label-reader`

## Switch to Azure AI Foundry later
Deploy a vision-capable model in Foundry, then in `.env` set `EXTRACTOR=azure`,
`AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_DEPLOYMENT`. `extractors/azure_extractor.py`
is written but untested until an account exists.

## Layout
- `app.py` Flask routes (`/`, `/read`, `/export/csv`, `/export/json`, `/api/read-label`, `/health`)
- `extractors/` OCR + rules (`local_extractor.py`), Azure version, image cleanup
- `templates/` Jinja pages (base.html, index.html), `static/style.css`
- `demo/` sample labels, generator script, screenshots
- `tests/` API tests

## Known limits
- Rule-based reading is brittle on messy real-world photos; edit fields in the UI, or use the Azure model.
- OCR can confuse look-alike characters (the transformer sample reads `VT-1500K` as `VT-L500K`).
