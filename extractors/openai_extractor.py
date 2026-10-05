"""
openai_extractor.py
--------------------
Extracts equipment-label fields using the OpenAI Chat Completions API
with vision support (gpt-4o, gpt-4-turbo, etc.).

Set in .env:
    EXTRACTOR=openai
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o          # optional, default gpt-4o
"""

import base64
import json
import os

from openai import OpenAI

from .base import LabelExtractor
from .preprocess import preprocess

SYSTEM_PROMPT = (
    "You are an expert at reading industrial equipment nameplates and labels. "
    "Examine the image and return ONLY a valid JSON object — no markdown, no code fences — "
    "with exactly these keys: "
    "manufacturer, model, serial_number, voltage, frequency, power, manufacture_date, other. "
    "'other' should be a JSON object containing any additional fields you can read from the label "
    "(e.g. phase, rpm, IP rating, weight). "
    "Use null for any field not visible or not legible. Do not guess or invent values."
)


class OpenAIExtractor(LabelExtractor):
    """
    Calls the OpenAI vision API to read equipment labels.

    Requires:
        OPENAI_API_KEY  environment variable (or set via .env)
        OPENAI_MODEL    optional; defaults to 'gpt-4o'
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    def extract(self, image_bytes: bytes) -> dict:
        clean_bytes = preprocess(image_bytes)
        b64 = base64.b64encode(clean_bytes).decode()

        # Detect mime for the data-URL
        mime = "image/jpeg" if clean_bytes[:3] == b"\xff\xd8\xff" else "image/png"

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all label fields from this nameplate photo."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                        },
                    ],
                },
            ],
            max_tokens=512,
        )

        raw_json = response.choices[0].message.content
        try:
            fields = json.loads(raw_json)
        except json.JSONDecodeError:
            # Graceful fallback — return empty fields rather than crashing
            fields = {}

        return {"raw_text": None, "fields": fields}
