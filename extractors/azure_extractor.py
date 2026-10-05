"""
azure_extractor.py
-------------------
Extracts equipment-label fields via a vision-capable model deployed in
Azure AI Foundry (Azure OpenAI-compatible endpoint).

Set in .env:
    EXTRACTOR=azure
    AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/
    AZURE_API_KEY=<key>
    AZURE_DEPLOYMENT=<vision-model-deployment-name>
    AZURE_API_VERSION=2024-10-21   # optional, this is the default
"""

import base64
import json

from openai import AzureOpenAI

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


class AzureFoundryExtractor(LabelExtractor):
    """
    Calls a vision-capable model deployed in Azure AI Foundry.

    Requires:
        AZURE_ENDPOINT    Azure OpenAI resource URL
        AZURE_API_KEY     Azure OpenAI API key
        AZURE_DEPLOYMENT  Name of the deployed vision model (e.g. gpt-4o)
        AZURE_API_VERSION optional; default 2024-10-21
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-10-21",
    ):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment = deployment

    def extract(self, image_bytes: bytes) -> dict:
        clean_bytes = preprocess(image_bytes)
        b64 = base64.b64encode(clean_bytes).decode()

        mime = "image/jpeg" if clean_bytes[:3] == b"\xff\xd8\xff" else "image/png"

        response = self.client.chat.completions.create(
            model=self.deployment,
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
            fields = {}

        return {"raw_text": None, "fields": fields}
