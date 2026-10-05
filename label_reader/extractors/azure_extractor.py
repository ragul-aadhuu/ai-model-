import base64
import json

from .base import LabelExtractor

SYSTEM_PROMPT = (
    "You read equipment nameplate/label photos. Return ONLY a JSON object with keys: "
    "manufacturer, model, serial_number, voltage, frequency, power, manufacture_date, "
    "other (object for any extra fields). Use null for anything not visible. Do not guess."
)


class AzureFoundryExtractor(LabelExtractor):
    """Calls a vision-capable model deployed in Azure AI Foundry (Azure OpenAI-compatible API).

    Not tested yet: needs an Azure account and a deployed model.
    """

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str):
        from openai import AzureOpenAI  # imported lazily so local mode needs no Azure setup

        self.client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.deployment = deployment

    def extract(self, image_bytes: bytes) -> dict:
        b64 = base64.b64encode(image_bytes).decode()
        resp = self.client.chat.completions.create(
            model=self.deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the label fields."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
        )
        fields = json.loads(resp.choices[0].message.content)
        return {"raw_text": None, "fields": fields}
