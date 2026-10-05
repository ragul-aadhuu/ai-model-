import re

import pytesseract

from .base import LabelExtractor
from .preprocess import preprocess

PATTERNS = {
    "serial_number": r"(?:S/?N|Serial(?:[ \t]*No\.?)?)[ \t]*[:#]?[ \t]*([A-Z0-9\-]{4,})",
    "model": r"(?:Model|MOD(?:EL)?)[ \t]*[:#]?[ \t]*([A-Z0-9\-\/]{3,})",
    "manufacturer": r"(?:Mfr|Manufacturer|Made by)[ \t]*[:#]?[ \t]*([^\n]+)",
    "voltage": r"(?<![\w\-])(\d{2,6}[ \t]?V(?:AC|DC)?)(?![A-Za-z])",
    "frequency": r"(?<![\w\-])(\d{2,3}[ \t]?Hz)\b",
    "power": r"(?<![\w\-])(\d+(?:\.\d+)?[ \t]?(?:kVA|kW|HP|VA|W))(?![A-Za-z])",
    "manufacture_date": r"(?:Date|Mfg\.?[ \t]*Date|DOM)[ \t]*[:#]?[ \t]*([0-9]{1,4}[\/\-\.][0-9]{1,2}(?:[\/\-\.][0-9]{1,4})?)",
}


class LocalExtractor(LabelExtractor):
    def extract(self, image_bytes: bytes) -> dict:
        processed = preprocess(image_bytes)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        fields = {}
        for name, pattern in PATTERNS.items():
            m = re.search(pattern, text, re.IGNORECASE)
            fields[name] = m.group(1).strip() if m else None
        if not fields["manufacturer"]:
            first = next((l.strip() for l in text.splitlines() if l.strip()), None)
            fields["manufacturer"] = first  # heuristic: brand is usually the top line
        return {"raw_text": text, "fields": fields}
