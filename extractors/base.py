from abc import ABC, abstractmethod


class LabelExtractor(ABC):
    @abstractmethod
    def extract(self, image_bytes: bytes) -> dict:
        """Return {'raw_text': str | None, 'fields': {...}}"""
