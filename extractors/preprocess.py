import io
from PIL import Image


def preprocess(image_bytes: bytes) -> bytes:
    """
    Validate and convert any input image (JPEG, PNG, WEBP, BMP, TIFF, etc.)
    into RGB JPEG bytes ready for the vision API.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert any mode (RGBA, P, LA, L, CMYK, etc.) to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception as e:
        raise ValueError(
            "Could not decode image. Please choose a valid image file (JPEG, PNG, WEBP, etc.)."
        ) from e
