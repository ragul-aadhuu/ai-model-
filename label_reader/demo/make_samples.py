"""Generate sample nameplate images for the demo (no real photos needed)."""
import glob
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = (glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True) or [None])[0]

SAMPLES = {
    "motor": ((205, 205, 210), [
        "ACME INDUSTRIAL MOTORS", "Model: AX-4500/T", "Serial No: SN-2024-778213",
        "Voltage: 415 VAC", "Frequency: 50 Hz", "Power: 7.5 kW", "Mfg Date: 2024/03/18"]),
    "pump": ((222, 214, 190), [
        "HYDROFLOW PUMPS LTD", "Model: HF-220/B", "S/N: HP9981234",
        "Voltage: 230 VAC", "Frequency: 60 Hz", "Power: 3 HP", "Date: 2023/11/02"]),
    "transformer": ((190, 200, 210), [
        "VOLTEX POWER SYSTEMS", "Model: VT-1500K", "Serial: TR-55-20931",
        "Voltage: 11000 V", "Frequency: 50 Hz", "Power: 500 kVA", "DOM: 2022/07/15"]),
}


def make(name, bg, lines, blur=0.0):
    font = ImageFont.truetype(FONT, 34) if FONT else ImageFont.load_default()
    img = Image.new("RGB", (900, 520), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 889, 509], outline=(40, 40, 40), width=4)
    y = 35
    for line in lines:
        d.text((40, y), line, fill=(20, 20, 20), font=font)
        y += 65
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    img.save(os.path.join(HERE, f"{name}.png"))


if __name__ == "__main__":
    for name, (bg, lines) in SAMPLES.items():
        make(name, bg, lines)
    print("samples written to", HERE)
