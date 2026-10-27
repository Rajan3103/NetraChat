"""Draw the detected objects on the image (boxes + labels) for the UI."""

from PIL import Image, ImageDraw, ImageFont

_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
            "#f032e6", "#bcf60c", "#008080", "#9a6324"]


def annotate(image: Image.Image, detections) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 300)
    try:
        font = ImageFont.load_default(size=max(12, img.width // 50))
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    colors = {}
    for d in detections:
        color = colors.setdefault(d.label, _PALETTE[len(colors) % len(_PALETTE)])
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{d.label} {d.confidence:.0%}"
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
        top = max(0, y1 - (ty2 - ty1) - 4)
        draw.rectangle([x1, top, x1 + (tx2 - tx1) + 6, top + (ty2 - ty1) + 4], fill=color)
        draw.text((x1 + 3, top + 1), text, fill="white", font=font)
    return img
