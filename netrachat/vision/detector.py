"""
Object detection: find the objects in an image.

We use YOLOv8 ("You Only Look Once"), a deep convolutional neural network
pre-trained on the COCO dataset (80 everyday object classes). For every object
it finds, it returns:
  * a class label  (e.g. "dog")
  * a confidence   (0.0 - 1.0, how sure the network is)
  * a bounding box (x1, y1, x2, y2 pixel corners)

On top of that, this module works out extra facts the chatbot can talk about:
the main colour of each object, where it sits in the picture (left / right,
top / bottom) and how big it is.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from dataclasses import dataclass, field

from pathlib import Path

import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_WEIGHTS = "yolov8s.pt"


@dataclass
class Detection:
    """One object found in the image."""
    label: str                     # detector class name, e.g. "person"
    confidence: float              # 0.0 - 1.0
    box: tuple                     # (x1, y1, x2, y2) in pixels
    color: str | None = None       # plain colour name, e.g. "red"
    extra: dict = field(default_factory=dict)

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def color_name(rgb) -> str:
    """Turn an (R, G, B) value into a plain colour name.

    We convert to HSV (hue = which colour, saturation = how vivid,
    value = how bright), which matches human colour naming better than RGB.
    """
    r, g, b = (float(c) / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hue = h * 360
    if v < 0.22:
        return "black"
    if s < 0.22:
        if v > 0.78:
            return "white"
        return "grey" if v > 0.38 else "black"
    if hue < 15 or hue >= 345:
        return "pink" if (s < 0.5 and v > 0.7) else "red"
    if hue < 40:
        return "brown" if v < 0.6 else "orange"
    if hue < 70:
        return "brown" if v < 0.45 else "yellow"
    if hue < 165:
        return "green"
    if hue < 260:
        return "blue"
    if hue < 290:
        return "purple"
    return "pink" if v > 0.6 else "purple"


_CHROMATIC = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"}


def dominant_color(image: Image.Image, box, label: str | None = None) -> str:
    """Estimate the main colour of the object inside `box`.

    We look at the central part of the box (the edges usually contain
    background), give each pixel a colour name and let the pixels "vote".
    For a person we look at the upper body, so the answer describes the
    clothing. Vivid colours win if they cover a fair share of the object,
    because dark windows and shadows otherwise drown them out (a blue bus
    has a lot of black glass).
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if label == "person":
        inner = (x1 + 0.25 * w, y1 + 0.25 * h, x2 - 0.25 * w, y1 + 0.55 * h)
    else:
        inner = (x1 + 0.15 * w, y1 + 0.15 * h, x2 - 0.15 * w, y2 - 0.15 * h)
    crop = image.convert("RGB").crop(tuple(int(v) for v in inner))
    if crop.width < 1 or crop.height < 1:
        return "grey"
    crop.thumbnail((48, 48))
    pixels = np.asarray(crop).reshape(-1, 3)
    votes = Counter(color_name(px) for px in pixels)
    total = sum(votes.values())
    chromatic = [(c, n) for c, n in votes.most_common() if c in _CHROMATIC]
    if chromatic and chromatic[0][1] >= 0.2 * total:
        return chromatic[0][0]
    return votes.most_common(1)[0][0]


class ObjectDetector:
    """Thin wrapper around an Ultralytics YOLO model.

    The model weights download automatically the first time:
      yolov8n.pt  ~6 MB   fastest, least accurate
      yolov8s.pt  ~22 MB  good balance (default)
      yolov8m.pt  ~50 MB  most accurate of the three, slower
    You can also pass the path to your own fine-tuned weights
    (see scripts/train_detector.py).
    """

    def __init__(self, weights: str = DEFAULT_WEIGHTS, min_confidence: float = 0.30):
        from ultralytics import YOLO  # imported here so tests run without it

        path = Path(weights)
        if not path.exists() and path.parent == Path("."):
            # Keep downloaded weights tidy inside the models/ folder.
            MODELS_DIR.mkdir(exist_ok=True)
            path = MODELS_DIR / weights
        self.model = YOLO(str(path))
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[Detection]:
        image = image.convert("RGB")
        result = self.model(image, verbose=False, conf=self.min_confidence)[0]
        detections = []
        for b in result.boxes:
            box = tuple(float(v) for v in b.xyxy[0].tolist())
            label = result.names[int(b.cls)]
            detections.append(Detection(
                label=label,
                confidence=float(b.conf),
                box=box,
                color=dominant_color(image, box, label),
            ))
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
