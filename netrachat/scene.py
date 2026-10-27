"""
Scene understanding: turn a list of detections into facts about the image.

The detector only tells us "there is a box labelled X here". This module
reasons over all the boxes together to answer questions like:
  * How many of each object are there?
  * Where is an object (left, right, top, bottom, centre)?
  * What is next to / above / left of what?
  * What is the main subject (biggest object)?
  * What kind of place is this (street, kitchen, office...)?
"""

from __future__ import annotations

import math
from collections import Counter

from .vision.detector import Detection

# Objects that hint at the kind of place shown. Each object adds a vote.
SCENE_HINTS = {
    "a street scene": {"car", "bus", "truck", "traffic light", "stop sign",
                       "motorcycle", "bicycle", "fire hydrant", "parking meter"},
    "a kitchen": {"oven", "microwave", "refrigerator", "toaster", "sink", "knife",
                  "bowl", "spoon"},
    "a living room": {"couch", "tv", "remote", "potted plant", "vase", "clock"},
    "a bedroom": {"bed", "teddy bear"},
    "a dining scene": {"dining table", "fork", "wine glass", "cup", "pizza",
                       "sandwich", "cake", "hot dog", "donut", "bottle"},
    "an office or workspace": {"laptop", "keyboard", "mouse", "book", "cell phone"},
    "a wildlife or farm scene": {"elephant", "zebra", "giraffe", "cow", "sheep",
                                 "horse", "bear", "bird"},
    "a sports scene": {"sports ball", "frisbee", "tennis racket", "baseball bat",
                       "baseball glove", "skis", "snowboard", "skateboard", "kite"},
    "a beach or water scene": {"surfboard", "boat"},
    "an airport": {"airplane"},
    "a railway scene": {"train"},
    "a bathroom": {"toilet", "toothbrush", "hair drier"},
}


class Scene:
    def __init__(self, detections: list[Detection], width: int, height: int,
                 caption: str | None = None):
        self.detections = detections
        self.width = width
        self.height = height
        self.caption = caption
        self.counts = Counter(d.label for d in detections)

    # ---- basic facts --------------------------------------------------------
    def labels(self) -> list[str]:
        """Detected classes, most frequent first."""
        return [label for label, _ in self.counts.most_common()]

    def count(self, label: str) -> int:
        return self.counts.get(label, 0)

    def of(self, label: str) -> list[Detection]:
        return [d for d in self.detections if d.label == label]

    def main_subject(self) -> Detection | None:
        """The largest object usually is what the photo is 'about'."""
        return max(self.detections, key=lambda d: d.area, default=None)

    # ---- positions ----------------------------------------------------------
    def position(self, det: Detection) -> str:
        """Plain-English location, e.g. 'on the left', 'at the bottom right'."""
        cx, cy = det.center
        fx, fy = cx / self.width, cy / self.height
        horiz = "left" if fx < 0.36 else "right" if fx > 0.64 else ""
        vert = "top" if fy < 0.33 else "bottom" if fy > 0.67 else ""
        if horiz and vert:
            return f"at the {vert} {horiz}"
        if vert:
            return f"at the {vert}"
        if horiz:
            return f"on the {horiz}"
        return "in the centre"

    def size_word(self, det: Detection) -> str:
        frac = det.area / float(self.width * self.height)
        if frac > 0.4:
            return "large"
        if frac > 0.08:
            return "medium-sized"
        return "small"

    # ---- relations between two objects -------------------------------------
    def relation(self, a: Detection, b: Detection) -> str:
        """Describe where `a` is relative to `b`."""
        (ax, ay), (bx, by) = a.center, b.center
        dx, dy = ax - bx, ay - by
        diag = math.hypot(self.width, self.height)
        if self._overlap(a, b) > 0.5:
            return "in front of" if a.box[3] >= b.box[3] else "behind"
        if math.hypot(dx, dy) < 0.18 * diag:
            return "next to"
        if abs(dx) >= abs(dy):
            return "to the left of" if dx < 0 else "to the right of"
        return "above" if dy < 0 else "below"

    @staticmethod
    def _overlap(a: Detection, b: Detection) -> float:
        """Share of the smaller box that is covered by the other box."""
        x1, y1 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
        x2, y2 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        smaller = min(a.area, b.area) or 1
        return inter / smaller

    def neighbours(self, det: Detection, k: int = 3) -> list[Detection]:
        """The k other objects closest to `det`."""
        others = [d for d in self.detections if d is not det]
        cx, cy = det.center
        others.sort(key=lambda d: math.hypot(d.center[0] - cx, d.center[1] - cy))
        return others[:k]

    # ---- whole-image interpretation ----------------------------------------
    def scene_type(self) -> str | None:
        votes = Counter()
        for name, hints in SCENE_HINTS.items():
            votes[name] = sum(self.counts[l] for l in hints if l in self.counts)
        best, score = votes.most_common(1)[0] if votes else (None, 0)
        if score > 0:
            return best
        if self.counts and set(self.counts) == {"person"}:
            return "a photo of people"
        return None
