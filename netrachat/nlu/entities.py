"""
Entity extraction: pull the important words out of the user's message.

Example:  "What colour is the red car next to the bus?"
  -> objects   : ["car", "bus"]
  -> colors    : ["red"]
  -> relation  : "next to"

The intent classifier (intent.py) learns *what kind* of question was asked;
this module finds *what the question is about*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..vocabulary import CATEGORY_WORDS, COLOR_WORDS, PHRASE_TO_LABEL

PRONOUNS = {"it", "its", "they", "them", "their", "he", "she", "him", "her",
            "this one", "that one", "that", "those", "these"}

RELATION_WORDS = {
    "next to": "next to", "beside": "next to", "near": "next to",
    "close to": "next to", "around": "next to", "nearby": "next to",
    "closest to": "next to", "surround": "next to", "surrounds": "next to",
    "left of": "to the left of", "to the left": "to the left of",
    "right of": "to the right of", "to the right": "to the right of",
    "above": "above", "over": "above", "on top of": "above",
    "below": "below", "under": "below", "beneath": "below", "underneath": "below",
    "behind": "behind", "in front of": "in front of",
}

# Longest phrases first so "traffic light" wins over "light".
_OBJECT_PHRASES = sorted(PHRASE_TO_LABEL, key=len, reverse=True)
_CATEGORY_PHRASES = sorted(CATEGORY_WORDS, key=len, reverse=True)
_RELATION_PHRASES = sorted(RELATION_WORDS, key=len, reverse=True)


@dataclass
class Entities:
    objects: list = field(default_factory=list)     # detector labels, in order
    categories: list = field(default_factory=list)  # e.g. ["animal"]
    colors: list = field(default_factory=list)
    relation: str | None = None
    has_pronoun: bool = False
    generic_objects: bool = False                   # "objects", "things", "items"


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find(phrases, text):
    """Find phrases as whole words; return [(position, phrase)] without overlaps."""
    found, taken = [], [False] * len(text)
    for ph in phrases:
        for m in re.finditer(rf"\b{re.escape(ph)}\b", text):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            found.append((m.start(), ph))
    return sorted(found)


def extract(text: str) -> Entities:
    t = normalize(text)
    ents = Entities()
    for _, ph in _find(_CATEGORY_PHRASES, t):
        cat = CATEGORY_WORDS[ph]
        if cat not in ents.categories:
            ents.categories.append(cat)
    for _, ph in _find(_OBJECT_PHRASES, t):
        label = PHRASE_TO_LABEL[ph]
        if label not in ents.objects:
            ents.objects.append(label)
    if ents.objects:
        # A word that is both an object and a category word (e.g. "sports ball")
        # should count as the object only.
        ents.categories = [c for c in ents.categories
                           if not any(c in o for o in ents.objects)]
    ents.colors = [c for c in COLOR_WORDS if re.search(rf"\b{c}\b", t)]
    if re.search(r"\bgray\b", t) and "grey" not in ents.colors:
        ents.colors.append("grey")
    rel = _find(_RELATION_PHRASES, t)
    if rel:
        ents.relation = RELATION_WORDS[rel[0][1]]
    words = set(t.split())
    ents.has_pronoun = bool(words & PRONOUNS) or "this one" in t or "that one" in t
    ents.generic_objects = bool(words & {"objects", "object", "things", "thing", "items",
                                         "item", "everything", "stuff"})
    return ents


def mask(text: str) -> str:
    """Replace object/category/colour words with placeholder tokens.

    The intent classifier then sees "how many obj are there" whatever object
    the user asked about, so it generalises to objects it was never trained on.
    """
    t = normalize(text)
    for _, ph in reversed(_find(_OBJECT_PHRASES + _CATEGORY_PHRASES, t)):
        t = re.sub(rf"\b{re.escape(ph)}\b", "obj", t, count=1)
    for c in COLOR_WORDS + ["gray"]:
        t = re.sub(rf"\b{c}\b", "colorword", t)
    return t
