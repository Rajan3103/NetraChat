"""
Intent classification: decide what the user wants.

This is the part of NetraChat that is *trained*. We take the example
sentences in data/intents.json, mask object and colour words (see
entities.mask), and train a machine-learning classifier:

    text  --TF-IDF-->  numeric features  --Logistic Regression-->  intent

TF-IDF turns a sentence into numbers based on which words and letter
sequences it contains; Logistic Regression learns which features point to
which intent. Letter n-grams make it tolerant of typos ("hw mny dogs").
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .entities import mask

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "intents.json"
MODEL_PATH = ROOT / "models" / "intent_model.joblib"


def load_dataset(path: Path = DATA_PATH):
    """Return (texts, labels) with object placeholders masked."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))["intents"]
    texts, labels = [], []
    for intent, examples in data.items():
        for ex in examples:
            texts.append(mask(ex.replace("{obj}", "dog")))
            labels.append(intent)
    return texts, labels


_PREFIXES = ["", "", "please ", "can you tell me ", "hey ", "ok ", "so ", "bot, ",
             "could you tell me ", "i want to know "]
_SUFFIXES = ["", "", " please", " in the image", " in this picture", " now", " ?"]


def augment(texts, labels, copies: int = 3, seed: int = 0):
    """Create extra training sentences from the real ones.

    Adds polite prefixes/suffixes and small typos so the model copes with
    the many ways real users type. Used only on training data, never on the
    sentences used to measure accuracy.
    """
    rng = random.Random(seed)
    out_t, out_l = list(texts), list(labels)
    for text, label in zip(texts, labels):
        for _ in range(copies):
            t = rng.choice(_PREFIXES) + text + rng.choice(_SUFFIXES)
            if len(t) > 6 and rng.random() < 0.3:          # simulate a typo
                i = rng.randrange(1, len(t) - 1)
                t = t[:i] + t[i + 1:]
            out_t.append(t.strip())
            out_l.append(label)
    return out_t, out_l


# High-precision keyword rules. When one matches we trust it over the
# statistical model; everything else is left to the classifier.
RULES = [
    (r"^how many\b|\bcount (the|all|them|how)\b|\bnumber of\b", "count"),
    (r"\bwhat colou?rs?\b|\bwhich colou?r\b|\bcolou?r of\b", "color"),
    (r"\bwhere (is|are) (the|it|they|that|those|these) (?!place|photo|picture|image)", "location"),
    (r"^(is|are) there (a|an|any)\b", "exists"),
    (r"^(tell me|talk|say something) (more )?about (the|this|that|these|those) obj", "describe"),
    (r"^(thanks|thank you|thx)\b", "thanks"),
    (r"^(bye|goodbye|good bye|see you)\b", "goodbye"),
]


def rule_intent(masked_text: str) -> str | None:
    for pattern, intent in RULES:
        if re.search(pattern, masked_text):
            return intent
    return None


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                  token_pattern=r"[a-z']+", sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=20, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def train(path: Path = DATA_PATH) -> Pipeline:
    texts, labels = augment(*load_dataset(path))
    model = build_pipeline()
    model.fit(texts, labels)
    return model


def save(model: Pipeline, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "sklearn": sklearn.__version__}, path)


def load_saved(path: Path = MODEL_PATH):
    """Load the saved model, or return None if it is missing or was saved by a
    different scikit-learn version (then we simply retrain, which takes seconds)."""
    try:
        saved = joblib.load(path)
        if saved.get("sklearn") == sklearn.__version__:
            return saved["model"]
    except Exception:
        pass
    return None


def known_words(path: Path = DATA_PATH) -> set[str]:
    """Every word that appears in the training sentences."""
    texts, _ = load_dataset(path)
    return {w for t in texts for w in t.split()}


class IntentClassifier:
    """Loads the saved model (or trains one on the spot if none is saved)."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = load_saved(Path(model_path)) or train()
        self.vocabulary = known_words()

    def is_gibberish(self, text: str) -> bool:
        """True if not a single word of the message is one we know."""
        words = mask(text).split()
        return bool(words) and not any(w in self.vocabulary or w == "obj" for w in words)

    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent, probability) for one message."""
        masked = mask(text)
        ruled = rule_intent(masked)
        if ruled:
            return ruled, 1.0
        probs = self.model.predict_proba([masked])[0]
        best = probs.argmax()
        return self.model.classes_[best], float(probs[best])
