"""
Train and evaluate the intent classifier.

    python scripts/train_intent.py

1. Measures accuracy with 5-fold cross-validation: the data is split into 5
   parts; each time the model trains on 4 parts and is tested on the 5th,
   so every score comes from sentences the model never saw while training.
2. Trains the final model on all the data and saves it to
   models/intent_model.joblib, which the chatbot loads at start-up.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from netrachat.nlu.intent import (MODEL_PATH, augment, build_pipeline,  # noqa: E402
                                  load_dataset, rule_intent, save, train)


def main():
    texts, labels = load_dataset()
    print(f"Loaded {len(texts)} examples across {len(set(labels))} intents.\n")

    ml_pred, hybrid_pred, truth = [], [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in cv.split(texts, labels):
        tr_t, tr_l = augment([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        model = build_pipeline().fit(tr_t, tr_l)
        test_t = [texts[i] for i in test_idx]
        preds = model.predict(test_t)
        ml_pred += list(preds)
        hybrid_pred += [rule_intent(t) or p for t, p in zip(test_t, preds)]
        truth += [labels[i] for i in test_idx]

    print(f"Machine-learning model alone : {accuracy_score(truth, ml_pred):.1%} accuracy")
    print(f"Model + keyword rules (used) : {accuracy_score(truth, hybrid_pred):.1%} accuracy\n")
    print("Per-intent results (model + rules) on unseen sentences:")
    print(classification_report(truth, hybrid_pred, digits=3, zero_division=0))

    save(train())
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
