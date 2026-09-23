"""Load the trained pipeline and classify new text.

Kept separate from train.py on purpose: the web app must never import
training code. It loads one artifact and calls .predict_proba() on it.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib

from dataset import clean_text

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
META_PATH = MODELS_DIR / "model_meta.json"

# ---------------------------------------------------------------------
# The neutral band.
#
# The model was trained on a binary dataset, so predict() can only ever say
# negative or positive -- there is no third option to output. But plenty of
# real comments carry no sentiment at all ("what's the price?", "@Rahim look
# at this"). Forcing those into pos/neg is how a dashboard ends up reporting
# swings that are pure noise.
#
# The fix costs nothing: read the *probability* rather than the hard label.
# A genuinely neutral comment has no strong evidence either way, so it lands
# near 0.5. We carve out a band around the middle and call it neutral.
#
# This is a calibration choice, not a trained one -- widen the band and you
# get fewer, more confident pos/neg calls; narrow it and you get more
# coverage but more noise. Tune it by eyeballing your own comments.
# ---------------------------------------------------------------------
NEUTRAL_LOW = 0.40
NEUTRAL_HIGH = 0.60


@lru_cache(maxsize=1)
def load_pipeline():
    """Load the fitted pipeline once and reuse it for the process lifetime.

    lru_cache matters here: joblib.load() off disk takes ~100ms, which is
    fine once and unacceptable per request.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.\n"
            f"Train one first:  uv run python train.py"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_meta() -> dict:
    if not META_PATH.exists():
        return {}
    return json.loads(META_PATH.read_text())


def classify(texts: list[str],
             low: float = NEUTRAL_LOW,
             high: float = NEUTRAL_HIGH) -> list[dict]:
    """Classify a batch of raw comment strings.

    Always pass a *list*, even for one comment -- vectorising 500 strings in
    one call is dramatically faster than 500 separate calls, because the
    sparse matrix operations are what get vectorised.

    Returns one dict per input:
        {"sentiment": "positive"|"negative"|"neutral",
         "score": P(positive) as a float,
         "confidence": distance from the fence, 0..1}
    """
    if not texts:
        return []

    pipeline = load_pipeline()

    # Identical cleaning to training. If these two ever drift apart, the model
    # sees a different distribution than it was fitted on and accuracy drops
    # for no visible reason -- which is why both call the same function.
    cleaned = [clean_text(t or "") for t in texts]

    scores = pipeline.predict_proba(cleaned)[:, 1]

    results = []
    for raw, clean, score in zip(texts, cleaned, scores):
        if not clean:
            # Nothing survived cleaning (emoji-only, URL-only, Bangla script).
            # The model has no basis for an opinion; saying so is more honest
            # than reporting whatever the all-zero vector happens to produce.
            results.append({
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.0,
                "reason": "no usable text after cleaning",
            })
            continue

        if score >= high:
            sentiment = "positive"
        elif score <= low:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        results.append({
            "sentiment": sentiment,
            "score": round(float(score), 4),
            "confidence": round(float(abs(score - 0.5) * 2), 4),
        })

    return results


def classify_one(text: str) -> dict:
    """Single-comment convenience wrapper, for the try-it-yourself box."""
    return classify([text])[0]


if __name__ == "__main__":
    # Quick smoke test: uv run python predict.py
    samples = [
        "absolutely love this product, best purchase ever",
        "terrible quality, broke after two days, waste of money",
        "what is the price?",
        "when will you restock this item",
        "not bad at all, pretty happy with it",
        "i cant wait for the next one",
        "",
    ]
    width = max(len(s) for s in samples) + 2
    for text, result in zip(samples, classify(samples)):
        print(f"{text!r:<{width}} -> {result['sentiment']:<9} "
              f"score={result['score']:.3f} conf={result['confidence']:.3f}")
