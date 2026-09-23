"""Train the sentiment classifier.

Run:
    uv run python train.py                # fast: 200k sample, good for iterating
    uv run python train.py --sample 50000 # faster still
    uv run python train.py --full         # all 1.6M rows

What this script demonstrates, beyond "fit a model":

  1. Pipeline    -- vectoriser and classifier travel together as ONE object
  2. val vs test -- configurations are compared on validation; test is touched once
  3. CV          -- cross-validation, so a good score can't be a lucky split
  4. persistence -- the fitted pipeline is saved, so the API never re-trains
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from dataset import LABEL_NAMES, SEED, load_splits

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
META_PATH = MODELS_DIR / "model_meta.json"


# Each entry is one experiment. The interesting comparison is unigram vs
# bigram, and note what changes alongside it: the bigram configs drop
# stop-word removal. That is not an oversight.
#
#   sklearn's English stop-word list contains "not", "no", "cannot".
#   Strip those and "not bad" and "not happy" cease to exist -- you would be
#   deleting the exact negation cues that bigrams were added to capture.
#
# Feature engineering decisions interact. Changing one in isolation is how
# you end up with a worse model and no idea why.
CONFIGS: dict[str, dict] = {
    "unigram-5k": dict(ngram_range=(1, 1), max_features=5_000, stop_words="english"),
    "bigram-10k": dict(ngram_range=(1, 2), max_features=10_000, stop_words=None),
    "bigram-50k": dict(ngram_range=(1, 2), max_features=50_000, stop_words=None),
}


def build_pipeline(seed: int = SEED, **vectorizer_kwargs) -> Pipeline:
    """Bundle the vectoriser and the classifier into a single fitted object.

    This is the single most important habit in this file. The common beginner
    bug is to save only the LogisticRegression and keep the TfidfVectorizer as
    a loose variable. At prediction time you then either cannot transform new
    text at all, or -- much worse -- you call fit_transform() on the new text,
    which builds a *brand new vocabulary* from it. Column 7 no longer means
    what it meant during training, and predictions silently become garbage.
    No exception is raised. The accuracy just quietly collapses.

    A Pipeline makes that mistake unexpressible: .fit() fits both, .predict()
    transforms-then-predicts, and joblib.dump() saves both together.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, **vectorizer_kwargs)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])


def evaluate(pipeline: Pipeline, X, y, label: str) -> dict:
    """Score a fitted pipeline on one split and print a short report."""
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    # ROC-AUC reads the ranking quality of the probabilities rather than the
    # hard 0/1 calls, so it is unaffected by where you put the threshold --
    # which matters here, because we shift that threshold later for "neutral".
    auc = roc_auc_score(y, probabilities)

    print(f"\n  {label}:  accuracy {accuracy:.4f}   roc-auc {auc:.4f}")
    return {"accuracy": float(accuracy), "roc_auc": float(auc)}


def show_top_features(pipeline: Pipeline, n: int = 15) -> dict[str, list[str]]:
    """Print the features pushing hardest toward each class.

    Logistic regression assigns one weight per feature, so the model is
    directly readable -- a property you lose the moment you move to a neural
    model. Always look at these: they are your cheapest bug detector. If a
    username, a date or a stray token shows up in the top weights, your
    cleaning has a leak.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    top_positive = np.argsort(coefficients)[-n:][::-1]
    top_negative = np.argsort(coefficients)[:n]

    print(f"\n  Strongest POSITIVE features:")
    print("   ", ", ".join(f"{names[i]}" for i in top_positive))
    print(f"  Strongest NEGATIVE features:")
    print("   ", ", ".join(f"{names[i]}" for i in top_negative))

    return {
        "positive": [str(names[i]) for i in top_positive],
        "negative": [str(names[i]) for i in top_negative],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200_000,
                        help="rows to sample (default 200000)")
    parser.add_argument("--full", action="store_true",
                        help="use all 1.6M rows, overrides --sample")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv", type=int, default=0,
                        help="cross-validation folds on the winner (0 = skip)")
    args = parser.parse_args()

    sample = None if args.full else args.sample
    splits = load_splits(sample=sample, seed=args.seed)

    X_train, y_train = splits["train"]["clean_text"], splits["train"]["sentiment"]
    X_val, y_val = splits["val"]["clean_text"], splits["val"]["sentiment"]
    X_test, y_test = splits["test"]["clean_text"], splits["test"]["sentiment"]

    # ---- Phase 1: compare configurations on the VALIDATION set ----------
    print("\n" + "=" * 62)
    print("PHASE 1  comparing configurations on validation")
    print("=" * 62)

    results: dict[str, dict] = {}
    for name, config in CONFIGS.items():
        print(f"\n[{name}]  {config}")
        pipeline = build_pipeline(seed=args.seed, **config)
        pipeline.fit(X_train, y_train)
        results[name] = {
            "config": {k: list(v) if isinstance(v, tuple) else v
                       for k, v in config.items()},
            "val": evaluate(pipeline, X_val, y_val, "val"),
            "pipeline": pipeline,
        }

    print("\n" + "-" * 62)
    print(f"{'config':<14}{'val accuracy':>16}{'val roc-auc':>16}")
    print("-" * 62)
    for name, result in results.items():
        print(f"{name:<14}{result['val']['accuracy']:>16.4f}"
              f"{result['val']['roc_auc']:>16.4f}")

    best_name = max(results, key=lambda n: results[n]["val"]["accuracy"])
    best = results[best_name]
    best_pipeline: Pipeline = best["pipeline"]
    print("-" * 62)
    print(f"winner: {best_name}")

    # ---- Phase 2: cross-validate the winner ----------------------------
    cv_scores = None
    if args.cv:
        print("\n" + "=" * 62)
        print(f"PHASE 2  {args.cv}-fold cross-validation on the winner")
        print("=" * 62)
        # A single val score is one draw from a distribution. CV tells you the
        # spread: a 0.5% gap between two configs is meaningless if the fold
        # standard deviation is 1%.
        cv_scores = cross_val_score(
            build_pipeline(seed=args.seed, **CONFIGS[best_name]),
            X_train, y_train, cv=args.cv, scoring="accuracy", n_jobs=-1,
        )
        print(f"  folds: {', '.join(f'{s:.4f}' for s in cv_scores)}")
        print(f"  mean {cv_scores.mean():.4f}  std {cv_scores.std():.4f}")

    # ---- Phase 3: the ONE look at the test set -------------------------
    print("\n" + "=" * 62)
    print("PHASE 3  final evaluation on the held-out test set")
    print("=" * 62)
    print("  (first and only time this split is used - it is the honest number)")

    test_metrics = evaluate(best_pipeline, X_test, y_test, "test")
    test_predictions = best_pipeline.predict(X_test)

    print("\n" + classification_report(
        y_test, test_predictions,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], digits=4,
    ))

    matrix = confusion_matrix(y_test, test_predictions)
    print("  confusion matrix (rows = actual, cols = predicted)")
    print(f"                predicted_neg  predicted_pos")
    print(f"    actual_neg  {matrix[0][0]:>13,}  {matrix[0][1]:>13,}")
    print(f"    actual_pos  {matrix[1][0]:>13,}  {matrix[1][1]:>13,}")

    top_features = show_top_features(best_pipeline)

    # ---- Phase 4: persist ----------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH, compress=3)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config_name": best_name,
        "config": best["config"],
        "seed": args.seed,
        "rows_used": int(len(splits["train"]) + len(splits["val"]) + len(splits["test"])),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "val_metrics": {n: r["val"] for n, r in results.items()},
        "test_metrics": test_metrics,
        "cv_scores": [float(s) for s in cv_scores] if cv_scores is not None else None,
        "confusion_matrix": matrix.tolist(),
        "top_features": top_features,
        "label_names": LABEL_NAMES,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved pipeline -> {MODEL_PATH.name} ({size_mb:.1f} MB)")
    print(f"Saved metadata -> {META_PATH.name}")
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
