"""Sentiment140 loading, cleaning and splitting.

This module owns every decision about *data*, so that train.py can stay
focused on *modelling*. Keeping the two apart is what lets you re-train a
different model on identical splits and trust the comparison.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"

# The file ships with no header row. These names come from the dataset's
# own documentation (Go, Bhayani & Huang, 2009).
COLUMNS = ["sentiment", "id", "date", "query", "user", "text"]

# Sentiment140 encodes negative as 0 and positive as 4. We remap 4 -> 1 so the
# labels are the 0/1 that every sklearn metric expects by default.
LABEL_MAP = {0: 0, 4: 1}
LABEL_NAMES = {0: "negative", 1: "positive"}

SEED = 42


def clean_text(text: str) -> str:
    """Normalise one tweet/comment into a bag-of-words-friendly string.

    Deliberately conservative: we strip things that are noise for a
    word-frequency model (URLs, @mentions, punctuation, digits) and keep
    everything else. Note that we do NOT strip stopwords here -- that is a
    *vectoriser* decision, and hard-coding it here would stop us from
    experimenting with it later. Cleaning should be reversible in spirit:
    never throw away a signal the model might want.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs carry no sentiment
    text = re.sub(r"@\w+", " ", text)             # @mentions are just names
    text = re.sub(r"[^a-z\s]", " ", text)         # keep letters only
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


def load_raw(sample: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Read the raw CSV, clean it, and optionally take a random subsample.

    `sample` is the single most useful knob while learning: training on
    200_000 rows takes seconds instead of minutes, and every conclusion you
    draw still holds at full scale. Move to the full 1.6M only once the
    pipeline is settled.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run:  uv run python download_data.py"
        )

    # encoding='latin-1' is required: the file is NOT valid UTF-8 and pandas
    # will raise UnicodeDecodeError on the default encoding.
    df = pd.read_csv(RAW_CSV, encoding="latin-1", names=COLUMNS)

    # ---------------------------------------------------------------
    # TRAP: the raw file is sorted by label -- rows 0..799,999 are all
    # negative and 800,000..1,599,999 are all positive. df.head(n) therefore
    # hands you a 100% negative dataset and a model that has never seen a
    # positive example. Always sample randomly.
    # ---------------------------------------------------------------
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed)

    df = df[["sentiment", "text"]].copy()
    df["sentiment"] = df["sentiment"].map(LABEL_MAP)
    df["clean_text"] = df["text"].apply(clean_text)

    # Cleaning can empty a row out entirely (a comment that was only a URL,
    # or only emoji). An empty string is an all-zero feature vector, which is
    # noise for training -- drop them rather than let the model fit on them.
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped:,} rows that cleaned to an empty string")

    return df


def make_splits(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Split into train / validation / test (70 / 15 / 15), stratified.

    Why three splits and not the two the Kaggle notebook uses:

      train -- the model fits on this
      val   -- you compare configurations on this, as often as you like
      test  -- you touch this ONCE, at the very end, to report a number

    If you tune against the test set (picking bigrams over unigrams because
    test accuracy went up, say) then your reported accuracy is no longer an
    estimate of real-world performance -- you have leaked the answers.
    `stratify` keeps the 50/50 class balance identical in all three.
    """
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["sentiment"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["sentiment"]
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def load_splits(sample: int | None = None, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Convenience wrapper: raw CSV -> cleaned -> three splits."""
    print(f"Loading Sentiment140 (sample={sample or 'all'})...")
    df = load_raw(sample=sample, seed=seed)
    splits = make_splits(df, seed=seed)
    for name, part in splits.items():
        balance = part["sentiment"].value_counts(normalize=True).mul(100).round(1)
        print(
            f"  {name:5} {len(part):>9,} rows   "
            f"neg {balance.get(0, 0)}% / pos {balance.get(1, 0)}%"
        )
    return splits
