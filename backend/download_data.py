"""One-off: fetch Sentiment140 from its original Stanford host.

No Kaggle account or API token needed -- this is the canonical release from
the authors of the 2009 paper. ~81 MB zipped, ~230 MB extracted.
"""
from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import requests

URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "sentiment140.zip"
TRAIN_CSV = DATA_DIR / "training.1600000.processed.noemoticon.csv"


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_CSV.exists():
        print(f"Already present: {TRAIN_CSV.name} "
              f"({TRAIN_CSV.stat().st_size / 1e6:.0f} MB) - nothing to do.")
        return

    if not ZIP_PATH.exists():
        print(f"Downloading {URL} ...")
        # stream=True keeps the 81 MB out of memory; we write as it arrives.
        with requests.get(URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            written = 0
            with open(ZIP_PATH, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = written / total * 100
                        print(f"\r  {written/1e6:6.1f} / {total/1e6:.1f} MB "
                              f"({pct:5.1f}%)", end="", flush=True)
            print()

    print(f"Extracting {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for info in archive.infolist():
            print(f"  {info.filename:<55} {info.file_size/1e6:7.1f} MB")
        archive.extractall(DATA_DIR)

    if not TRAIN_CSV.exists():
        sys.exit(f"Extraction finished but {TRAIN_CSV.name} is missing.")
    print(f"\nReady: {TRAIN_CSV}")


if __name__ == "__main__":
    download()
