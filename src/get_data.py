"""Fetch CIFAKE into data/cifake/ so the repo reproduces from a clean clone.

    python -m src.get_data

Pulls the two parquet shards (~50MB total) of the HuggingFace mirror of CIFAKE,
which carries the authors' original 100k/20k train/test split. On Kaggle the
dataset can instead be attached from the Add Input panel
(``birdy654/cifake-real-and-ai-generated-synthetic-images``); pass
``--kaggle-dir`` to convert that folder layout into the same parquet files.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cifake"
BASE = "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/data"
FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def download() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split, remote in FILES.items():
        dest = OUT / f"{split}.parquet"
        if dest.exists():
            print(f"{dest} exists, skipping")
            continue
        print(f"downloading {split} ...", flush=True)
        r = requests.get(f"{BASE}/{remote}", timeout=600)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  wrote {dest} ({len(r.content) / 1e6:.1f} MB)")


def from_kaggle(kaggle_dir: Path) -> None:
    """Convert the Kaggle folder layout (train|test / REAL|FAKE) into parquet."""
    import pandas as pd
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = []
        for cls, label in (("FAKE", 0), ("REAL", 1)):  # CIFAKE's own convention
            for p in sorted((kaggle_dir / split / cls).glob("*")):
                buf = io.BytesIO()
                Image.open(p).convert("RGB").save(buf, format="JPEG", quality=95)
                rows.append({"image": {"bytes": buf.getvalue(), "path": p.name},
                             "label": label})
        pd.DataFrame(rows).to_parquet(OUT / f"{split}.parquet")
        print(f"wrote {OUT / f'{split}.parquet'} ({len(rows):,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-dir", default=None,
                    help="path to the attached Kaggle CIFAKE dataset, e.g. "
                         "/kaggle/input/cifake-real-and-ai-generated-synthetic-images")
    args = ap.parse_args()
    if args.kaggle_dir:
        from_kaggle(Path(args.kaggle_dir))
    else:
        download()

    from src.data import load_split
    for split in ("train", "test"):
        imgs, labels = load_split(split)
        print(f"{split}: {imgs.shape} images, {labels.mean():.1%} AI-generated")


if __name__ == "__main__":
    main()
