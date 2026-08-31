"""Fetch a small full-resolution evaluation slice from SID_Set.

    python -m src.fetch_fullres --per-class 300

Why this exists: CIFAKE is 32x32, and at that size several conditions in the
transform suite are close to degenerate -- "resize to 0.25x" means an 8x8
thumbnail, and an 80% centre crop is 25 pixels wide. A robustness claim measured
only there would not be credible at the resolution real platform images have.
This pulls a few hundred genuinely full-resolution images (~1024px) so the claim
can be checked where the transforms mean what they say.

Images come from the SID_Set *validation* split (label 0 = real / OpenImages,
label 1 = fully synthetic). Nothing here is used for training -- it is a
held-out, out-of-distribution test set from a different source, different
generators and a different resolution than CIFAKE.

Note: the datasets-server serves JPEG-transcoded copies rather than the original
files, so these images have been through one JPEG encode before we see them.
That is stated in the README rather than glossed over; it makes the JPEG row of
the full-resolution table a second-generation re-encode, which is if anything
the more realistic scenario.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fullres"
API = "https://datasets-server.huggingface.co/rows"
DATASET = "saberzl/SID_Set"

# SID_Set label -> our binary convention (1 = AI-generated)
KEEP = {0: 0, 1: 1}  # label 2 ("tampered") is a different task and is skipped


def fetch_rows(offset: int, length: int, session: requests.Session) -> list[dict]:
    r = session.get(API, params={"dataset": DATASET, "config": "default",
                                 "split": "validation", "offset": offset,
                                 "length": length}, timeout=120)
    r.raise_for_status()
    return [row["row"] for row in r.json()["rows"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--min-side", type=int, default=256,
                    help="skip images smaller than this; the point is full resolution")
    args = ap.parse_args()

    for lab in ("real", "ai"):
        (OUT / lab).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    counts = {0: 0, 1: 0}
    manifest, offset, pages = [], 0, 0

    while min(counts.values()) < args.per_class and pages < args.max_pages:
        try:
            rows = fetch_rows(offset, args.page, session)
        except Exception as e:  # transient 5xx from the datasets-server
            print(f"  page at offset {offset} failed ({e}); retrying once")
            time.sleep(5)
            rows = fetch_rows(offset, args.page, session)
        if not rows:
            break
        offset += len(rows)
        pages += 1

        for row in rows:
            lab = row["label"]
            if lab not in KEEP or counts[lab] >= args.per_class:
                continue
            if min(row["width"], row["height"]) < args.min_side:
                continue
            try:
                blob = session.get(row["image"]["src"], timeout=120).content
                img = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception as e:
                print(f"  skip {row['img_id']}: {e}")
                continue
            y = KEEP[lab]
            name = f"{row['img_id']}.png".replace("/", "_")
            path = OUT / ("ai" if y else "real") / name
            img.save(path)  # PNG: no further compression before the transform suite
            manifest.append({"path": str(path.relative_to(ROOT)), "label": y,
                             "img_id": row["img_id"], "width": img.width,
                             "height": img.height})
            counts[lab] += 1
        print(f"  offset {offset}: real={counts[0]} ai={counts[1]}")

    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    sides = [min(m["width"], m["height"]) for m in manifest]
    print(f"\nwrote {len(manifest)} images to {OUT}")
    print(f"  real={counts[0]}  ai={counts[1]}  "
          f"min side: min={min(sides)} median={sorted(sides)[len(sides) // 2]} max={max(sides)}")


if __name__ == "__main__":
    main()
