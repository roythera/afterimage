"""Robustness evaluation at real resolution, on a different dataset.

    python -m src.eval_fullres --checkpoints baseline robust

This is the credibility check on the CIFAKE result. It differs from
``eval_robustness.py`` in three ways that all make it harder:

* **Resolution.** ~1024px images, so "resize 0.25x" is a 256px thumbnail and an
  80% centre crop is 800px -- the transforms mean what the problem statement
  says they mean, rather than degenerating as they do at 32x32.
* **Distribution.** SID_Set validation: real images from OpenImages, synthetic
  images from generators that are not Stable Diffusion 1.4. Neither the source
  nor the generator appears anywhere in training.
* **Inference path.** The 32x32-trained model is applied by tiling at native
  resolution (see predict.py). ``--modes patch resize`` measures what that
  choice is worth.

Nothing here is trained on. It is a held-out, out-of-distribution test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.eval_robustness import load_checkpoint
from src.predict import predict_image
from src.train import pick_device
from src.transforms import CONDITIONS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "fullres" / "manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["baseline", "robust"])
    ap.add_argument("--modes", nargs="+", default=["patch", "resize"])
    ap.add_argument("--max-patches", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit("run `python -m src.fetch_fullres` first")
    manifest = json.load(open(MANIFEST))
    if args.limit:
        manifest = manifest[: args.limit]
    labels = np.array([m["label"] for m in manifest])
    device = pick_device(args.device)
    sides = sorted(min(m["width"], m["height"]) for m in manifest)
    print(f"{len(manifest)} full-resolution images, {labels.mean():.1%} AI-generated, "
          f"median min-side {sides[len(sides) // 2]}px, device={device}")

    # Decode once; the transform suite is applied to these in memory.
    images = [Image.open(ROOT / m["path"]).convert("RGB") for m in manifest]

    results = {}
    for tag in args.checkpoints:
        model, ck = load_checkpoint(tag, device)
        model.eval()
        for mode in args.modes:
            key = f"{tag}|{mode}"
            print(f"\n=== {tag}, inference mode={mode} ===", flush=True)
            per_cond = {}
            for cond in CONDITIONS:
                rng = np.random.default_rng(1234)
                p = np.array([
                    predict_image(model, cond(im, np.random.default_rng(1234 + i)),
                                  device, mode, args.max_patches, rng)
                    for i, im in enumerate(images)
                ])
                acc = float(((p > 0.5).astype(int) == labels).mean())
                auc = float(roc_auc_score(labels, p))
                per_cond[cond.key] = {"family": cond.family, "param": cond.param,
                                      "acc": acc, "auc": auc,
                                      "mean_score": float(p.mean())}
                print(f"  {cond.key:<20} acc {acc:.4f}  auc {auc:.4f}", flush=True)
            degraded = [v["auc"] for k, v in per_cond.items() if k != "clean"]
            results[key] = {"conditions": per_cond,
                            "clean_auc": per_cond["clean"]["auc"],
                            "clean_acc": per_cond["clean"]["acc"],
                            "mean_transformed_auc": float(np.mean(degraded)),
                            "worst_transformed_auc": float(np.min(degraded))}
            print(f"  -> clean AUC {per_cond['clean']['auc']:.4f} | "
                  f"mean transformed AUC {np.mean(degraded):.4f} | "
                  f"worst {np.min(degraded):.4f}", flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "fullres.json", "w") as f:
        json.dump({"n": len(manifest), "results": results}, f, indent=2)

    keys = list(results)
    lines = ["", f"Full-resolution out-of-distribution slice: {len(manifest)} SID_Set "
                 f"validation images (median min-side {sides[len(sides) // 2]}px). "
                 "AUC, since class balance and calibration both shift off CIFAKE.", "",
             "| Condition | Parameter | " + " | ".join(keys) + " |",
             "|---|---|" + "---|" * len(keys)]
    for cond in CONDITIONS:
        cells = [f"{results[k]['conditions'][cond.key]['auc'] * 100:.2f}" for k in keys]
        lines.append(f"| {cond.family} | {cond.param} | " + " | ".join(cells) + " |")
    lines.append("")
    for k in keys:
        r = results[k]
        lines.append(f"- **{k}**: clean AUC {r['clean_auc'] * 100:.2f}, "
                     f"mean transformed {r['mean_transformed_auc'] * 100:.2f}, "
                     f"worst {r['worst_transformed_auc'] * 100:.2f}.")
    table = "\n".join(lines)
    print(table)
    with open(out / "fullres_table.md", "w") as f:
        f.write(table)
    print("\nwrote results/fullres.json and results/fullres_table.md")


if __name__ == "__main__":
    main()
