"""Evaluate a checkpoint across the full transform suite.

    python -m src.eval_robustness --checkpoints baseline robust

Every condition is scored on the same held-out 20k CIFAKE test split, and the
stochastic conditions (noise, jitter) are seeded per image index, so all models
see byte-identical inputs. Two metrics are reported per condition:

* **Accuracy** at a fixed threshold of 0.5. Legible, and it is what a deployed
  system with a fixed threshold would actually get.
* **AUC**, which is threshold-free. The distinction matters: a transform can
  leave the ranking almost intact while shifting the score distribution across
  0.5, which collapses accuracy but not AUC. That gap is diagnostic -- it says
  the failure is calibration, not a loss of discriminative signal, and it is
  recoverable by re-thresholding. Reporting only accuracy would hide it.

Writes results/robustness_<tag>.json and prints the markdown table that goes
into docs/robustness.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import CifakeDataset, load_split
from src.model import build_model
from src.train import pick_device
from src.transforms import CONDITIONS

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def score(model, ds, device, batch_size=512, workers=4) -> np.ndarray:
    """-> P(AI-generated) for every item in ``ds``, in order."""
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    out = []
    for x, _ in ld:
        out.append(torch.sigmoid(model(x.to(device))).float().cpu().numpy())
    return np.concatenate(out)


def load_checkpoint(tag: str, device):
    ck = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["baseline", "robust"])
    ap.add_argument("--limit", type=int, default=0, help="subsample the test set")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-scores", action="store_true",
                    help="also dump per-image scores, for the error analysis")
    args = ap.parse_args()

    device = pick_device(args.device)
    imgs, labels = load_split("test")
    if args.limit:
        # The parquet is class-ordered, so a head slice is single-class and AUC
        # is undefined. Subsample stratified and with a fixed seed instead.
        rng = np.random.default_rng(0)
        take = np.concatenate([
            rng.choice(np.where(labels == c)[0], args.limit // 2, replace=False)
            for c in (0, 1)
        ])
        take.sort()
        imgs, labels = imgs[take], labels[take]
    print(f"test set: {len(labels):,} images, {labels.mean():.1%} AI-generated, device={device}")

    results, raw_scores = {}, {}
    for tag in args.checkpoints:
        model, ck = load_checkpoint(tag, device)
        print(f"\n=== {tag} (epoch {ck['epoch']}, clean val_acc {ck['val_acc']:.4f}, "
              f"{ck['n_params']:,} params) ===")
        per_cond = {}
        for cond in CONDITIONS:
            # seed fixed across models and runs -> identical perturbed inputs
            ds = CifakeDataset(imgs, labels, post=cond, seed=1234)
            p = score(model, ds, device, workers=args.workers)
            acc = float(((p > 0.5).astype(int) == labels).mean())
            auc = float(roc_auc_score(labels, p))
            # class-conditional accuracy: which side of the decision the
            # transform pushes the scores toward
            tpr = float(((p > 0.5).astype(int) == 1)[labels == 1].mean())
            tnr = float(((p > 0.5).astype(int) == 0)[labels == 0].mean())
            per_cond[cond.key] = {"family": cond.family, "param": cond.param,
                                  "acc": acc, "auc": auc, "tpr": tpr, "tnr": tnr,
                                  "mean_score": float(p.mean())}
            print(f"  {cond.key:<20} acc {acc:.4f}  auc {auc:.4f}  "
                  f"tpr {tpr:.4f}  tnr {tnr:.4f}")
            if args.save_scores:
                raw_scores.setdefault(tag, {})[cond.key] = p.tolist()
        clean = per_cond["clean"]["acc"]
        degraded = [v["acc"] for k, v in per_cond.items() if k != "clean"]
        results[tag] = {
            "conditions": per_cond,
            "clean_acc": clean,
            "mean_transformed_acc": float(np.mean(degraded)),
            "worst_transformed_acc": float(np.min(degraded)),
            "mean_drop": float(clean - np.mean(degraded)),
            "worst_drop": float(clean - np.min(degraded)),
            "n_params": ck["n_params"],
        }
        print(f"  -> clean {clean:.4f} | mean transformed {results[tag]['mean_transformed_acc']:.4f} "
              f"| worst {results[tag]['worst_transformed_acc']:.4f} "
              f"| mean drop {results[tag]['mean_drop'] * 100:.1f} pts")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "robustness.json", "w") as f:
        json.dump({"n_test": len(labels), "results": results}, f, indent=2)
    if args.save_scores:
        np.savez_compressed(out / "scores.npz", labels=labels,
                            **{f"{t}|{k}": np.array(v)
                               for t, d in raw_scores.items() for k, v in d.items()})
        print(f"\nwrote results/scores.npz")

    print(markdown_table(results, len(labels)))
    with open(out / "robustness_table.md", "w") as f:
        f.write(markdown_table(results, len(labels)))
    print("wrote results/robustness.json and results/robustness_table.md")


def markdown_table(results: dict, n_test: int) -> str:
    tags = list(results)
    lines = [
        "",
        f"Metric: accuracy at threshold 0.5, and AUC. Test set: {n_test:,} held-out "
        "CIFAKE test images, never seen in training.",
        "",
        "| Condition | Parameter | " + " | ".join(f"{t} acc | {t} AUC" for t in tags)
        + (" | Delta acc |" if len(tags) == 2 else " |"),
        "|---|---|" + "---|---|" * len(tags) + ("---|" if len(tags) == 2 else ""),
    ]
    for cond in CONDITIONS:
        cells = []
        for t in tags:
            c = results[t]["conditions"][cond.key]
            cells += [f"{c['acc'] * 100:.2f}", f"{c['auc'] * 100:.2f}"]
        row = f"| {cond.family} | {cond.param} | " + " | ".join(cells)
        if len(tags) == 2:
            d = (results[tags[1]]["conditions"][cond.key]["acc"]
                 - results[tags[0]]["conditions"][cond.key]["acc"]) * 100
            row += f" | {d:+.2f} |"
        else:
            row += " |"
        lines.append(row)
    lines.append("")
    for t in tags:
        r = results[t]
        lines.append(f"- **{t}**: clean {r['clean_acc'] * 100:.2f}%, "
                     f"mean over transformed conditions {r['mean_transformed_acc'] * 100:.2f}%, "
                     f"worst {r['worst_transformed_acc'] * 100:.2f}%, "
                     f"mean drop {r['mean_drop'] * 100:.2f} pts.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
