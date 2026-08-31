"""Error analysis: where the detector still fails, and why.

    python -m src.eval_robustness --save-scores      # produces results/scores.npz
    python -m src.error_analysis

Five questions, each answered with a number rather than an assertion:

1. Which *direction* does each transform push the scores? An AIGC detector keyed
   on high-frequency generator fingerprints should, when those frequencies are
   destroyed, call things *real*. If the errors are asymmetric that way, it is
   evidence about the mechanism, not just the magnitude, of the failure.
2. How much of the accuracy loss is discriminative loss and how much is mere
   miscalibration? Re-thresholding each condition optimally separates the two:
   whatever accuracy comes back for free was a threshold problem, not a
   signal problem.
3. What do the residual false positives and false negatives look like? Contact
   sheets plus image statistics.
4. Is there a systematic property (brightness, saturation, high-frequency
   energy) that distinguishes errors from correct predictions?
5. What did augmentation cost on clean data, and what is the false-positive rate
   at a threshold a moderation system could actually deploy?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.data import load_split
from src.transforms import CONDITIONS

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def best_threshold(labels: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Accuracy-maximising threshold, found exactly over candidate splits."""
    order = np.argsort(p)
    s, y = p[order], labels[order]
    # accuracy if we threshold just above s[i]: predict 1 for indices > i
    tp = np.concatenate([[y.sum()], y.sum() - np.cumsum(y)])
    tn = np.concatenate([[0], np.cumsum(1 - y)])
    acc = (tp + tn) / len(y)
    i = int(np.argmax(acc))
    thr = 0.0 if i == 0 else float(s[i - 1] + 1e-12)
    return thr, float(acc[i])


def highfreq_energy(img: np.ndarray) -> float:
    """Mean absolute Laplacian, a cheap proxy for high-frequency content."""
    g = img.astype(np.float64).mean(axis=2) / 255.0
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(np.abs(lap).mean())


def saturation(img: np.ndarray) -> float:
    a = img.astype(np.float64) / 255.0
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-8), 0).mean())


def contact_sheet(imgs: np.ndarray, idx: np.ndarray, path: Path, cols: int = 12,
                  cell: int = 64) -> None:
    idx = idx[: cols * 3]
    rows = max(1, int(np.ceil(len(idx) / cols)))
    sheet = Image.new("RGB", (cols * (cell + 4), rows * (cell + 4)), "white")
    for k, i in enumerate(idx):
        im = Image.fromarray(imgs[i]).resize((cell, cell), Image.NEAREST)
        sheet.paste(im, ((k % cols) * (cell + 4), (k // cols) * (cell + 4)))
    sheet.save(path)


def main() -> None:
    z = np.load(RES / "scores.npz")
    labels = z["labels"]
    imgs, _ = load_split("test")
    imgs = imgs[: len(labels)]
    tags = sorted({k.split("|")[0] for k in z.files if "|" in k})
    report: dict = {"tags": tags, "n": int(len(labels))}
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("# Error analysis\n")
    say(f"Test set: {len(labels):,} held-out CIFAKE images "
        f"({int(labels.sum()):,} AI-generated, {int((1 - labels).sum()):,} real).")
    say(f"Models compared: {', '.join(tags)}.\n")

    # ---------------------------------------------------------------- Q1 + Q2
    say("## 1. Direction of failure, and how much of it is calibration\n")
    say("`mean score` is the mean P(AI-generated) the model assigns. A transform that "
        "destroys generator fingerprints should drag scores *down*, toward 'real', and "
        "cost recall (TPR) rather than precision. `acc@0.5` is the deployed number; "
        "`acc@best` re-thresholds that condition optimally, so the gap between them is "
        "the part of the loss that is pure miscalibration and is recoverable without "
        "retraining.\n")
    rows = {}
    for tag in tags:
        say(f"### {tag}\n")
        say("| Condition | mean score (real) | mean score (AI) | acc@0.5 | acc@best | "
            "recoverable | AUC |")
        say("|---|---|---|---|---|---|---|")
        for cond in CONDITIONS:
            key = f"{tag}|{cond.key}"
            if key not in z.files:
                continue
            p = z[key]
            a05 = float(((p > 0.5).astype(int) == labels).mean())
            thr, abest = best_threshold(labels, p)
            auc = float(roc_auc_score(labels, p))
            rows[key] = {"acc05": a05, "accbest": abest, "auc": auc, "thr": thr,
                         "mean_real": float(p[labels == 0].mean()),
                         "mean_ai": float(p[labels == 1].mean())}
            label = cond.family if cond.family == "clean" else f"{cond.family} {cond.param}"
            say(f"| {label} | {p[labels == 0].mean():.3f} | {p[labels == 1].mean():.3f} "
                f"| {a05 * 100:.2f} | {abest * 100:.2f} | {(abest - a05) * 100:+.2f} "
                f"| {auc * 100:.2f} |")
        say()
    report["conditions"] = rows

    # ------------------------------------------------------------------- Q3/4
    say("## 2. Residual failures on clean data\n")
    for tag in tags:
        key = f"{tag}|clean"
        if key not in z.files:
            continue
        p = z[key]
        pred = (p > 0.5).astype(int)
        fp = np.where((pred == 1) & (labels == 0))[0]  # real called AI-generated
        fn = np.where((pred == 0) & (labels == 1))[0]  # AI called real
        # most confident mistakes first -- these are the informative ones
        fp = fp[np.argsort(-p[fp])]
        fn = fn[np.argsort(p[fn])]
        contact_sheet(imgs, fp, RES / f"errors_{tag}_false_positive.png")
        contact_sheet(imgs, fn, RES / f"errors_{tag}_false_negative.png")

        stats = {}
        for name, fn_ in [("high-freq energy", highfreq_energy), ("saturation", saturation),
                          ("brightness", lambda a: float(a.mean() / 255.0))]:
            allv = np.array([fn_(imgs[i]) for i in range(0, len(imgs), 4)])
            fpv = np.array([fn_(imgs[i]) for i in fp[:400]]) if len(fp) else np.array([np.nan])
            fnv = np.array([fn_(imgs[i]) for i in fn[:400]]) if len(fn) else np.array([np.nan])
            stats[name] = (float(allv.mean()), float(fpv.mean()), float(fnv.mean()))

        say(f"### {tag}\n")
        say(f"{len(fp)} false positives ({len(fp) / max(1, (labels == 0).sum()) * 100:.2f}% of real "
            f"images), {len(fn)} false negatives "
            f"({len(fn) / max(1, labels.sum()) * 100:.2f}% of AI images).")
        say(f"Contact sheets: `results/errors_{tag}_false_positive.png`, "
            f"`results/errors_{tag}_false_negative.png` (most confident mistakes first).\n")
        say("| Image statistic | all test images | false positives | false negatives |")
        say("|---|---|---|---|")
        for name, (a, f_, n_) in stats.items():
            say(f"| {name} | {a:.4f} | {f_:.4f} | {n_:.4f} |")
        say()
        report.setdefault("clean_errors", {})[tag] = {
            "n_fp": int(len(fp)), "n_fn": int(len(fn)),
            "fp_rate": float(len(fp) / max(1, (labels == 0).sum())),
            "fn_rate": float(len(fn) / max(1, labels.sum())),
            "stats": stats,
        }

    # --------------------------------------------------------------------- Q5
    if len(tags) == 2:
        a, b = tags
        say("## 3. What augmentation cost, and the deployable operating point\n")
        ca, cb = rows[f"{a}|clean"], rows[f"{b}|clean"]
        say(f"Clean accuracy: {a} {ca['acc05'] * 100:.2f}% -> {b} {cb['acc05'] * 100:.2f}% "
            f"({(cb['acc05'] - ca['acc05']) * 100:+.2f} pts). "
            f"Clean AUC: {ca['auc'] * 100:.2f} -> {cb['auc'] * 100:.2f} "
            f"({(cb['auc'] - ca['auc']) * 100:+.2f}).\n")
        say("A moderation system cannot run at a 50% false-positive-tolerant threshold: "
            "wrongly labelling a real photograph as AI-generated is the expensive error. "
            "Recall at fixed low false-positive rates:\n")
        say("| Model | Condition | TPR @ 1% FPR | TPR @ 5% FPR |")
        say("|---|---|---|---|")
        opr = {}
        for tag in tags:
            for ckey in ["clean", "jpeg_q=30", "blur_sigma=2.0", "resize_0.25x"]:
                k = f"{tag}|{ckey}"
                if k not in z.files:
                    continue
                p = z[k]
                neg = np.sort(p[labels == 0])
                res = []
                for fpr in (0.01, 0.05):
                    thr = neg[int(round((1 - fpr) * (len(neg) - 1)))]
                    res.append(float((p[labels == 1] > thr).mean()))
                opr[k] = res
                say(f"| {tag} | {ckey} | {res[0] * 100:.2f} | {res[1] * 100:.2f} |")
        say()
        report["operating_points"] = opr

    with open(RES / "error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(RES / "error_analysis.md", "w") as f:
        f.write("\n".join(out))
    print("\nwrote results/error_analysis.md and results/error_analysis.json")


if __name__ == "__main__":
    main()
