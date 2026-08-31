"""Self-test for the transform suite.

Run before trusting any robustness number:

    python -m src.selftest

Checks that each condition (a) actually changes the image, (b) preserves shape
and dynamic range, (c) is reproducible under a fixed seed, and (d) is monotone
in severity where it should be. Also writes a contact sheet to
``results/transform_grid.png`` so the transforms can be eyeballed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data import load_split
from src.transforms import CONDITIONS, RobustAugment

ROOT = Path(__file__).resolve().parent.parent


def _rmse(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, np.float64) / 255.0
    y = np.asarray(b, np.float64) / 255.0
    return float(np.sqrt(((x - y) ** 2).mean()))


def main() -> int:
    imgs, labels = load_split("test")
    fails = []

    base = Image.fromarray(imgs[0])
    print(f"{'condition':<22}{'RMSE vs clean':>14}   checks")
    print("-" * 60)

    rmses = {}
    for cond in CONDITIONS:
        out = cond(base, np.random.default_rng(0))
        out2 = cond(base, np.random.default_rng(0))
        arr = np.asarray(out)

        checks = []
        if out.size != base.size:
            checks.append("SHAPE-CHANGED")
        if out.mode != "RGB":
            checks.append(f"MODE={out.mode}")
        if arr.min() < 0 or arr.max() > 255:
            checks.append("RANGE")
        if _rmse(out, out2) != 0.0:
            checks.append("NON-DETERMINISTIC")
        r = _rmse(out, base)
        rmses[cond.key] = r
        if cond.family != "clean" and r == 0.0:
            checks.append("NO-OP")
        if cond.family == "clean" and r != 0.0:
            checks.append("CLEAN-MODIFIED")

        if checks:
            fails.append((cond.key, checks))
        print(f"{cond.key:<22}{r:>14.5f}   {'ok' if not checks else ' '.join(checks)}")

    # severity should be monotone within a family
    print()
    for fam, keys in [
        ("jpeg", ["jpeg_q=90", "jpeg_q=70", "jpeg_q=50", "jpeg_q=30"]),
        ("blur", ["blur_sigma=0.5", "blur_sigma=1.0", "blur_sigma=2.0"]),
        ("resize", ["resize_0.5x", "resize_0.25x"]),
        ("noise", ["noise_sigma=0.02", "noise_sigma=0.05", "noise_sigma=0.10"]),
    ]:
        vals = [rmses[k] for k in keys]
        ok = all(a < b for a, b in zip(vals, vals[1:]))
        print(f"monotone severity {fam:<8} {'ok' if ok else 'FAIL'}  {[round(v, 4) for v in vals]}")
        if not ok:
            fails.append((fam, ["NON-MONOTONE"]))

    # JPEG must be a real codec round trip, not a smoothing approximation: a
    # second encode at the same quality should be close to idempotent.
    q50 = next(c for c in CONDITIONS if c.key == "jpeg_q=50")
    once = q50(base, np.random.default_rng(0))
    twice = q50(once, np.random.default_rng(0))
    print(f"jpeg idempotence   {_rmse(once, twice):.5f}  (should be << {rmses['jpeg_q=50']:.5f})")
    if _rmse(once, twice) >= rmses["jpeg_q=50"]:
        fails.append(("jpeg", ["NOT-A-CODEC"]))

    # training augmentation should fire and vary
    aug = RobustAugment()
    outs = [_rmse(aug(base, np.random.default_rng(s)), base) for s in range(200)]
    fired = sum(r > 0 for r in outs) / len(outs)
    print(f"RobustAugment      fires {fired:.0%} of the time, {len(set(np.round(outs, 6)))} distinct outputs")
    if not (0.75 < fired < 1.0):
        fails.append(("augment", [f"FIRE-RATE={fired}"]))

    # contact sheet
    idx = [int(np.where(labels == 1)[0][0]), int(np.where(labels == 0)[0][0])]
    cell, pad = 64, 4
    sheet = Image.new("RGB", (len(CONDITIONS) * (cell + pad), len(idx) * (cell + pad)), "white")
    for r, i in enumerate(idx):
        src = Image.fromarray(imgs[i])
        for c, cond in enumerate(CONDITIONS):
            out = cond(src, np.random.default_rng(0)).resize((cell, cell), Image.NEAREST)
            sheet.paste(out, (c * (cell + pad), r * (cell + pad)))
    (ROOT / "results").mkdir(exist_ok=True)
    sheet.save(ROOT / "results" / "transform_grid.png")
    print(f"\nwrote results/transform_grid.png  (row 0 = AI-generated, row 1 = real)")
    print("columns:", ", ".join(c.key for c in CONDITIONS))

    if fails:
        print("\nFAILURES:", fails)
        return 1
    print("\nall transform checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
