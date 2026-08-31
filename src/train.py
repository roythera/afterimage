"""Train the detector.

The project rests on comparing two runs that differ in exactly one variable:

    python -m src.train --aug none   --tag baseline   # clean training data
    python -m src.train --aug robust --tag robust     # + transform augmentation

Everything else -- architecture, seed, schedule, split -- is held fixed, so the
difference between their robustness profiles is attributable to the augmentation
and nothing else.

The 20k official CIFAKE test split is never touched here. Model selection uses a
5k validation slice held out of the 100k training split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import CifakeDataset, load_split
from src.model import build_model, count_params
from src.transforms import RobustAugment

ROOT = Path(__file__).resolve().parent.parent
VAL_SIZE = 5_000


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_train_val(n: int, seed: int = 0):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:-VAL_SIZE], perm[-VAL_SIZE:]


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """-> (accuracy, mean BCE loss) at threshold 0.5."""
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).float()
        logit = model(x)
        loss_sum += crit(logit, y).item()
        correct += ((logit > 0).float() == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", choices=["none", "robust"], default="none",
                    help="'none' = clean training (the baseline); 'robust' = transform augmentation")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="transform families held out of augmentation, e.g. --exclude noise")
    ap.add_argument("--tag", default=None, help="checkpoint name; defaults to --aug")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="subsample training set, for smoke tests")
    args = ap.parse_args()

    tag = args.tag or args.aug
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    imgs, labels = load_split("train")
    tr_idx, va_idx = split_train_val(len(labels), seed=args.seed)
    if args.limit:
        tr_idx = tr_idx[: args.limit]

    post = RobustAugment(exclude=tuple(args.exclude)) if args.aug == "robust" else None
    train_ds = CifakeDataset(imgs[tr_idx], labels[tr_idx], post=post,
                             seed=args.seed + 1, train_flip=True)
    # Validation is always clean: it measures whether the model is learning, and
    # keeping it fixed makes the two runs' curves directly comparable.
    val_ds = CifakeDataset(imgs[va_idx], labels[va_idx], post=None)

    pin = device.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    model = build_model().to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_ld), pct_start=0.25
    )
    crit = nn.BCEWithLogitsLoss()

    print(f"tag={tag}  aug={args.aug}  exclude={args.exclude or '-'}  device={device}")
    print(f"train={len(train_ds):,}  val={len(val_ds):,}  params={n_params:,} "
          f"({n_params / 1e9:.6f}B, limit 2B)")

    history, best_acc = [], 0.0
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    t_start = time.time()

    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)  # fresh augmentation draw each epoch
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * y.numel()
            seen += y.numel()
        val_acc, val_loss = evaluate(model, val_ld, device)
        history.append({"epoch": epoch, "train_loss": run_loss / seen,
                        "val_acc": val_acc, "val_loss": val_loss,
                        "secs": time.time() - t0})
        star = ""
        if val_acc > best_acc:
            best_acc = val_acc
            star = " *"
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "tag": tag, "n_params": n_params, "val_acc": val_acc,
                        "epoch": epoch}, ckpt_dir / f"{tag}.pt")
        print(f"  epoch {epoch:2d}  train_loss {run_loss / seen:.4f}  "
              f"val_acc {val_acc:.4f}  val_loss {val_loss:.4f}  "
              f"{history[-1]['secs']:.0f}s{star}", flush=True)

    total = time.time() - t_start
    print(f"done in {total / 60:.1f} min  best clean val_acc {best_acc:.4f}  "
          f"-> checkpoints/{tag}.pt")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"train_{tag}.json", "w") as f:
        json.dump({"tag": tag, "args": vars(args), "n_params": n_params,
                   "best_val_acc": best_acc, "total_secs": total,
                   "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
