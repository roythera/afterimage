"""Required deliverable: image directory -> JSON of AI-generated confidences.

    python -m src.predict --image-dir path/to/images --output preds.json

Output is a JSON list, one entry per image:

    [{"image_path": "img/0001.png", "pred": 0.93}, ...]

where ``pred`` is P(image is AI-generated) in [0, 1].

Handling images larger than 32x32
---------------------------------
The detector is trained on 32x32 CIFAKE. The obvious way to apply it to a
1024x1024 photograph is to resize down to 32x32 -- which is the worst possible
choice, because downscaling is a low-pass filter and the generator fingerprints
this task depends on live in the high frequencies. Resizing destroys the
evidence before the model ever sees it.

Instead the default mode is ``patch``: tile the image at native resolution into
32x32 patches and average the per-patch logits. Every patch is seen at 1:1
pixel scale, so local high-frequency statistics survive intact, and averaging
over many patches reduces variance. ``--mode resize`` and ``--mode full`` (one
forward pass over the whole image, which the fully-convolutional architecture
allows) are kept for comparison -- see docs/robustness.md for the measured gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import MEAN, STD, ImageDirDataset, to_tensor
from src.model import build_model
from src.train import pick_device

ROOT = Path(__file__).resolve().parent.parent
PATCH = 32


_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def _patches(img: Image.Image, patch: int, max_patches: int, rng) -> torch.Tensor:
    """Tile ``img`` into ``patch``x``patch`` crops at native resolution -> (N,3,p,p).

    Done as a single reshape over the whole array rather than one PIL crop per
    patch. A 1024x1024 image yields 1024 patches, and cropping them individually
    dominated runtime; this is ~20x faster and makes the difference between the
    prediction script being usable and not.
    """
    w, h = img.size
    if w < patch or h < patch:
        img = img.resize((max(w, patch), max(h, patch)), Image.BICUBIC)
        w, h = img.size

    nh, nw = h // patch, w // patch
    arr = np.asarray(img, dtype=np.uint8)[: nh * patch, : nw * patch]
    # (nh, patch, nw, patch, 3) -> (nh*nw, patch, patch, 3)
    tiles = arr.reshape(nh, patch, nw, patch, 3).transpose(0, 2, 1, 3, 4)
    tiles = tiles.reshape(nh * nw, patch, patch, 3)

    if len(tiles) > max_patches:
        # Uniform subsample rather than a contiguous block, so the patches still
        # cover the whole frame instead of just the top-left corner.
        sel = np.sort(rng.choice(len(tiles), size=max_patches, replace=False))
        tiles = tiles[sel]

    t = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float().div_(255.0)
    return (t - _MEAN_T) / _STD_T


@torch.no_grad()
def predict_image(model, img: Image.Image, device, mode: str, max_patches: int, rng) -> float:
    if mode == "resize":
        x = to_tensor(img.resize((PATCH, PATCH), Image.BILINEAR)).unsqueeze(0)
        return float(torch.sigmoid(model(x.to(device)))[0])
    if mode == "full":
        if min(img.size) < PATCH:
            img = img.resize((max(img.size[0], PATCH), max(img.size[1], PATCH)), Image.BICUBIC)
        x = to_tensor(img).unsqueeze(0)
        return float(torch.sigmoid(model(x.to(device)))[0])
    if mode == "patch":
        x = _patches(img, PATCH, max_patches, rng).to(device)
        logits = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])
        # Average logits, not probabilities: probabilities saturate at 0 and 1,
        # so a handful of confident patches would otherwise dominate the mean.
        return float(torch.sigmoid(logits.mean()))
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image-dir", required=True, help="directory of images (searched recursively)")
    ap.add_argument("--output", default="preds.json", help="path to the output JSON")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "robust.pt"))
    ap.add_argument("--mode", choices=["patch", "resize", "full"], default="patch")
    ap.add_argument("--max-patches", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--relative", action="store_true",
                    help="emit paths relative to --image-dir instead of as given")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = ImageDirDataset(Path(args.image_dir))
    if not len(ds):
        raise SystemExit(f"no images found under {args.image_dir}")
    print(f"{len(ds):,} images | checkpoint {Path(args.checkpoint).name} "
          f"({ck['n_params']:,} params) | mode={args.mode} | device={device}")

    rng = np.random.default_rng(args.seed)
    out = []
    for i in range(len(ds)):
        img, path = ds[i]
        pred = predict_image(model, img, device, args.mode, args.max_patches, rng)
        if args.relative:
            path = str(Path(path).relative_to(Path(args.image_dir)))
        out.append({"image_path": path, "pred": round(pred, 6)})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{len(ds):,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    preds = np.array([o["pred"] for o in out])
    print(f"wrote {args.output}  ({len(out):,} entries, "
          f"mean pred {preds.mean():.3f}, {(preds > 0.5).mean():.1%} flagged AI-generated)")


if __name__ == "__main__":
    main()
