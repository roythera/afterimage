"""CIFAKE loading and the two dataset views (clean / augmented).

CIFAKE is 60k real (CIFAR-10) + 60k fake (Stable Diffusion v1.4) at 32x32,
already split 100k train / 20k test by the authors. The whole thing is ~50MB of
parquet and decodes to ~370MB of uint8, so it is held in memory as a single
array; that removes the dataloader from the critical path and lets the transform
suite (which is CPU-bound PIL work) use all the workers.

Label convention throughout the project: 1 = AI-generated, 0 = real. The
prediction script emits P(AI-generated), so this ordering must not be flipped.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cifake"

# CIFAKE's own label column is 0 = FAKE, 1 = REAL (folder order). We invert it so
# that 1 = AI-generated, matching the "pred is a confidence that the image is
# AI-generated" requirement in the problem statement.
_CIFAKE_LABEL_IS_REAL = 1


def load_split(split: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [N,32,32,3], labels int64 [N]) with 1 = AI-generated."""
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    imgs = np.stack(
        [
            np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
            for rec in df["image"]
        ]
    )
    labels = (df["label"].to_numpy() != _CIFAKE_LABEL_IS_REAL).astype(np.int64)
    return imgs, labels


# Channel statistics of the CIFAKE training split (float64 accumulation -- a
# float32 sum over 10^8 pixels silently saturates and returns nonsense). Hard
# coded so evaluation and the prediction script never depend on the training
# data being present. Reproduce with: python -m src.data --stats
MEAN = (0.4720, 0.4629, 0.4178)
STD = (0.2376, 0.2374, 0.2660)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> normalised CHW float tensor."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)


class CifakeDataset(Dataset):
    """CIFAKE with an optional post-processing transform applied per item.

    ``post`` is a callable ``(PIL.Image, np.random.Generator) -> PIL.Image`` --
    either a single :class:`~src.transforms.Condition` (evaluation) or a
    :class:`~src.transforms.RobustAugment` (training).

    Randomness is seeded per index. During evaluation ``epoch`` is fixed, so a
    stochastic condition such as noise or jitter draws the *same* perturbation
    for a given image on every run and for every model -- baseline and robust are
    compared on byte-identical inputs. During training ``set_epoch`` advances the
    seed so each epoch sees fresh augmentation.
    """

    def __init__(self, images, labels, post=None, seed: int = 0, train_flip: bool = False):
        self.images = images
        self.labels = labels
        self.post = post
        self.seed = seed
        self.epoch = 0
        self.train_flip = train_flip

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = Image.fromarray(self.images[i])
        if self.post is not None or self.train_flip:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            if self.train_flip and rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.post is not None:
                img = self.post(img, rng)
        return to_tensor(img), int(self.labels[i])


class ImageDirDataset(Dataset):
    """Flat list of image files, for the prediction script.

    Images are returned at their native size, so batching is left to the caller
    (predict.py runs the patch-based path one image at a time).
    """

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, root: Path):
        root = Path(root)
        self.paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in self.EXTS and p.is_file()
        )
        self.root = root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        return Image.open(p).convert("RGB"), str(p)


if __name__ == "__main__":
    import sys

    if "--stats" in sys.argv:
        imgs, labels = load_split("train")
        n = imgs.shape[0] * imgs.shape[1] * imgs.shape[2]
        mean = imgs.sum(axis=(0, 1, 2), dtype=np.float64) / n / 255.0
        sq = np.square(imgs.astype(np.float64)).sum(axis=(0, 1, 2)) / n / 255.0**2
        print("N =", len(labels), " frac AI-generated =", float(labels.mean()))
        print("MEAN =", tuple(round(float(x), 4) for x in mean))
        print("STD  =", tuple(round(float(x), 4) for x in np.sqrt(sq - mean**2)))
