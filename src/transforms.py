"""Post-processing transform suite.

This module is the measuring instrument for the whole project, so it is written
to be exact rather than convenient:

* JPEG is a real encode/decode round trip through a PIL buffer. A blur
  approximation would defeat the purpose -- the compression artefacts are what
  destroy the high-frequency generator fingerprints that AIGC detectors key on.
* Gaussian noise sigma is in normalised [0, 1] units, applied to the float image
  before normalisation, and clamped back to [0, 1].
* Resize downscales and then upscales *back to the original size*, so the tensor
  shape is unchanged and only information loss is being measured.
* Centre crop likewise resizes back to the original size after cropping.

Every transform has the signature ``fn(img: PIL.Image, rng) -> PIL.Image`` and
operates on RGB images. Stochastic transforms (noise, jitter) draw from the
supplied ``numpy.random.Generator`` so evaluation is exactly reproducible.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _to_float(img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 array in [0, 1], shape (H, W, 3)."""
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_pil(arr: np.ndarray) -> Image.Image:
    """float32 array in [0, 1] -> PIL RGB, with an explicit clamp."""
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="RGB")


# --------------------------------------------------------------------------- #
# the six transform families
# --------------------------------------------------------------------------- #


def jpeg(img: Image.Image, rng=None, *, quality: int) -> Image.Image:
    """Real JPEG encode/decode round trip."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img: Image.Image, rng=None, *, sigma: float) -> Image.Image:
    """Gaussian blur with an explicit sigma.

    torchvision is used rather than ``PIL.ImageFilter.GaussianBlur`` because the
    latter approximates a Gaussian with successive box blurs, so its ``radius``
    is only loosely a standard deviation. Kernel size is the usual 3-sigma
    support, rounded up to the next odd integer.
    """
    k = 2 * int(math.ceil(3.0 * sigma)) + 1
    t = TF.to_tensor(img)
    t = TF.gaussian_blur(t, kernel_size=[k, k], sigma=[sigma, sigma])
    return _to_pil(t.permute(1, 2, 0).numpy())


def resize(img: Image.Image, rng=None, *, scale: float) -> Image.Image:
    """Downscale by ``scale`` then upscale back to the original size.

    Downsampling uses BILINEAR, which PIL applies with a support-scaled
    (anti-aliased) filter, and upsampling uses BICUBIC -- the combination a
    thumbnail pipeline followed by a viewer actually performs.
    """
    w, h = img.size
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


def noise(img: Image.Image, rng: np.random.Generator, *, sigma: float) -> Image.Image:
    """Additive Gaussian noise, sigma in normalised [0, 1] units."""
    arr = _to_float(img)
    arr = arr + rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return _to_pil(arr)  # _to_pil clamps


def jitter(
    img: Image.Image,
    rng: np.random.Generator,
    *,
    strength: float = 0.2,
) -> Image.Image:
    """Brightness / contrast / saturation each scaled by a factor in 1 +/- strength.

    Factors are drawn independently and uniformly, and applied in a shuffled
    order because these operations do not commute.
    """
    lo, hi = 1.0 - strength, 1.0 + strength
    b, c, s = rng.uniform(lo, hi, size=3)
    ops = [
        lambda im: TF.adjust_brightness(im, float(b)),
        lambda im: TF.adjust_contrast(im, float(c)),
        lambda im: TF.adjust_saturation(im, float(s)),
    ]
    for i in rng.permutation(3):
        img = ops[i](img)
    return img


def centre_crop(img: Image.Image, rng=None, *, frac: float = 0.8) -> Image.Image:
    """Centre crop to ``frac`` of each side, then resize back to original size."""
    w, h = img.size
    cw, ch = max(1, int(round(w * frac))), max(1, int(round(h * frac)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)


# --------------------------------------------------------------------------- #
# the evaluation grid
# --------------------------------------------------------------------------- #


def identity(img: Image.Image, rng=None) -> Image.Image:
    return img


@dataclass(frozen=True)
class Condition:
    """One row of the robustness table.

    ``fn`` is a module-level function and ``kwargs`` a plain dict rather than a
    bound closure, because these objects are pickled to DataLoader workers and
    macOS spawns rather than forks them -- a lambda here fails at spawn time.
    """

    family: str        # "jpeg", "blur", ...
    param: str         # human-readable severity, for the results table
    fn: Callable[..., Image.Image]
    kwargs: dict

    @property
    def key(self) -> str:
        return "clean" if self.family == "clean" else f"{self.family}_{self.param}"

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        return self.fn(img, rng, **self.kwargs)


def _c(family, param, fn, **kw) -> Condition:
    return Condition(family, param, fn, kw)


#: Every condition in the problem statement, each severity kept separate. An
#: averaged "robust accuracy" would hide the structure that makes the result
#: interesting, so nothing here is collapsed.
CONDITIONS: list[Condition] = [
    _c("clean", "-", identity),
    _c("jpeg", "q=90", jpeg, quality=90),
    _c("jpeg", "q=70", jpeg, quality=70),
    _c("jpeg", "q=50", jpeg, quality=50),
    _c("jpeg", "q=30", jpeg, quality=30),
    _c("blur", "sigma=0.5", blur, sigma=0.5),
    _c("blur", "sigma=1.0", blur, sigma=1.0),
    _c("blur", "sigma=2.0", blur, sigma=2.0),
    _c("resize", "0.5x", resize, scale=0.5),
    _c("resize", "0.25x", resize, scale=0.25),
    _c("noise", "sigma=0.02", noise, sigma=0.02),
    _c("noise", "sigma=0.05", noise, sigma=0.05),
    _c("noise", "sigma=0.10", noise, sigma=0.10),
    _c("jitter", "+/-20%", jitter, strength=0.2),
    _c("crop", "80%", centre_crop, frac=0.8),
]

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


# --------------------------------------------------------------------------- #
# training-time augmentation
# --------------------------------------------------------------------------- #

#: Severity ranges for training augmentation. These are *continuous intervals
#: spanning* the discrete evaluation severities rather than the evaluation
#: severities themselves. Training on the exact test settings would inflate the
#: robustness table without the model having learned anything general; sampling
#: the interval means every evaluation point is an interpolation, and severities
#: beyond the range (see --eval-ood) remain a genuine extrapolation test.
AUG_RANGES = {
    "jpeg": (30, 95),        # quality
    "blur": (0.3, 2.0),      # sigma
    "resize": (0.25, 0.9),   # scale
    "noise": (0.0, 0.10),    # sigma, normalised units
    "jitter": (0.0, 0.20),   # strength
    "crop": (0.8, 1.0),      # kept fraction
}


def _sample_one(family: str, rng: np.random.Generator, img: Image.Image) -> Image.Image:
    lo, hi = AUG_RANGES[family]
    if family == "jpeg":
        return jpeg(img, rng, quality=int(rng.integers(lo, hi + 1)))
    if family == "blur":
        return blur(img, rng, sigma=float(rng.uniform(lo, hi)))
    if family == "resize":
        return resize(img, rng, scale=float(rng.uniform(lo, hi)))
    if family == "noise":
        return noise(img, rng, sigma=float(rng.uniform(lo, hi)))
    if family == "jitter":
        return jitter(img, rng, strength=float(rng.uniform(lo, hi)))
    if family == "crop":
        return centre_crop(img, rng, frac=float(rng.uniform(lo, hi)))
    raise ValueError(family)


class RobustAugment:
    """Randomly applies a short chain of post-processing transforms.

    Chaining (up to ``max_ops``) matters: in the wild an image is screenshotted,
    resized and re-encoded, and the composition is harsher than any single step.
    A model trained only on single transforms is measurably weaker on composites.

    ``exclude`` holds out a transform family from training so that its row in the
    robustness table becomes a test of generalisation to an *unseen* corruption
    rather than a test of memorisation.
    """

    def __init__(self, p: float = 0.9, max_ops: int = 2, exclude: tuple[str, ...] = ()):
        self.p = p
        self.max_ops = max_ops
        self.families = [f for f in AUG_RANGES if f not in exclude]
        self.exclude = tuple(exclude)

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        if rng.random() > self.p:
            return img
        n = int(rng.integers(1, self.max_ops + 1))
        picks = rng.choice(len(self.families), size=n, replace=False)
        for i in picks:
            img = _sample_one(self.families[i], rng, img)
        return img
