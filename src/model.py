"""Detector architecture.

Design notes, since the architecture choice is part of the robustness argument:

* **Small ResNet trained from scratch, not an ImageNet backbone.** The inputs are
  32x32. A pretrained backbone expects 224x224 and would spend its first layers
  destroying exactly the high-frequency structure the task depends on, and the
  pretrained features are tuned for semantic content ("is this a dog") rather
  than for provenance ("was this sampled from a diffusion model").

* **No downsampling in the stem.** Standard ResNet opens with a stride-2 7x7
  conv plus a stride-2 max-pool, discarding 3/4 of the spatial signal before the
  first residual block. Generator fingerprints are a high-frequency phenomenon,
  so the stem here is a stride-1 3x3 conv and downsampling is deferred to the
  stage transitions.

* **Fully convolutional with global average pooling.** The classifier therefore
  accepts any input size, which is what makes the patch-based full-resolution
  inference in ``predict.py`` possible without retraining.

* **Single logit.** The output is P(AI-generated) after a sigmoid, which is
  exactly the ``pred`` field the problem statement asks for.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class RobustNet(nn.Module):
    """Compact fully-convolutional residual classifier. ~1.2M parameters."""

    def __init__(self, width: int = 48, blocks: tuple[int, ...] = (2, 2, 2), dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * (2**i)
            for j in range(n):
                layers.append(BasicBlock(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """-> raw logit of shape (N,). Sigmoid gives P(AI-generated)."""
        x = self.stages(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x)).squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(**kw) -> RobustNet:
    return RobustNet(**kw)


if __name__ == "__main__":
    m = build_model()
    n = count_params(m)
    print(f"parameters: {n:,} ({n / 1e9:.6f}B) -- limit is 2B")
    for size in (32, 64, 256):
        out = m(torch.zeros(2, 3, size, size))
        print(f"input {size}x{size} -> logits {tuple(out.shape)}")
