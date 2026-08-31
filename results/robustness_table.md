
Metric: accuracy at threshold 0.5, and AUC. Test set: 20,000 held-out CIFAKE test images, never seen in training.

| Condition | Parameter | baseline acc | baseline AUC | robust acc | robust AUC | Delta acc |
|---|---|---|---|---|---|---|
| clean | - | 96.69 | 99.48 | 95.58 | 99.21 | -1.11 |
| jpeg | q=90 | 96.47 | 99.42 | 95.62 | 99.17 | -0.85 |
| jpeg | q=70 | 96.46 | 99.41 | 95.73 | 99.22 | -0.73 |
| jpeg | q=50 | 94.01 | 98.50 | 94.38 | 98.77 | +0.38 |
| jpeg | q=30 | 87.91 | 97.80 | 92.69 | 98.11 | +4.77 |
| blur | sigma=0.5 | 80.53 | 98.06 | 94.65 | 98.90 | +14.12 |
| blur | sigma=1.0 | 62.11 | 78.12 | 92.59 | 97.97 | +30.47 |
| blur | sigma=2.0 | 61.32 | 74.73 | 89.39 | 96.11 | +28.07 |
| resize | 0.5x | 62.61 | 77.78 | 92.08 | 97.79 | +29.47 |
| resize | 0.25x | 63.06 | 73.83 | 87.94 | 95.16 | +24.89 |
| noise | sigma=0.02 | 95.98 | 99.33 | 95.41 | 99.13 | -0.58 |
| noise | sigma=0.05 | 91.30 | 97.38 | 94.41 | 98.79 | +3.11 |
| noise | sigma=0.10 | 75.56 | 87.47 | 91.42 | 97.42 | +15.85 |
| jitter | +/-20% | 95.52 | 99.08 | 94.45 | 98.74 | -1.07 |
| crop | 80% | 71.74 | 94.91 | 92.05 | 97.77 | +20.31 |

- **baseline**: clean 96.69%, mean over transformed conditions 81.04%, worst 61.32%, mean drop 15.65 pts.
- **robust**: clean 95.58%, mean over transformed conditions 93.06%, worst 87.94%, mean drop 2.52 pts.