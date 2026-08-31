
Full-resolution out-of-distribution slice: 600 SID_Set validation images (median min-side 1024px). AUC, since class balance and calibration both shift off CIFAKE.

| Condition | Parameter | baseline|patch | baseline|resize | robust|patch | robust|resize |
|---|---|---|---|---|---|
| clean | - | 77.17 | 65.06 | 74.40 | 63.10 |
| jpeg | q=90 | 77.15 | 65.14 | 74.51 | 63.06 |
| jpeg | q=70 | 78.52 | 65.28 | 75.62 | 63.31 |
| jpeg | q=50 | 78.58 | 64.89 | 74.85 | 62.92 |
| jpeg | q=30 | 73.08 | 65.29 | 68.19 | 63.55 |
| blur | sigma=0.5 | 73.93 | 65.09 | 72.48 | 63.12 |
| blur | sigma=1.0 | 61.96 | 65.22 | 67.98 | 63.15 |
| blur | sigma=2.0 | 47.31 | 65.71 | 59.73 | 63.34 |
| resize | 0.5x | 62.54 | 65.19 | 67.85 | 63.13 |
| resize | 0.25x | 48.63 | 65.55 | 59.10 | 63.30 |
| noise | sigma=0.02 | 62.89 | 64.95 | 66.92 | 62.97 |
| noise | sigma=0.05 | 46.22 | 64.96 | 59.69 | 62.83 |
| noise | sigma=0.10 | 44.20 | 65.54 | 55.63 | 63.28 |
| jitter | +/-20% | 73.84 | 64.28 | 70.40 | 62.35 |
| crop | 80% | 70.02 | 59.47 | 72.56 | 61.82 |

- **baseline|patch**: clean AUC 77.17, mean transformed 64.20, worst 44.20.
- **baseline|resize**: clean AUC 65.06, mean transformed 64.75, worst 59.47.
- **robust|patch**: clean AUC 74.40, mean transformed 67.54, worst 55.63.
- **robust|resize**: clean AUC 63.10, mean transformed 63.01, worst 61.82.