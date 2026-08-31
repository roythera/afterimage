# Error analysis

Test set: 20,000 held-out CIFAKE images (10,000 AI-generated, 10,000 real).
Models compared: baseline, robust.

## 1. Direction of failure, and how much of it is calibration

`mean score` is the mean P(AI-generated) the model assigns. A transform that destroys generator fingerprints should drag scores *down*, toward 'real', and cost recall (TPR) rather than precision. `acc@0.5` is the deployed number; `acc@best` re-thresholds that condition optimally, so the gap between them is the part of the loss that is pure miscalibration and is recoverable without retraining.

### baseline

| Condition | mean score (real) | mean score (AI) | acc@0.5 | acc@best | recoverable | AUC |
|---|---|---|---|---|---|---|
| clean | 0.035 | 0.963 | 96.69 | 96.76 | +0.07 | 99.48 |
| jpeg q=90 | 0.039 | 0.962 | 96.47 | 96.52 | +0.05 | 99.42 |
| jpeg q=70 | 0.049 | 0.970 | 96.46 | 96.54 | +0.08 | 99.41 |
| jpeg q=50 | 0.066 | 0.931 | 94.01 | 94.08 | +0.08 | 98.50 |
| jpeg q=30 | 0.019 | 0.771 | 87.91 | 92.55 | +4.63 | 97.80 |
| blur sigma=0.5 | 0.003 | 0.609 | 80.53 | 92.97 | +12.44 | 98.06 |
| blur sigma=1.0 | 0.036 | 0.280 | 62.11 | 70.38 | +8.27 | 78.12 |
| blur sigma=2.0 | 0.657 | 0.886 | 61.32 | 68.13 | +6.81 | 74.73 |
| resize 0.5x | 0.039 | 0.290 | 62.61 | 70.17 | +7.56 | 77.78 |
| resize 0.25x | 0.590 | 0.843 | 63.06 | 67.40 | +4.34 | 73.83 |
| noise sigma=0.02 | 0.034 | 0.945 | 95.98 | 96.25 | +0.27 | 99.33 |
| noise sigma=0.05 | 0.096 | 0.894 | 91.30 | 91.37 | +0.07 | 97.38 |
| noise sigma=0.10 | 0.417 | 0.891 | 75.56 | 78.72 | +3.15 | 87.47 |
| jitter +/-20% | 0.047 | 0.949 | 95.52 | 95.62 | +0.10 | 99.08 |
| crop 80% | 0.004 | 0.440 | 71.74 | 87.84 | +16.10 | 94.91 |

### robust

| Condition | mean score (real) | mean score (AI) | acc@0.5 | acc@best | recoverable | AUC |
|---|---|---|---|---|---|---|
| clean | 0.059 | 0.959 | 95.58 | 95.71 | +0.13 | 99.21 |
| jpeg q=90 | 0.056 | 0.955 | 95.62 | 95.63 | +0.01 | 99.17 |
| jpeg q=70 | 0.056 | 0.958 | 95.73 | 95.75 | +0.03 | 99.22 |
| jpeg q=50 | 0.052 | 0.927 | 94.38 | 94.53 | +0.14 | 98.77 |
| jpeg q=30 | 0.055 | 0.893 | 92.69 | 93.02 | +0.33 | 98.11 |
| blur sigma=0.5 | 0.045 | 0.923 | 94.65 | 94.76 | +0.11 | 98.90 |
| blur sigma=1.0 | 0.100 | 0.924 | 92.59 | 92.69 | +0.10 | 97.97 |
| blur sigma=2.0 | 0.139 | 0.881 | 89.39 | 89.53 | +0.14 | 96.11 |
| resize 0.5x | 0.110 | 0.926 | 92.08 | 92.42 | +0.35 | 97.79 |
| resize 0.25x | 0.169 | 0.880 | 87.94 | 88.11 | +0.17 | 95.16 |
| noise sigma=0.02 | 0.055 | 0.949 | 95.41 | 95.45 | +0.04 | 99.13 |
| noise sigma=0.05 | 0.062 | 0.935 | 94.41 | 94.47 | +0.05 | 98.79 |
| noise sigma=0.10 | 0.099 | 0.906 | 91.42 | 91.55 | +0.13 | 97.42 |
| jitter +/-20% | 0.067 | 0.943 | 94.45 | 94.49 | +0.04 | 98.74 |
| crop 80% | 0.076 | 0.900 | 92.05 | 92.18 | +0.13 | 97.77 |

## 2. Residual failures on clean data

### baseline

325 false positives (3.25% of real images), 337 false negatives (3.37% of AI images).
Contact sheets: `results/errors_baseline_false_positive.png`, `results/errors_baseline_false_negative.png` (most confident mistakes first).

| Image statistic | all test images | false positives | false negatives |
|---|---|---|---|
| high-freq energy | 0.1325 | 0.1305 | 0.1361 |
| saturation | 0.3498 | 0.3040 | 0.3151 |
| brightness | 0.4424 | 0.4558 | 0.4316 |

### robust

530 false positives (5.30% of real images), 354 false negatives (3.54% of AI images).
Contact sheets: `results/errors_robust_false_positive.png`, `results/errors_robust_false_negative.png` (most confident mistakes first).

| Image statistic | all test images | false positives | false negatives |
|---|---|---|---|
| high-freq energy | 0.1325 | 0.1257 | 0.1363 |
| saturation | 0.3498 | 0.3138 | 0.3219 |
| brightness | 0.4424 | 0.4698 | 0.4196 |

## 3. What augmentation cost, and the deployable operating point

Clean accuracy: baseline 96.69% -> robust 95.58% (-1.11 pts). Clean AUC: 99.48 -> 99.21 (-0.27).

A moderation system cannot run at a 50% false-positive-tolerant threshold: wrongly labelling a real photograph as AI-generated is the expensive error. Recall at fixed low false-positive rates:

| Model | Condition | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---|---|---|
| baseline | clean | 90.31 | 98.09 |
| baseline | jpeg_q=30 | 70.84 | 89.43 |
| baseline | blur_sigma=2.0 | 12.76 | 28.99 |
| baseline | resize_0.25x | 12.07 | 27.36 |
| robust | clean | 87.07 | 96.20 |
| robust | jpeg_q=30 | 75.38 | 90.56 |
| robust | blur_sigma=2.0 | 58.79 | 80.87 |
| robust | resize_0.25x | 52.36 | 76.46 |
