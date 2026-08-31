# Robustness results

Every cell comes from a run. Regenerate with:

```
python -m src.eval_robustness --checkpoints baseline robust --save-scores
```

Raw output: `results/robustness.json`, `results/robustness_table.md`,
`results/eval_log.txt`.

**Metric:** accuracy at a fixed 0.5 threshold, and AUC. Both are reported because
they answer different questions. Accuracy is what a deployed system with one
fixed threshold gets. AUC is threshold-free, so where accuracy collapses but AUC
does not, the failure is *calibration* — the score distribution has shifted
across 0.5 while the ranking survived — and is recoverable by re-thresholding
rather than retraining. The two models here differ in exactly that respect, and a
single-metric table would have missed it.

**Test set:** the 20,000-image official CIFAKE test split (10,000 real, 10,000
AI-generated). It is not used in training or model selection; model selection
uses a 5,000-image slice held out of the 100,000-image train split. Stochastic
conditions (noise, colour jitter) are seeded per image index, so both models are
scored on byte-identical inputs.

**Models:** identical architecture (1,562,353 params), seed, schedule and split.
The only difference is whether the transform suite was applied as training-time
augmentation.

## Main table

| Condition | Parameter | Baseline acc | Baseline AUC | Robust acc | Robust AUC | Delta acc |
|---|---|---|---|---|---|---|
| Clean | — | 96.69 | 99.48 | 95.58 | 99.21 | −1.11 |
| JPEG | q=90 | 96.47 | 99.42 | 95.62 | 99.17 | −0.85 |
| JPEG | q=70 | 96.46 | 99.41 | 95.73 | 99.22 | −0.73 |
| JPEG | q=50 | 94.01 | 98.50 | 94.38 | 98.77 | +0.38 |
| JPEG | q=30 | 87.91 | 97.80 | 92.69 | 98.11 | +4.77 |
| Gaussian blur | sigma=0.5 | 80.53 | 98.06 | 94.65 | 98.90 | +14.12 |
| Gaussian blur | sigma=1.0 | 62.11 | 78.12 | 92.59 | 97.97 | +30.47 |
| Gaussian blur | sigma=2.0 | 61.32 | 74.73 | 89.39 | 96.11 | +28.07 |
| Resize | 0.5x | 62.61 | 77.78 | 92.08 | 97.79 | +29.47 |
| Resize | 0.25x | 63.06 | 73.83 | 87.94 | 95.16 | +24.89 |
| Gaussian noise | sigma=0.02 | 95.98 | 99.33 | 95.41 | 99.13 | −0.58 |
| Gaussian noise | sigma=0.05 | 91.30 | 97.38 | 94.41 | 98.79 | +3.11 |
| Gaussian noise | sigma=0.10 | 75.56 | 87.47 | 91.42 | 97.42 | +15.85 |
| Colour jitter | +/−20% | 95.52 | 99.08 | 94.45 | 98.74 | −1.07 |
| Centre crop | 80% | 71.74 | 94.91 | 92.05 | 97.77 | +20.31 |

| Summary | Baseline | Robust |
|---|---|---|
| Clean accuracy | 96.69 | 95.58 |
| Mean over 14 transformed conditions | 81.04 | **93.06** |
| Worst transformed condition | 61.32 | **87.94** |
| Mean drop from clean | 15.65 pts | **2.52 pts** |

## Reading the table

**The cliff is real and it is specific.** The clean-trained baseline is not
uniformly fragile — it barely notices colour jitter (−1.2 pts) or mild JPEG
(−0.2 pts). It falls apart precisely on the transforms that attenuate high
spatial frequencies: blur (−34.6 pts at sigma=1.0), downscale-and-restore (−34.1
pts at 0.5x), and heavy noise (−21.1 pts). That pattern is the evidence for the
mechanism: the detector keys on high-frequency generator fingerprints, and those
transforms are low-pass filters.

**Centre crop is the exception that proves the point.** It costs the baseline
25.0 points of accuracy but only 4.6 of AUC. Cropping does not remove
high-frequency content — it removes *context and scale*, shifting the score
distribution without destroying the ranking. Re-thresholding recovers +16.10
points (see `results/error_analysis.md`). This is a categorically different
failure from blur and would have been indistinguishable from it in an
accuracy-only table.

**Augmentation closes the gap by 6.2x** — from a 15.65-point mean drop to 2.52 —
and lifts the worst case from 61.32% to 87.94%. It costs 1.11 points of clean
accuracy.

**Augmentation also fixes calibration, not just accuracy.** For the robust model
the accuracy recoverable by optimal re-thresholding is at most +0.35 points in
any condition, against +16.10 for the baseline. The robust model's confidence
means roughly the same thing whether the input is clean or degraded, which is
what a single deployed threshold requires.

## Class-conditional structure: the failure reverses

Accuracy alone hides the most interesting result. Splitting into TPR (recall on
AI-generated) and TNR (recall on real) for the **baseline**:

| Condition | TPR | TNR | Mean score, real | Mean score, AI |
|---|---|---|---|---|
| Clean | 0.966 | 0.968 | 0.035 | 0.963 |
| Blur sigma=0.5 | **0.613** | 0.998 | 0.003 | 0.609 |
| Blur sigma=1.0 | **0.265** | 0.977 | 0.036 | 0.280 |
| Blur sigma=2.0 | 0.902 | **0.324** | 0.657 | 0.886 |
| Resize 0.5x | **0.279** | 0.973 | 0.039 | 0.290 |
| Resize 0.25x | 0.863 | **0.398** | 0.590 | 0.843 |
| Crop 80% | **0.438** | 0.997 | 0.004 | 0.440 |

At mild-to-moderate severity the failure is one-sided: AI images get called real
while real images are still classified almost perfectly (blur sigma=1.0 gives TPR
0.265 against TNR 0.977). The evidence of generation has been filtered away, so
generated images look authentic. This is the predicted behaviour.

Past a severity threshold it **reverses**. At blur sigma=2.0 the baseline scores
TPR 0.902 / TNR 0.324, and the mean score assigned to *real* images jumps from
0.003 to 0.657. Heavy smoothing makes authentic photographs look synthetic,
because unnatural smoothness is itself one of the cues the model learned. Resize
shows the identical reversal between 0.5x and 0.25x.

This is why the problem statement's instruction to evaluate each severity
separately matters. Averaging blur sigma=0.5, 1.0 and 2.0 into one "blur
robustness" number yields ~68%, which reads as uniform mild degradation and
describes nothing that is actually happening.

For the robust model the asymmetry is largely gone. Its largest |TPR − TNR| gap
over all 15 conditions is **0.0505** (JPEG q=30: TPR 0.9016 / TNR 0.9521),
against 0.712 for the baseline at blur sigma=1.0 — a 14x reduction. No condition
shows the reversal, and no condition is one-sided enough to matter.

## Full-resolution, out-of-distribution check

The CIFAKE table above is measured at 32x32, where several conditions degenerate:
"resize 0.25x" is an 8x8 thumbnail and an 80% centre crop is 25 pixels wide. This
section re-runs the identical transform suite at real resolution, on a dataset
neither model has seen.

**Set:** 600 SID_Set validation images (300 real from OpenImages, 300 fully
synthetic), median min-side 1024px. Not used for training, validation or model
selection. Regenerate with:

```
python -m src.fetch_fullres --per-class 300
python -m src.eval_fullres --checkpoints baseline robust --modes patch resize --max-patches 64
```

**Metric is AUC only here.** Accuracy at 0.5 is not meaningful on this set: the
threshold was calibrated on CIFAKE, and the mean score both models assign to
clean full-resolution images is 0.14–0.16, so nearly everything lands on one side
of it and accuracy sits at 55–61% regardless of how well the model ranks. AUC is
the only honest read.

| Condition | Parameter | baseline·patch | baseline·resize | robust·patch | robust·resize |
|---|---|---|---|---|---|
| Clean | — | 77.17 | 65.06 | 74.40 | 63.10 |
| JPEG | q=90 | 77.15 | 65.14 | 74.51 | 63.06 |
| JPEG | q=70 | 78.52 | 65.28 | 75.62 | 63.31 |
| JPEG | q=50 | 78.58 | 64.89 | 74.85 | 62.92 |
| JPEG | q=30 | 73.08 | 65.29 | 68.19 | 63.55 |
| Gaussian blur | sigma=0.5 | 73.93 | 65.09 | 72.48 | 63.12 |
| Gaussian blur | sigma=1.0 | 61.96 | 65.22 | 67.98 | 63.15 |
| Gaussian blur | sigma=2.0 | **47.31** | 65.71 | 59.73 | 63.34 |
| Resize | 0.5x | 62.54 | 65.19 | 67.85 | 63.13 |
| Resize | 0.25x | **48.63** | 65.55 | 59.10 | 63.30 |
| Gaussian noise | sigma=0.02 | 62.89 | 64.95 | 66.92 | 62.97 |
| Gaussian noise | sigma=0.05 | **46.22** | 64.96 | 59.69 | 62.83 |
| Gaussian noise | sigma=0.10 | **44.20** | 65.54 | 55.63 | 63.28 |
| Colour jitter | +/−20% | 73.84 | 64.28 | 70.40 | 62.35 |
| Centre crop | 80% | 70.02 | 59.47 | 72.56 | 61.82 |

| Summary (AUC) | baseline·patch | robust·patch |
|---|---|---|
| Clean | 77.17 | 74.40 |
| Mean over 14 transformed | 64.20 | **67.54** |
| Worst transformed | 44.20 | **55.63** |

### What this does and does not show

**The absolute numbers are much weaker, and the reason is generator
generalisation, not robustness.** Clean AUC drops from 99.48 on CIFAKE to 77.17
here. Nothing was transformed to produce that drop — it is the cost of moving
from Stable Diffusion v1.4 at 32x32 to unseen generators at 1024px. The ceiling
in this table is set by transfer, and the transform suite operates underneath it.
The two effects are separable and it would be dishonest to report the robustness
result without saying so.

**The robustness finding itself transfers, with the same sign and shape.** In the
patch-inference column, augmentation lifts the mean transformed AUC from 64.20 to
67.54 and the worst case from 44.20 to 55.63, for 2.77 points of clean AUC. That
is the same trade the CIFAKE table shows — worst-case robustness bought with a
small amount of clean performance — reproduced on a different dataset, a
different generator family and a 32x higher resolution. The magnitude is smaller
because the headroom is smaller.

**Below-chance AUC is the reversal from the CIFAKE table, at real resolution.**
The baseline scores 44.20 at noise sigma=0.10 and 47.31 at blur sigma=2.0 —
worse than a coin flip, which means the ranking is systematically *inverted*, not
merely destroyed. This is the same mechanism as the TPR/TNR flip in the
class-conditional table: past a severity threshold the transform makes authentic
photographs look synthetic. A model that had simply lost its signal would sit at
50. Augmentation removes the inversion everywhere — the robust model's minimum is
55.63.

**Per-family, the pattern matches CIFAKE.** JPEG is nearly free (73–79 across all
four qualities, versus 77.17 clean); colour jitter and crop cost little; blur,
downscale and noise are where both models fail. The transforms that hurt are the
low-pass ones, at 32x32 and at 1024px alike.

## Inference mode at full resolution

The detector is trained at 32x32. Applying it to a 1024px image can be done by
resizing the image down to 32x32, or by tiling it into 32x32 patches at native
resolution and averaging the per-patch logits. Downscaling is a low-pass filter
and therefore destroys exactly the evidence the detector was trained to use, so
the patch route should win.

On clean images it wins decisively: **77.17 AUC versus 65.06** for the baseline
(+12.11), and 74.40 versus 63.10 for the robust model (+11.30). Resizing a 1024px
image to 32x32 costs about twelve points of AUC, which is the cost of the
low-pass filter alone.

But the two modes fail in completely different ways, and the interesting result
is in the *spread* rather than the mean:

| Mode | AUC range across all 15 conditions | Spread |
|---|---|---|
| baseline·patch | 44.20 – 78.58 | 34.38 pts |
| baseline·resize | 59.47 – 65.71 | 6.24 pts |
| robust·patch | 55.63 – 75.62 | 19.99 pts |
| robust·resize | 61.82 – 63.55 | **1.74 pts** |

**Resize-mode is nearly transform-invariant, and that is not a virtue.** The
robust model in resize mode varies by 1.74 points of AUC across the entire
transform suite — from clean to noise sigma=0.10. It looks like the most robust
configuration in the project. It is actually the least informative one: resizing
to 32x32 has *already* destroyed the high-frequency evidence that blur, noise and
downscaling destroy, so there is nothing left for the transforms to take. Its
flatness is a floor, not resilience. This is the clearest argument in the project
for not reporting robustness as a single averaged degradation number — by that
metric alone, resize-mode wins.

**Augmentation widens the range over which native-resolution inference pays.**
Counting conditions where patch beats resize: the baseline wins 8 of 15 — it
gives up under blur sigma>=1.0, both resize levels and all three noise levels.
The robust model wins **11 of 15**, holding on through blur sigma=1.0, resize
0.5x and noise sigma=0.02, and losing only in the four harshest low-pass
conditions. Augmentation does not just improve the model; it extends the range of
input degradation over which it is still worth paying for native-resolution
inference.

**Practical consequence, and what `predict.py` does.** `--mode patch` is the
default because it is the better choice on clean and mildly-degraded input, which
is the majority case. The crossover is real, though: on input known to be heavily
blurred or downscaled, resize-mode is better. A deployed system with a degradation
estimate could switch on it. That is not implemented — it is stated because the
measurement supports it and the table above is where someone would start.
