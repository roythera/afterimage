# Error analysis

Generated numbers in `results/error_analysis.md` and `results/error_analysis.json`;
contact sheets in `results/errors_*.png`. Regenerate with:

```
python -m src.eval_robustness --checkpoints baseline robust --save-scores
python -m src.error_analysis
```

All figures below are on the 20,000-image held-out CIFAKE test split (10,000
real, 10,000 AI-generated).

## Summary

| | Baseline | Robust |
|---|---|---|
| False positives (real called AI) | 325 (3.25% of real) | 530 (5.30%) |
| False negatives (AI called real) | 337 (3.37% of AI) | 354 (3.54%) |

The robust model makes *more* clean-data errors, and they are disproportionately
false positives. That trade is discussed at the end; it is the main cost of
augmentation and the reason threshold choice matters more than the headline
accuracy.

## False positives — real images called AI-generated

The confident false positives are strikingly homogeneous. In both models the
contact sheets (`results/errors_robust_false_positive.png`) are dominated by
**isolated subjects on smooth, uniform backgrounds**: aircraft against blank sky,
cars against blurred road, boats on flat water, birds against plain gradients.
Frequently the subject is centred, the background is a near-constant colour
field, and there is little fine texture anywhere in the frame.

These are authentic photographs that happen to carry the *compositional*
signature of a generated image — one salient object, shallow depth of field, low
background complexity. That is what text-to-image models produced in 2022, and
CIFAR-10 contains plenty of real photographs that look the same way.

Measured, this shows up as reduced high-frequency content:

| Image statistic | All test images | False positives | False negatives |
|---|---|---|---|
| High-frequency energy (mean abs. Laplacian) | 0.1325 | **0.1257** | **0.1363** |
| Saturation | 0.3498 | 0.3138 | 0.3219 |
| Brightness | 0.4424 | 0.4698 | 0.4196 |

(Robust model. The baseline shows the same ordering: 0.1305 / 0.1361 around a
mean of 0.1325.)

## False negatives — AI images called real

The opposite. `results/errors_robust_false_negative.png` is dominated by **busy,
densely textured natural scenes**: foliage, grass, tree canopies, animal fur,
cluttered backgrounds. Where the image is full of high-frequency natural texture,
that texture masks the generator fingerprint — the statistics the detector relies
on are swamped by legitimate high-frequency content.

Their mean high-frequency energy (0.1363) is *above* the test-set mean, exactly
inverting the false-positive case.

## The organising principle

Both error classes lie on a single axis — **scene texture density** — and the two
classes fail at opposite ends of it:

- Low texture → real images get called generated (smoothness reads as synthetic).
- High texture → generated images get called real (natural texture masks the
  fingerprint).

This is not a separate observation from the robustness result; it is the same
one. Blur, downscaling and JPEG all move images *down* the texture axis, which is
precisely why they break the clean-trained baseline, and why the breakage is
asymmetric (see `docs/robustness.md`): at moderate severity they push AI images
into the "looks real" regime, and at high severity they push real images into the
"too smooth to be real" regime.

The single most useful consequence: **the detector is not measuring
authenticity, it is measuring texture statistics that correlate with
authenticity.** Augmentation improves robustness by widening the range of texture
statistics associated with each class, not by teaching the model a new cue.

## Trade-offs

### Robustness vs clean accuracy

Augmentation cost 1.11 points of clean accuracy (96.69 → 95.58) and 0.27 of clean
AUC, in exchange for reducing the mean drop across transformed conditions from
15.65 points to 2.52 and lifting worst-case accuracy from 61.32% to 87.94%. On
any realistic distribution of post-processed inputs this is strongly favourable.

### Robustness vs false-positive rate

This is the real cost, and it is not visible in the headline table. The robust
model's clean false-positive rate is 5.30% against the baseline's 3.25% — a 63%
relative increase. The mechanism follows directly from the error analysis above:
augmentation teaches the model that smooth, low-texture images can be
*authentic-but-degraded*, which necessarily moves the boundary and pulls some
genuinely smooth real photographs across it.

### Calibration

Re-thresholding each condition optimally recovers, for the **baseline**:

| Condition | acc @ 0.5 | acc @ best threshold | Recoverable |
|---|---|---|---|
| Centre crop 80% | 71.74 | 87.84 | **+16.10** |
| Blur sigma=0.5 | 80.53 | 92.97 | **+12.44** |
| Blur sigma=1.0 | 62.11 | 70.38 | +8.27 |
| Resize 0.5x | 62.61 | 70.17 | +7.56 |
| JPEG q=30 | 87.91 | 92.55 | +4.63 |

For the **robust** model the largest recoverable gap in any condition is **+0.35**
points. Augmentation did not only make the model more accurate under transforms;
it made its confidence mean approximately the same thing across all of them.
For a deployed system that must pick one threshold and keep it, this matters as
much as the accuracy gain.

### Operating point, and the cost of a false accusation

In a moderation setting, labelling a real photograph as AI-generated is the
expensive error, so a 0.5 threshold is not the relevant operating point. Recall
at fixed low false-positive rates:

| Model | Condition | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---|---|---|
| Baseline | clean | 90.31 | 98.09 |
| Robust | clean | 87.07 | 96.20 |
| Baseline | JPEG q=30 | 70.84 | 89.43 |
| Robust | JPEG q=30 | **75.38** | **90.56** |
| Baseline | blur sigma=2.0 | 12.76 | 28.99 |
| Robust | blur sigma=2.0 | **58.79** | **80.87** |
| Baseline | resize 0.25x | 12.07 | 27.36 |
| Robust | resize 0.25x | **52.36** | **76.46** |

On clean images the baseline is slightly better (90.31 vs 87.07 at 1% FPR). On
degraded images it is not close: under blur sigma=2.0 the baseline catches 1 in 8
AI images at a 1% false-positive budget, the robust model nearly 3 in 5. Since
essentially every image on a real platform has been re-encoded and resized at
least once, the robust model is the correct choice despite the worse clean
false-positive rate — but it should be deployed at a threshold chosen from this
table, not at 0.5.

### Generalisation to unseen generators

Distinct from robustness, and on this evidence much harder. CIFAKE's synthetic
half is entirely Stable Diffusion v1.4. Measured on 600 held-out SID_Set images
at ~1024px, from generators and a source dataset that appear nowhere in training:

| | CIFAKE clean AUC | Full-res clean AUC | Gap |
|---|---|---|---|
| Baseline | 99.48 | 77.17 | −22.31 |
| Robust | 99.21 | 74.40 | −24.81 |

No transform is applied in either column. The whole gap is generator and
resolution transfer, and it is the same order of magnitude as the worst
post-processing effect measured anywhere in this project: the worst single
transform costs the baseline 25.65 points of AUC on CIFAKE (99.48 → 73.83 at
resize 0.25x), and transfer costs 22.31 before the transform suite is applied at
all. The two are comparable in size, and they compound.

The two failures are also **different in kind**. Post-processing failure is
substantially miscalibration and is recoverable by re-thresholding (up to +16.10
points, above). Transfer failure is not: the AUC itself has moved, so no
threshold recovers it. Everything else in this document characterises the first
failure. The second is neither a threshold problem nor an augmentation problem —
it needs training data from more than one generator, which is the highest-value
next change to this project and was out of reach in a one-day build.

Augmentation does not help here and slightly hurts (−2.77 clean AUC relative to
the baseline at full resolution), which is consistent with its mechanism: it
teaches invariance to *degradation*, not invariance to *generator*. What matters
is that it does not hurt much, and that under transforms at full resolution it
still wins — mean transformed AUC 67.54 against 64.20, worst case 55.63 against
44.20. The robustness benefit survives the distribution shift even though the
absolute performance level does not.

Robustness to post-processing was achieved; generator generalisation was not. The
two should not be conflated, and the headline number in this project is a claim
about the first only.
