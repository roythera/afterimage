# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 — Track 5

## Project overview

A 1.56M-parameter convolutional detector that separates AI-generated images from
authentic ones and, more importantly, keeps working after the image has been
through the things that happen to every image on a social platform: JPEG
re-encoding, blur, downscaling, sensor noise, filter apps and cropping.

The project is built around one controlled comparison. The **same architecture,
seed, schedule and data split** are trained twice — once on clean images, once
with the post-processing transform suite applied as training-time augmentation —
and both are evaluated across all 15 transform conditions on the same held-out
20,000-image test set. Everything else is held fixed, so the difference between
the two robustness profiles is attributable to the augmentation and nothing else.

**Headline: the accuracy cliff under post-processing shrinks from 15.65 points to
2.52 points — a 6.2x reduction — at a cost of 1.11 points of clean accuracy.**
Worst-case accuracy over all conditions rises from 61.32% to 87.94%.

## Headline result

Accuracy (%) on the 20,000-image held-out CIFAKE test split.

| | Clean | JPEG 30 | Blur 2.0 | Resize 0.25x | Noise 0.10 | Jitter | Crop 80% |
|---|---|---|---|---|---|---|---|
| Baseline (clean-trained) | 96.69 | 87.91 | 61.32 | 63.06 | 75.56 | 95.52 | 71.74 |
| Robust (augmented) | 95.58 | 92.69 | **89.39** | **87.94** | **91.42** | 94.45 | **92.05** |
| Delta | −1.11 | +4.77 | **+28.07** | **+24.89** | **+15.85** | −1.07 | **+20.31** |

Summary over all 14 transformed conditions:

| | Clean | Mean transformed | Worst transformed | Mean drop from clean |
|---|---|---|---|---|
| Baseline | 96.69 | 81.04 | 61.32 | **15.65 pts** |
| Robust | 95.58 | 93.06 | 87.94 | **2.52 pts** |

Full per-severity table with AUC in [`docs/robustness.md`](docs/robustness.md).

## Approach

### The measuring instrument comes first

The transform suite (`src/transforms.py`) was written and verified before any
model was trained, because every claim in this report is a claim about it. It is
checked by `python -m src.selftest`, which verifies shape and range preservation,
determinism under a fixed seed, and monotonicity in severity. Two of those checks
are load-bearing:

- **Noise sigma is in normalised [0,1] units.** Measured RMSE against the clean
  image is 0.0198 / 0.0490 / 0.0965 for requested sigmas of 0.02 / 0.05 / 0.10 —
  within 2%. Had sigma been interpreted in 0-255 units the perturbation would
  have been 255x too large and every noise row would have been meaningless.
- **JPEG is a real encode/decode round trip**, not a blur approximation.
  Re-encoding an already-q50 image at q50 changes it by RMSE 0.00000, which only
  a real codec does; a smoothing stand-in keeps degrading on each application.
  This matters more than it sounds, because JPEG's high-frequency quantisation is
  precisely what destroys generator fingerprints — approximating it would have
  removed the phenomenon under study.

### Architecture

`RobustNet`, a compact residual CNN trained from scratch. Three choices are
deliberate:

- **No ImageNet backbone.** A pretrained network expects 224x224 and opens with a
  stride-2 conv and stride-2 pool that discard 3/4 of the spatial signal before
  the first block. Generator fingerprints are a high-frequency phenomenon, so
  that stem throws away the evidence. Pretrained features also encode semantics
  ("is this a dog"), not provenance ("was this sampled from a diffusion model").
- **Stride-1 stem.** Downsampling is deferred to the stage transitions for the
  same reason.
- **Fully convolutional with global average pooling.** The classifier accepts any
  input size, which is what makes patch-based full-resolution inference possible
  without retraining.

### Augmentation strategy

Two decisions here are not obvious and both are deliberate:

**Continuous severity ranges, not the evaluation severities.** Training samples
JPEG quality uniformly from [30, 95], blur sigma from [0.3, 2.0], and so on —
intervals that *span* the discrete test severities without containing them as
special values. Training on exactly the test settings would inflate the
robustness table without demonstrating anything general; sampling the interval
makes every evaluation point an interpolation rather than a memorised case.

**Chained transforms.** Up to two transforms are composed per image. In the wild
an image is screenshotted, resized and re-encoded, and the composition is harsher
than any single step.

## Robustness evaluation

Method: all 15 conditions (clean plus 14 transformed) are applied to the same
20,000 held-out CIFAKE test images, which appear in no training run. Stochastic
conditions (noise, colour jitter) are seeded per image index, so both models are
scored on byte-identical inputs — the comparison is exact, not statistical.

Two metrics are reported per condition:

- **Accuracy at a fixed 0.5 threshold**, which is what a deployed system with a
  fixed threshold actually gets.
- **AUC**, which is threshold-free.

Reporting both is load-bearing, not decoration. A transform can leave the ranking
almost intact while shifting the whole score distribution across 0.5. That
collapses accuracy but not AUC, and it means the failure is *calibration*, not
loss of signal — recoverable by re-thresholding rather than requiring retraining.
The baseline under an 80% centre crop is exactly this case: accuracy 71.74% but
AUC 94.91. Reporting accuracy alone would have misdiagnosed it.

Every severity is reported separately. A single averaged "robust accuracy" would
have hidden the most interesting finding in the project.

## Key findings

### 1. The baseline fails asymmetrically, and past a threshold the asymmetry reverses

At blur sigma=0.5 the clean-trained baseline scores TPR 0.613 against TNR 0.998:
it still recognises real images almost perfectly but has stopped recognising
AI-generated ones. That is the exact signature of a detector whose evidence has
been low-pass filtered away — blur destroys the generator fingerprint, so
generated images start to look authentic.

At blur sigma=2.0 it *flips* to TPR 0.902 / TNR 0.324. Past a certain severity,
blur makes **real** images look synthetic, because unnatural smoothness is itself
a cue the model uses. The mean score assigned to real images goes from 0.003 at
sigma=0.5 to 0.657 at sigma=2.0. The same reversal appears between resize 0.5x
(TPR 0.279) and resize 0.25x (TPR 0.863).

A single averaged robustness number would have shown a smooth decline and hidden
this entirely. It is the main argument for the per-severity table.

### 2. Much of the baseline's failure is miscalibration, not lost signal

Re-thresholding each condition optimally recovers, for the baseline, +16.10
points under centre crop and +12.44 under blur sigma=0.5 — those failures were
threshold placement, not lost discriminative power. Under blur sigma=1.0 only
+8.27 points come back from an AUC of 78.12, so that failure is genuine signal
loss.

For the robust model the recoverable gap is at most +0.35 points in any
condition. Augmentation did not merely raise accuracy; it made the model's
confidence *mean the same thing* across transforms. That is the property a
deployed system running one fixed threshold actually needs.

### 3. JPEG hurts far less than expected, and the reason is in the data

The baseline loses only 8.8 points at JPEG q=30 versus 34.6 at blur sigma=1.0.
CIFAKE is *already* stored as JPEG, so every JPEG condition here is a
second-generation re-encode and the most fragile artefacts are gone before
training begins. This is a property of the dataset, not a general finding about
JPEG, and it is flagged rather than claimed as a result.

### 4. At real resolution the finding survives — but transfer, not robustness, is the binding constraint

CIFAKE is 32x32, so the headline table is re-run on 600 held-out full-resolution
SID_Set images (~1024px, unseen generators, unseen source dataset). AUC, patch
inference:

| | Clean | Mean transformed | Worst transformed |
|---|---|---|---|
| Baseline | 77.17 | 64.20 | 44.20 |
| Robust | 74.40 | **67.54** | **55.63** |

The ordering holds — same trade, same direction, smaller magnitude. But clean AUC
falls from 99.48 to 77.17 *with no transform applied at all*. That 22.31-point
generator/resolution transfer gap is the same order of magnitude as the worst
single transform in the entire CIFAKE table (25.65 points of AUC at resize
0.25x), and it is paid before the transform suite is applied. Robustness to
post-processing and generalisation to unseen generators are separate problems of
comparable size, and this project only solves the first. Saying so is more useful
than reporting the improvement alone.

Two things only visible at full resolution:

- **The baseline scores *below chance* under heavy noise (44.20 AUC).** Worse
  than a coin flip means the ranking is systematically inverted, not lost — the
  same reversal as finding 1, at 1024px. Augmentation removes it everywhere.
- **Resizing a 1024px image down to 32x32 makes the detector look robust by
  destroying its own evidence first.** In resize mode the robust model varies by
  **1.74 AUC points across all 15 conditions** — the flattest profile in the
  project, and the least useful, because patch inference beats it by 12.11 points
  on clean input. Its flatness is a floor, not resilience. This is the sharpest
  argument for the per-severity table over any single averaged robustness score.

## Error analysis

Full version in [`docs/error-analysis.md`](docs/error-analysis.md); generated
numbers and contact sheets in `results/`.

On clean data the robust model produces 530 false positives (5.30% of real
images) and 354 false negatives (3.54% of AI images). The two error classes have
visually distinct and opposite character:

- **False positives — real photos called AI-generated** are overwhelmingly
  *isolated subjects on smooth, uniform backgrounds*: aircraft against blank sky,
  cars against blurred backdrops, boats on flat water, birds against plain
  gradients. These are authentic photographs that happen to have the
  compositional signature of a generated image — one centred subject, shallow
  depth of field, low background texture. Mean high-frequency energy 0.1257
  versus 0.1325 across the test set.
- **False negatives — AI images called real** are busy, textured natural scenes:
  foliage, grass, animal fur, cluttered backgrounds. Mean high-frequency energy
  0.1363, above the test-set average.

The errors are therefore organised along a single axis — **scene texture
density** — pushing the two classes in opposite directions. Where natural texture
statistics dominate they mask the generator fingerprint; where an authentic photo
lacks texture, its smoothness is mistaken for the fingerprint. This also explains
finding 1: blur moves every image along that same axis.

### Deployable operating point

Accuracy at a 0.5 threshold is the wrong metric for moderation, where wrongly
labelling a real photograph as AI-generated is the expensive error. At a fixed
1% false-positive rate:

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

At the operating point a platform would actually deploy, the clean-trained
baseline catches 1 in 8 blurred AI images. The robust model catches nearly 3 in 5
— a 4.6x improvement where it matters, for 3.2 points of clean recall.

## Trade-offs

- **Robustness vs clean accuracy.** Augmentation cost 1.11 points of clean
  accuracy (96.69 → 95.58) and 0.27 of clean AUC. Cheap for a 6.2x reduction in
  the transformed-condition cliff.
- **False-positive rate.** The robust model's clean false-positive rate is higher
  (5.30% vs 3.25%). Augmentation teaches the model that smooth, low-texture
  images can still be authentic-but-degraded, which pulls some genuinely smooth
  real photographs across the boundary. In a moderation setting this is the error
  that matters most, and it is the clearest argument against simply shipping at
  threshold 0.5 — the operating-point table above is the honest way to set it.

## Limitations and future work

Stated plainly rather than left for a judge to find:

- **CIFAKE is 32x32.** At that resolution several conditions are close to
  degenerate: "resize 0.25x" means an 8x8 thumbnail, and an 80% centre crop is 25
  pixels wide. The full-resolution check exists because of this, and it is the
  main caveat on the headline table.
- **One generator.** CIFAKE's fake half is entirely Stable Diffusion v1.4.
  Robustness to post-processing is not the same as generalisation to unseen
  generators, and the full-resolution results show the latter is much harder.
- **The full-resolution slice is small** (600 images) and the images were served
  JPEG-transcoded, so they carry one prior encode.
- **Not tested:** adversarial perturbation, screenshotting, re-photography,
  platform-specific pipelines, or transform compositions deeper than two.
- **Next step:** train at full resolution on patches from SID_Set rather than
  transferring a 32x32 model, which the full-resolution numbers suggest is the
  binding constraint.

## Setup and reproduction

Verified from a clean clone.

```bash
git clone <repo>
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. data (~50MB)
python -m src.get_data

# 2. verify the transform suite BEFORE trusting any number it produces
python -m src.selftest

# 3. the core experiment: two runs, one variable
python -m src.train --aug none   --tag baseline --epochs 12
python -m src.train --aug robust --tag robust   --epochs 12

# 4. robustness table across all 15 conditions
python -m src.eval_robustness --checkpoints baseline robust --save-scores

# 5. error analysis
python -m src.error_analysis

# 6. full-resolution out-of-distribution check
python -m src.fetch_fullres --per-class 300
python -m src.eval_fullres --checkpoints baseline robust --modes patch resize
```

`notebooks/kaggle_run.ipynb` runs the same sequence on Kaggle (T4 x2, Internet
on). All scripts take `--device auto` and run on CPU, CUDA or Apple MPS.

### Required prediction script

Both trained checkpoints are committed (6MB each), so this runs immediately after
`pip install -r requirements.txt` — no dataset download and no retraining:

```bash
python -m src.predict --image-dir <dir> --output preds.json
```

Output:

```json
[{"image_path": "img/0001.png", "pred": 0.93}]
```

`pred` is P(image is AI-generated) in [0,1]. The directory is searched
recursively for png/jpg/jpeg/bmp/webp/tiff.

**Inference at resolutions above 32x32.** The obvious approach — resize the input
down to 32x32 — is the worst available choice, because downscaling is a low-pass
filter and destroys the high-frequency evidence the detector depends on. The
default `--mode patch` instead tiles the image at native resolution into 32x32
patches and averages the per-patch *logits* (logits, not probabilities, which
saturate and let a few confident patches dominate). Measured on the
full-resolution slice, this is worth **+12.11 AUC on clean images** (77.17 vs
65.06).

The choice is not unconditional, which is why `--mode resize` and `--mode full`
remain available. Under heavy blur, downscaling or noise the ordering flips —
patch inference beats resize in 11 of 15 conditions for the robust model but only
8 of 15 for the baseline. Full comparison in
[`docs/robustness.md`](docs/robustness.md).

## Model size

**1,562,353 parameters (0.00156B)** — three orders of magnitude below the 2B
limit. Print it yourself with `python -m src.model`.

## Datasets used

- **CIFAKE** (`dragonintelligence/CIFAKE-image-dataset`, the HuggingFace mirror
  of `birdy654/cifake-real-and-ai-generated-synthetic-images`) — 60k real
  (CIFAR-10) + 60k fake (Stable Diffusion v1.4) at 32x32. The authors' original
  split is used unchanged: 100k train (of which 5k is held out for model
  selection), 20k test. The test split is used for evaluation only.
- **SID_Set** (`saberzl/SID_Set`, validation split) — 600 full-resolution images,
  300 real (OpenImages) and 300 fully synthetic, used **only** as a held-out
  out-of-distribution test set.

**The designated demonstration subset was not used in training.** The WildFake
subset the organisers reserve for demonstration (4,998 COCO val2017 non-AIGC
images and 8,843 DALL-E Advanced AIGC images) does not appear anywhere in this
project — not in training, validation, or evaluation. Training data is
exclusively the CIFAKE train split.

## Development tools, APIs, libraries

- **PyTorch** 2.2.2 / **torchvision** 0.17.2 — model, training, `gaussian_blur`
  and the colour-jitter primitives.
- **Pillow** 11.3.0 — JPEG codec round trip, resampling, cropping.
- **NumPy** 1.26.4, **pandas** 2.3.3, **pyarrow** 21.0.0 — parquet decoding and
  array handling.
- **scikit-learn** 1.6.1 — `roc_auc_score`.
- **requests** — dataset download.
- **HuggingFace datasets-server API** — row-level fetch of the full-resolution
  SID_Set slice without downloading the 140GB dataset.
- **Claude Code** (Anthropic) — used throughout for implementation. What it got
  wrong is logged per-run in [`docs/iterations.md`](docs/iterations.md); the
  pattern was that it produced plausible code whose errors surfaced only when
  output was checked against a prior expectation, which is why the transform
  self-test was written before the model.

Hardware: Apple M2 (8GB), PyTorch MPS backend. Baseline 48 min, robust 33 min.

## Repository layout

```
src/transforms.py       transform suite + training augmentation
src/selftest.py         verifies the suite before it is trusted
src/data.py             CIFAKE loading, label convention, normalisation
src/model.py            RobustNet (1.56M params)
src/train.py            training, --aug none | robust
src/eval_robustness.py  15-condition table on the held-out test split
src/error_analysis.py   failure characterisation and operating points
src/eval_fullres.py     full-resolution out-of-distribution check
src/predict.py          REQUIRED: image dir -> JSON
src/fetch_fullres.py    pulls the SID_Set slice via the datasets-server API
src/get_data.py         downloads CIFAKE
checkpoints/*.pt        both trained models, committed so predict.py runs on clone
docs/robustness.md      full per-severity results
docs/error-analysis.md  error analysis writeup
docs/iterations.md      experiment log, including negative results
results/                raw output backing every number in this README
notebooks/kaggle_run.ipynb   same pipeline on Kaggle 2xT4
```

Every number in this README and in `docs/` is traceable to a file in `results/`:
`robustness.json` (CIFAKE table), `fullres.json` (full-resolution table),
`error_analysis.json` (calibration, operating points, error statistics).

## Team

Solo — Roy Sasson, NUS.
