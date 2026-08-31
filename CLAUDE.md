# TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images

## What this project is

Build a prototype that distinguishes AI-generated images from authentic ones and
**stays accurate after realistic post-processing** — compression, blur, resizing,
noise, colour adjustment, cropping. Clean-data accuracy is table stakes; the
result the judges care about is how little accuracy is lost under transformation.

Hardware: Kaggle Notebooks, 2x NVIDIA Tesla T4 (sm_75, 16GB each).
Time budget: one build day. Scope accordingly.

## Hard rules

1. **Model must be under 2B parameters.** Stated limit in the problem statement.
   Not a concern for anything we will realistically train, but state the
   parameter count in the README so it is visibly satisfied.
2. **Never report a number that did not come from a run.** No estimates. Paste
   the output.
3. **The validation set is off-limits for training.** See "Datasets" below. The
   organisers name a specific WildFake subset for demonstration only. Training on
   it invalidates the demonstration.
4. **Build the transform-evaluation harness before training anything.** It is the
   measuring instrument. Without it, no result means anything.
5. **Log every experiment** to `docs/iterations.md` as it finishes — hypothesis,
   change, numbers. This is the raw material for the robustness summary and error
   analysis, both of which are required deliverables.
6. **One variable at a time.** Changing the backbone and the augmentation set
   together tells you nothing about either.

## The core experiment

This is the whole project in four lines:

1. Train a detector on **clean** images only. Evaluate on clean + every transform.
   Expect a large accuracy cliff, especially under JPEG and downscale.
2. Train the same architecture with the **transforms applied as training-time
   augmentation**. Evaluate identically.
3. The gap between those two robustness profiles is the headline result.
4. Analyse where the robust model still fails. That is the error-analysis note.

Do not skip step 1. The unaugmented baseline is what makes the improvement
legible. A single robust model with nothing to compare against is a much weaker
submission.

## Transform suite (exact, from the problem statement)

Implement all of these. Evaluate each severity separately — a single averaged
"robust accuracy" number hides the interesting structure.

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG compression | quality = 90, 70, 50, 30 | social re-encode, messaging |
| Gaussian blur | sigma = 0.5, 1.0, 2.0 | out of focus |
| Resize | scale 0.5x / 0.25x, then upscale back | thumbnail generation |
| Gaussian noise | sigma = 0.02, 0.05, 0.10 | low-light sensor noise |
| Colour jitter | brightness/contrast/saturation +/-20% | filter apps, auto-enhance |
| Centre crop | 80% | profile-picture cropping |

Notes that matter:
- Noise sigma is in **normalised [0,1] units**, so apply before normalisation,
  and clamp back to [0,1] after.
- JPEG must be a **real encode/decode round trip** (PIL save to buffer, reload),
  not a blur approximation. The compression artefacts are the entire point —
  many AIGC detectors key on high-frequency generator fingerprints, which is
  exactly what JPEG destroys.
- Resize must upscale **back to the original size** so the tensor shape is
  unchanged; the information loss is what is being tested.
- Apply transforms to the **evaluation** set to measure robustness, and
  (in the second training run) to the **training** set as augmentation.

## Datasets

Primary, and the pragmatic choice given one day:
- **CIFAKE** — hosted on Kaggle (`birdy654/cifake-real-and-ai-generated-synthetic-images`),
  so it attaches to the notebook in two clicks with no download. 32x32,
  60k real / 60k fake, trains in minutes.

Also available:
- **SID_Set** — `saberzl/SID_Set` on HuggingFace. Full resolution.
- **WildFake** — on ModelScope (use the page's translate button).

**Off-limits for training:** the organisers designate a WildFake subset for
demonstration only — 4,998 non-AIGC images from COCO val2017 and 8,843 AIGC
images from DALL-E Advanced. It does not contribute to the final score and must
not appear in training data.

### Known limitation to handle honestly

CIFAKE is 32x32. At that resolution, "resize to 0.25x then upscale" and "JPEG
quality 30" are close to degenerate, and a 32x32 crop at 80% is 25 pixels wide.
Two acceptable responses, in order of preference:

1. Train on CIFAKE, then **validate the robustness claim on a full-resolution
   slice** of SID_Set or the WildFake demonstration subset. Even a few hundred
   images makes the finding credible at real resolution.
2. If time runs out, state the limitation explicitly in the README. Naming it
   yourself reads as insight; having a judge notice it reads as an oversight.

## Required deliverable: the prediction script

The problem statement specifies this precisely. Build it early, not last:

- Takes an **image directory** as input.
- Outputs a **JSON file** with `image_path` and `pred` for each image, where
  `pred` is a confidence score for "this is AI-generated".
- Must run from a clean clone by following the README.

Treat it as the interface everything else is built around.

## Judging weights

| Criterion | Weight |
|---|---|
| Technical Execution | 35% |
| Innovation & Problem Insight | 20% |
| Impact & Relevance | 20% |
| Feasibility & Practicality | 15% |
| Presentation & Communication | 10% (final event only) |

Implication: roughly 40% is framing and insight, not accuracy. The robustness
table and the error analysis carry more weight than the last point of accuracy.
Budget time accordingly.

## Scope

In scope: image-level detection, robustness to transformations, feature
engineering, model design, evaluation design, error analysis, explainability.

Out of scope: production deployment, platform-wide moderation systems, video or
audio. Do not build a web UI — a walkthrough video of inference and results is
explicitly accepted for backend submissions.

## Time discipline

At the **two-hours-remaining** mark, stop training and write. README, robustness
table, error-analysis note, demo video. A modest model with a rigorous writeup
beats a better model with no submission — and the deliverables are individually
required, so a missing one costs more than a weaker number.
