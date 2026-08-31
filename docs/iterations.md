# Experiment log

One entry per run, filled in when the run finishes. This is the raw material for
the robustness summary and the error-analysis note, both of which are required
deliverables. Do not reconstruct it at the end.

Hardware for all runs below: Apple M2, 8GB, PyTorch MPS backend. The code is
device-agnostic (`--device auto`) and runs unchanged on the Kaggle 2xT4 setup in
`kaggle_setup.txt`; at 32x32 it was faster to iterate locally.

---

## 00 — data loads, transform suite verified

**Hypothesis:** CIFAKE loads with correct label semantics and the transform
suite does what the problem statement says. Nothing downstream means anything
until the measuring instrument is checked, so this is built before any training.

**Change:** `src/transforms.py`, `src/data.py`, `src/selftest.py`.

**Command:**
```
python -m src.get_data
python -m src.selftest
```

**Result:**
```
condition              RMSE vs clean   checks
clean                        0.00000   ok
jpeg_q=90                    0.00886   ok
jpeg_q=70                    0.01292   ok
jpeg_q=50                    0.04053   ok
jpeg_q=30                    0.04799   ok
blur_sigma=0.5               0.03195   ok
blur_sigma=1.0               0.08945   ok
blur_sigma=2.0               0.13896   ok
resize_0.5x                  0.09788   ok
resize_0.25x                 0.14482   ok
noise_sigma=0.02             0.01976   ok
noise_sigma=0.05             0.04902   ok
noise_sigma=0.10             0.09652   ok
jitter_+/-20%                0.02209   ok
crop_80%                     0.19781   ok

monotone severity jpeg     ok  [0.0089, 0.0129, 0.0405, 0.048]
monotone severity blur     ok  [0.032, 0.0895, 0.139]
monotone severity resize   ok  [0.0979, 0.1448]
monotone severity noise    ok  [0.0198, 0.049, 0.0965]
jpeg idempotence   0.00000  (should be << 0.04053)
RobustAugment      fires 86% of the time, 163 distinct outputs
all transform checks passed
```

**Verdict:** kept. Two of these checks are load-bearing. The measured noise RMSE
lands within 2% of the requested sigma (0.0198 / 0.0490 / 0.0965 for 0.02 / 0.05
/ 0.10), confirming sigma really is in normalised [0,1] units and not 0-255. And
re-encoding an already-q50 image at q50 changes nothing (RMSE 0.00000),
confirming JPEG is a real codec round trip rather than a blur approximation — a
smoothing stand-in would keep degrading every time it was applied.

**Three things checked about the data itself, each of which could have silently
invalidated the whole report:**

- *Label semantics.* The HF mirror uses 0 = FAKE, 1 = REAL. The project
  convention is inverted (1 = AI-generated) so that `pred` means
  P(AI-generated), as the problem statement requires. Getting this backwards
  would have inverted every number.
- *Codec leakage.* Both classes are stored as JPEG with near-identical mean file
  size (925.2 vs 921.8 bytes over 200 samples each). Had real images been PNG
  and fakes JPEG, the model could have learned the container rather than the
  content and every result would have been an artefact.
- *Channel statistics.* The first attempt returned mean (0.164, 0.164, 0.164) —
  identical across channels, which is impossible for natural images. Cause was
  float32 accumulation over 3x10^8 pixels: the accumulator reaches ~4x10^7,
  where adding 0.4 falls below the representable increment. Recomputed in
  float64: (0.4720, 0.4629, 0.4178), consistent with CIFAR-10.

**AI workflow note:** Claude Code wrote the suite and caught none of the three
issues above from the code alone — each was found by checking output against a
prior expectation (channel means should be ~0.45; noise RMSE should equal sigma;
a real codec should be idempotent). Writing the self-test before the model was
the single highest-value decision in the project.

---

## 01 — baseline: trained on clean data only

**Hypothesis:** A detector trained only on clean CIFAKE reaches ~95% clean
accuracy and then falls off sharply under post-processing, worst under the
transforms that destroy high frequencies (blur, downscale), because that is
where the generator fingerprint lives. This is the comparison point that makes
any later improvement legible, so it is run first.

**Change:** `RobustNet`, 1,562,353 parameters, stride-1 stem (no early
downsampling), GAP head. Training data left clean apart from horizontal flips.

**Command:**
```
python -m src.train --aug none --tag baseline --epochs 12 --workers 4
```

**Result:**
```
tag=baseline  aug=none  exclude=-  device=mps
train=95,000  val=5,000  params=1,562,353 (0.001562B, limit 2B)
  epoch  0  train_loss 0.2584  val_acc 0.9222  val_loss 0.2058  144s *
  epoch  1  train_loss 0.1754  val_acc 0.8926  val_loss 0.2810  119s
  epoch  2  train_loss 0.1443  val_acc 0.8844  val_loss 0.3110  133s
  epoch  3  train_loss 0.1218  val_acc 0.9284  val_loss 0.1758  139s *
  epoch  4  train_loss 0.1052  val_acc 0.9296  val_loss 0.1675  142s *
  epoch  5  train_loss 0.0891  val_acc 0.9498  val_loss 0.1195  140s *
  epoch  6  train_loss 0.0744  val_acc 0.9590  val_loss 0.1132  134s *
  epoch  7  train_loss 0.0541  val_acc 0.9604  val_loss 0.1140  275s *
  epoch  8  train_loss 0.0342  val_acc 0.9624  val_loss 0.1112  1288s *
  epoch  9  train_loss 0.0146  val_acc 0.9652  val_loss 0.1180  116s *
  epoch 10  train_loss 0.0049  val_acc 0.9678  val_loss 0.1164  119s *
  epoch 11  train_loss 0.0029  val_acc 0.9676  val_loss 0.1172  132s
done in 48.0 min  best clean val_acc 0.9678
```

**Verdict:** kept as the baseline. 96.78% clean validation accuracy, above the
~92.98% the CIFAKE paper reports for its own CNN, so the comparison point is a
fair one and not a strawman. Train loss reaching 0.0029 while val loss flattens
at 0.117 shows the clean training set is memorised by epoch 10 — that
overfitting is itself part of the story about why the model is brittle.

(The 1288s epoch 8 is a machine artefact: memory pressure from a concurrent job
on the same laptop, not anything about the model.)

---

## 02 — robustness harness, and the shape of the baseline's failure

**Hypothesis:** The baseline collapses under high-frequency-destroying
transforms, and the collapse is *asymmetric* — AI images get called real rather
than the reverse, because the evidence being destroyed is evidence of
generation.

**Change:** `src/eval_robustness.py`. Reports accuracy at a fixed 0.5 threshold
*and* AUC, plus class-conditional TPR/TNR, because those three together
distinguish "the signal is gone" from "the threshold moved".

**Command:**
```
python -m src.eval_robustness --checkpoints baseline --limit 2000
```

**Result:** (2,000-image stratified subsample, run as a harness check before
committing to the full 20k evaluation)
```
=== baseline (epoch 10, clean val_acc 0.9678, 1,562,353 params) ===
  clean                acc 0.9655  auc 0.9954  tpr 0.9670  tnr 0.9640
  jpeg_q=90            acc 0.9625  auc 0.9945  tpr 0.9640  tnr 0.9610
  jpeg_q=70            acc 0.9670  auc 0.9945  tpr 0.9780  tnr 0.9560
  jpeg_q=50            acc 0.9465  auc 0.9871  tpr 0.9500  tnr 0.9430
  jpeg_q=30            acc 0.8855  auc 0.9793  tpr 0.7870  tnr 0.9840
  blur_sigma=0.5       acc 0.8195  auc 0.9821  tpr 0.6420  tnr 0.9970
  blur_sigma=1.0       acc 0.6205  auc 0.7849  tpr 0.2620  tnr 0.9790
  blur_sigma=2.0       acc 0.6305  auc 0.7590  tpr 0.9200  tnr 0.3410
  resize_0.5x          acc 0.6285  auc 0.7865  tpr 0.2860  tnr 0.9710
  resize_0.25x         acc 0.6390  auc 0.7502  tpr 0.8830  tnr 0.3950
  noise_sigma=0.02     acc 0.9580  auc 0.9941  tpr 0.9510  tnr 0.9650
  noise_sigma=0.05     acc 0.9170  auc 0.9750  tpr 0.9220  tnr 0.9120
  noise_sigma=0.10     acc 0.7485  auc 0.8839  tpr 0.9300  tnr 0.5670
  jitter_+/-20%        acc 0.9645  auc 0.9927  tpr 0.9620  tnr 0.9670
  crop_80%             acc 0.7280  auc 0.9552  tpr 0.4590  tnr 0.9970
  -> clean 0.9655 | mean transformed 0.8154 | worst 0.6205 | mean drop 15.0 pts
```

**Verdict:** hypothesis confirmed, with more structure than expected. Three
findings that shaped everything after:

1. **The predicted asymmetry is there, and past a threshold it reverses.** At
   blur sigma=0.5 the model gets TPR 0.642 against TNR 0.997 — it still
   recognises real images almost perfectly but has stopped recognising AI ones,
   exactly the signature of a detector whose evidence has been low-pass filtered
   away. At sigma=2.0 it flips to TPR 0.920 / TNR 0.341: past a certain
   severity, blur makes *real* images look synthetic, because unnatural
   smoothness is itself the cue the model uses. The same reversal appears
   between resize 0.5x (TPR 0.286) and 0.25x (TPR 0.883). A single averaged
   "robust accuracy" number would have hidden this entirely — which is the
   argument for reporting every severity separately.
2. **Some of the loss is calibration, not signal.** Centre crop 80% drops
   accuracy to 72.80% while AUC holds at 95.52 — the ranking is nearly intact
   and the scores have merely shifted across the 0.5 threshold. That failure is
   recoverable by re-thresholding; the blur sigma=1.0 failure (AUC 78.49) is
   not. This is why the table reports both metrics, and it is quantified in the
   error analysis.
3. **JPEG hurts far less than expected** (88.55% at q=30, versus 62.05% for blur
   sigma=1.0). CIFAKE is *already* stored as JPEG, so every condition here is a
   second-generation re-encode and the most fragile artefacts are gone before
   training even starts.

**Bugs found and fixed:**
- `--limit` took a head slice, but the test parquet is class-ordered, so the
  subsample was single-class and every AUC came back `nan`. Now stratified with
  a fixed seed. The `nan` was the only reason this was noticed; a silently
  class-imbalanced subsample would have produced plausible-looking but
  meaningless accuracies.
- `Condition` held a lambda, which cannot be pickled to DataLoader workers
  (macOS spawns rather than forks them). Replaced with module-level functions
  plus a kwargs dict.

**AI workflow note:** Claude Code proposed reporting a single mean "robust
accuracy". Reporting per-severity instead is what exposed the TPR/TNR reversal,
which is the most interesting finding in the project.

---

## 03 — robust: same model, transforms as training augmentation

**Hypothesis:** Applying the transform suite as training-time augmentation
recovers most of the cliff measured in run 02. Cost should be a small amount of
clean accuracy — the model now has to fit a much wider input distribution with
the same 1.56M parameters.

**Change:** exactly one variable against run 01: `--aug robust`. Same
architecture, seed, schedule, split, epoch count. Augmentation samples
*continuous* severity ranges spanning the evaluation points (JPEG q in [30,95],
blur sigma in [0.3,2.0], etc.) rather than the evaluation severities themselves,
and chains up to two transforms per image.

**Command:**
```
python -m src.train --aug robust --tag robust --epochs 12 --workers 4
```

**Result:**
```
tag=robust  aug=robust  exclude=-  device=mps
train=95,000  val=5,000  params=1,562,353 (0.001562B, limit 2B)
  epoch  0  train_loss 0.4049  val_acc 0.8744  val_loss 0.2921  148s *
  epoch  1  train_loss 0.3102  val_acc 0.8922  val_loss 0.2507  279s *
  epoch  2  train_loss 0.2708  val_acc 0.9182  val_loss 0.1949  151s *
  epoch  3  train_loss 0.2324  val_acc 0.9054  val_loss 0.2314  165s
  epoch  4  train_loss 0.2052  val_acc 0.9484  val_loss 0.1311  162s *
  epoch  5  train_loss 0.1828  val_acc 0.9488  val_loss 0.1273  159s *
  epoch  6  train_loss 0.1577  val_acc 0.9560  val_loss 0.1116  170s *
  epoch  7  train_loss 0.1293  val_acc 0.9534  val_loss 0.1193  150s
  epoch  8  train_loss 0.0931  val_acc 0.9516  val_loss 0.1335  150s
  epoch  9  train_loss 0.0517  val_acc 0.9606  val_loss 0.1295  150s *
  epoch 10  train_loss 0.0240  val_acc 0.9582  val_loss 0.1321  154s
  epoch 11  train_loss 0.0140  val_acc 0.9552  val_loss 0.1407  156s
done in 33.2 min  best clean val_acc 0.9606
```

**Verdict:** kept. 96.06% clean validation accuracy against the baseline's
96.78% — the predicted small cost, 0.72 points. Train loss stays an order of
magnitude higher than the baseline's throughout (0.014 vs 0.0029 at the end),
which is the augmentation doing its job: the model can no longer memorise the
training set because it never sees the same image twice.

---

## 04 — the headline comparison, full 20k test split

**Hypothesis:** The robust model's accuracy cliff is substantially smaller than
the baseline's, and the class-conditional asymmetry from run 02 is reduced.

**Command:**
```
python -m src.eval_robustness --checkpoints baseline robust --save-scores
python -m src.error_analysis
```

**Result:** (full table in `docs/robustness.md`; raw in `results/eval_log.txt`)
```
- baseline: clean 96.69%, mean over transformed conditions 81.04%,
            worst 61.32%, mean drop 15.65 pts.
- robust:   clean 95.58%, mean over transformed conditions 93.06%,
            worst 87.94%, mean drop  2.52 pts.

| Condition        | baseline acc | robust acc | Delta  |
| clean            |    96.69     |   95.58    |  -1.11 |
| jpeg q=30        |    87.91     |   92.69    |  +4.77 |
| blur sigma=0.5   |    80.53     |   94.65    | +14.12 |
| blur sigma=1.0   |    62.11     |   92.59    | +30.47 |
| blur sigma=2.0   |    61.32     |   89.39    | +28.07 |
| resize 0.5x      |    62.61     |   92.08    | +29.47 |
| resize 0.25x     |    63.06     |   87.94    | +24.89 |
| noise sigma=0.10 |    75.56     |   91.42    | +15.85 |
| jitter +/-20%    |    95.52     |   94.45    |  -1.07 |
| crop 80%         |    71.74     |   92.05    | +20.31 |
```

**Verdict:** headline result. The mean drop from clean falls from 15.65 points to
2.52 — a 6.2x reduction — for 1.11 points of clean accuracy. Worst-case accuracy
over all conditions rises from 61.32% to 87.94%.

**Two findings beyond the headline:**

1. **Augmentation fixed calibration, not just accuracy.** Optimally
   re-thresholding each condition recovers up to +16.10 points for the baseline
   (centre crop) but at most +0.35 points for the robust model in any condition.
   The robust model's confidence means the same thing on clean and degraded
   inputs. This was not the hypothesis and is arguably more useful than the
   accuracy gain, since a deployed system picks one threshold and keeps it.
2. **Robustness is not free at the operating point that matters.** At a fixed 1%
   false-positive rate the baseline is slightly *better* on clean images (90.31
   vs 87.07 TPR) — but under blur sigma=2.0 it catches 12.76% of AI images
   against the robust model's 58.79%. The robust model's clean false-positive
   rate is also worse (5.30% vs 3.25%). The honest summary is a trade, not a
   free win, and it is reported that way.

---

## 05 — full-resolution, out-of-distribution check

**Hypothesis:** CIFAKE is 32x32, where "resize 0.25x" is an 8x8 thumbnail and an
80% centre crop is 25 pixels wide. If the robustness result is real rather than
an artefact of that resolution, the ordering (robust > baseline under transforms)
should survive on genuinely full-resolution images. Absolute performance is
expected to drop sharply, because the test images also come from different
generators and a different source dataset.

**Change:** `src/fetch_fullres.py` pulls 600 SID_Set validation images (300 real
from OpenImages, 300 fully synthetic) at ~1024px via the HuggingFace
datasets-server row API, avoiding the 140GB full download. `src/eval_fullres.py`
applies the same transform suite at native resolution and runs the 32x32 model
by tiling into patches.

**Command:**
```
python -m src.fetch_fullres --per-class 300
python -m src.eval_fullres --checkpoints baseline robust --modes patch resize --max-patches 64
```

**Performance note:** the first attempt extracted patches with one `PIL.crop` per
patch, which took ~2.5 min per condition (2.5 h for the full grid). Replaced with
a single reshape over the whole array — bit-identical output, 4x faster on
extraction — which also makes `predict.py` usable on large directories.

**Result:** the hypothesis holds in ordering and fails in magnitude, which is
roughly what was expected. AUC, patch-inference mode:

| | Clean | Mean transformed | Worst transformed |
|---|---|---|---|
| Baseline | 77.17 | 64.20 | 44.20 |
| Robust | 74.40 | **67.54** | **55.63** |

The ordering survives: augmentation improves mean transformed AUC by 3.34 points
and worst-case by 11.43, for 2.77 points of clean AUC — the same trade as CIFAKE,
same sign, smaller magnitude. Per-family the shape also matches: JPEG is nearly
free (73–79 across all four qualities), blur/downscale/noise are where both
models fail.

**The dominant effect is not robustness, though.** Clean AUC falls from 99.48 to
77.17 with no transform applied at all. That 22.31-point gap is generator and
resolution transfer, and it is the same order of magnitude as the worst single
transform effect measured anywhere in this project (25.65 points of AUC, resize
0.25x on CIFAKE) — but it is paid up front, before any transform. Reporting the
robustness improvement without that caveat would be misleading, so
`docs/robustness.md` states it before the table.

**Unexpected result — below-chance AUC.** The baseline scores 44.20 at noise
sigma=0.10 and 46.22 at noise sigma=0.05: worse than a coin flip, meaning the
ranking is systematically *inverted*, not lost. A model that had merely run out
of signal would sit at 50. This is the same reversal the class-conditional CIFAKE
table shows via TPR/TNR, reproduced at real resolution. Augmentation removes it
everywhere — the robust model's minimum is 55.63.

**Unexpected result — resize-mode's flatness is a floor, not robustness.** The
robust model in resize mode varies by **1.74 AUC points across all 15
conditions** (61.82–63.55). On a single averaged "robustness" metric it is the
best configuration in the project. It is in fact the worst-informed one:
downscaling to 32x32 has already destroyed the high-frequency evidence that blur,
noise and downscaling destroy, so the transforms have nothing left to take. Patch
mode beats it by 12.11 AUC on clean input. This is the strongest single argument
against reporting robustness as one averaged number, and it was found only
because both inference modes were run across the whole grid rather than just on
clean images.

**Follow-on:** counting conditions where patch beats resize, the baseline wins 8
of 15 and the robust model 11 of 15. Augmentation extends the range of input
degradation over which native-resolution inference is worth its cost. The
crossover is real, so `predict.py` keeps `--mode resize` available rather than
hard-coding the patch path.

Full table: `docs/robustness.md`, `results/fullres_table.md`,
`results/fullres.json`.

