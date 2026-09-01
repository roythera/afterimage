# Afterimage

**Robust detection of AI-generated images under real-world transformations.**

## Inspiration

Most AI-image detectors are evaluated on pristine files. No image on a social
platform is pristine. Between the generator and the viewer sits an upload
pipeline that re-encodes, resizes, strips metadata and re-compresses — often
several times, as content is screenshotted and reshared.

The thing that made the problem click was realising the failure is not incidental
but *mechanical*. Detectors key on high-frequency generator fingerprints: the
periodic traces left by upsampling layers, which live in the top octave of the
spectrum. JPEG quantisation and downscaling are low-pass operations. They attack
precisely the band the detector depends on. A detector and a compression codec
are, in a real sense, fighting over the same coefficients — and the codec runs
last.

That suggested a project that measures the collapse rather than one that reports
another clean-benchmark accuracy.

## What I built

The whole project is one controlled comparison, and everything else exists to
make it trustworthy.

**The measuring instrument came first.** Before training anything, I built the
transform suite — JPEG at $q \in \{90,70,50,30\}$, Gaussian blur at
$\sigma \in \{0.5,1.0,2.0\}$, downscale-and-restore at $0.5\times$ and
$0.25\times$, Gaussian noise at $\sigma \in \{0.02,0.05,0.10\}$, $\pm 20\%$
colour jitter, and an 80% centre crop — plus a self-test that verifies it. Every
claim in the project is a claim about this code, so it is checked rather than
assumed. Two checks turned out to be load-bearing:

- **Noise $\sigma$ is in normalised $[0,1]$ units**, applied before
  normalisation and clamped back. Measured RMSE against the clean image is
  $0.0198 / 0.0490 / 0.0965$ for requested $\sigma$ of $0.02/0.05/0.10$ — within
  2%. Interpreting $\sigma$ in 0–255 units instead would have made every
  perturbation $255\times$ too large and every noise result meaningless.
- **JPEG is a real encode/decode round trip**, not a blur approximation.
  Re-encoding an already-$q50$ image at $q50$ changes it by RMSE $0.00000$, which
  only a real codec does. Since JPEG's high-frequency quantisation is the entire
  phenomenon under study, approximating it would have deleted the subject matter.

**Then two models, one variable.** Identical architecture, seed, schedule and
data split, trained twice — once on clean images, once with the transform suite
as training-time augmentation — and evaluated across all 15 conditions on the
same held-out 20,000-image CIFAKE test split. Stochastic conditions are seeded
per image index, so both models are scored on byte-identical inputs and the
comparison is exact rather than statistical.

The architecture is a 1,562,353-parameter residual CNN trained from scratch, not
a fine-tuned ImageNet backbone. Pretrained networks open with a stride-2
convolution and stride-2 pool that discard three quarters of the spatial signal
before the first block — they throw away the evidence. The stem here is stride-1
for that reason, and the network is fully convolutional with global average
pooling so it accepts any input size.

Augmentation samples *continuous* severity ranges that span the test severities
without containing them ($q \sim \mathcal{U}[30,95]$, blur
$\sigma \sim \mathcal{U}[0.3,2.0]$, and so on), chaining up to two transforms per
image. Training on exactly the test settings would have inflated the table
without demonstrating anything; sampling the interval makes every evaluation
point an interpolation rather than a memorised case.

**Result:** the mean accuracy drop across transformed conditions falls from
**15.65 points to 2.52** — a $6.2\times$ reduction — and worst-case accuracy
rises from 61.32% to 87.94%, for 1.11 points of clean accuracy.

## What I learned

**Averaging destroys the most interesting structure.** The baseline's failure is
one-sided at moderate severity — at blur $\sigma=0.5$ it scores TPR $0.613$
against TNR $0.998$, still recognising real images almost perfectly while
generated ones slip past. But at $\sigma=2.0$ it **reverses** to TPR $0.902$ /
TNR $0.324$: heavy smoothing makes *authentic* photographs look synthetic,
because unnatural smoothness is itself a cue the model learned. Averaging the
three blur severities yields ~68%, which reads as uniform mild degradation and
describes nothing that is actually happening.

**Accuracy and AUC answer different questions, and the gap between them is
diagnostic.** Under an 80% centre crop the baseline scores 71.74% accuracy but
94.91 AUC — the ranking survived while the score distribution slid across the
0.5 threshold. That failure is *calibration*, recoverable by re-thresholding
(+16.10 points), not lost signal. Under blur $\sigma=1.0$ only +8.27 points come
back, so that one is genuine. An accuracy-only table would have called these the
same failure. For the robust model the recoverable gap is at most +0.35 points in
any condition — augmentation did not just raise accuracy, it made the model's
confidence *mean the same thing* across transforms, which is what a deployed
system running one fixed threshold actually needs.

**The most useful finding was a configuration that looks best and is worst.** At
full resolution, applying the $32\times32$ detector by downscaling the image
gives a model that varies by only **1.74 AUC points across all 15 conditions** —
by far the flattest robustness profile in the project. It is also the least
informative one: downscaling has *already* destroyed the high-frequency evidence
that blur, noise and compression destroy, so the transforms have nothing left to
take. Its flatness is a floor, not resilience, and patch-based inference beats it
by 12.11 AUC on clean images. Robustness measured without regard to absolute
performance can be maximised by making the model useless.

**The detector is not measuring authenticity.** The error analysis puts both
failure classes on a single axis — scene texture density, measured as mean
absolute Laplacian $\frac{1}{HW}\sum |\nabla^2 I|$. False positives (real photos
called AI) are isolated subjects on smooth backgrounds: aircraft against blank
sky, boats on flat water, mean energy $0.1257$ against a test-set mean of
$0.1325$. False negatives are busy textured scenes — foliage, fur, clutter — at
$0.1363$, *above* the mean. Low texture reads as synthetic; high texture masks
the fingerprint. That is the same axis blur and downscaling move images along,
which is why they break the model, and why the breakage reverses. The detector is
measuring texture statistics that correlate with authenticity, and augmentation
works by widening the range of texture statistics associated with each class, not
by teaching a new cue.

## Challenges

**Silent numerical corruption.** Computing channel statistics over
$\approx 10^8$ pixels in float32 returned $(0.1638, 0.1638, 0.1638)$ — suspicious
because the three channels were identical. Once the accumulator passes $2^{25}$
the ULP is $4$, so adding a value near $0.4$ is a no-op and the sum silently
stops growing. In float64 the answer is $(0.4720, 0.4629, 0.4178)$, consistent
with CIFAR-10. Nothing crashed. I caught it only because I had an expectation to
check the output against, which became the working habit for the rest of the
build.

**A bug that announced itself, and the one that wouldn't have.** Evaluating on a
subsample produced `nan` AUC. The test parquet is class-ordered, so a head slice
was single-class and AUC was undefined. The fix was stratified sampling — but the
lesson was the near-miss: a subsample that was merely *imbalanced* rather than
single-class would have produced plausible, wrong accuracies and no error at all.

**Honest evaluation of a 32×32 model at 1024px.** CIFAKE is $32\times32$, where
"resize $0.25\times$" is an $8\times8$ thumbnail and an 80% centre crop is 25
pixels wide — several conditions are close to degenerate. So I validated on 600
full-resolution SID_Set images, pulled via the HuggingFace datasets-server row
API to avoid a 140GB download. Applying the model means tiling at native
resolution and averaging per-patch *logits* before the sigmoid,
$p = \sigma\!\left(\frac{1}{N}\sum_i z_i\right)$, since averaging probabilities
lets a few saturated patches dominate. The first implementation cropped patches
one at a time and projected to 2.5 hours; rewriting extraction as a single
reshape over the array made it $4\times$ faster with bit-identical output
($\max|\Delta| = 0$).

**The result I did not want.** At full resolution the robustness finding
transfers — augmentation lifts mean transformed AUC from 64.20 to 67.54 and
worst-case from 44.20 to 55.63 — but clean AUC falls from 99.48 to 77.17 *with no
transform applied at all*. That 22.31-point gap is generator and resolution
transfer, comparable in size to the worst post-processing effect measured
anywhere in the project. Robustness to post-processing and generalisation to
unseen generators are separate problems, and this solves only the first. The
baseline also drops *below chance* under heavy noise (44.20 AUC), meaning its
ranking is systematically inverted rather than merely destroyed — the same
reversal, at real resolution.

Reporting that plainly was the right call and is the part of the project I would
defend hardest. It is also the clearest signpost for what comes next: train on
patches from multiple generators at full resolution, rather than transferring a
$32\times32$ model.
