# Phase 3 — Multi-Frame Tracking & Analytics

**Days 13–17 milestone.** Deliverables: DeepSORT temporal tracking, improved
classification during occlusion via multi-frame information, spatio-temporal
analytics and risk assessment, and a working demo.

## Components

| Component | File | Role |
|---|---|---|
| Tracking | `src/tracking/deepsort_tracker.py` | DeepSORT with temporal wheel-count voting |
| Cross-camera ReID | `src/tracking/multi_camera_reid.py` | OSNet embeddings + time-limited gallery |
| UA-DETRAC loader | `src/tracking/detrac_loader.py` | Tracks + per-target occlusion ratio |
| HELMET loader | `src/tracking/helmet_video_loader.py` | Motorcycle clips, rider count, per-seat helmet use |
| Voting ablation | `scripts/run_occlusion_ablation.py` | N-sweep, flip-rate + balanced accuracy |
| Risk indexing | `src/analytics/risk_indexer.py` | Weighted hazard score, LOW/MEDIUM/HIGH |

## Backends: verifying they are actually running

Every heavy backend has a graceful fallback, which means a broken dependency
looks like success. Both were silently degraded before this phase:

| Backend | Was | Root cause | Now |
|---|---|---|---|
| DeepSORT | fallback IoU tracker | `openmim` downgraded setuptools; on Python 3.12 `pkg_resources` raises `pkgutil.ImpImporter` | `backend: deepsort` |
| OSNet ReID | colour histogram | torchreid ships under two layouts (`torchreid.reid.utils` vs `torchreid.utils`) | `backend: osnet`, 512-d |

Always assert the backend rather than assuming the run succeeded — see
`docs/HOW_TO_RUN.md` §2.

## Multi-frame voting ablation

**Data:** UA-DETRAC, 8 sequences, 22,480 classified crops. Tracks are grouped by
ground-truth id so the measurement isolates voting from tracker error.

### Why the headline metric is flip-rate, not accuracy

Two problems with accuracy on this dataset, both discovered during the run:

1. **Class imbalance.** UA-DETRAC is ~97% cars, so a model that only ever says
   "4-wheeler" scores 0.93 raw accuracy. Balanced accuracy (macro recall) fixes
   this. Subsampling tracks to equal class counts was tried first and was
   *worse* — it collapsed the evaluation to 4 tracks per class and emptied the
   heavy-occlusion band completely.
2. **Bands are not class-comparable.** The occlusion bands contain different
   class mixes:

   | Band | 4-wheeler | 6+ wheeler |
   |---|---|---|
   | none | 15,546 | 335 |
   | light | 2,404 | 109 |
   | medium | 1,981 | 6 |
   | heavy | 2,340 | **0** |

   The heavy band is entirely single-class, so its apparent 0.99 accuracy only
   means "the model says car when everything is a car". Per-band *accuracy*
   comparisons are therefore invalid here. Flip-rate does not depend on class
   balance and remains valid.

### Results

Prediction flip-rate (fraction of frame-to-frame label changes; lower is more stable):

| Band | N=1 | N=5 | N=10 | N=15 | N=30 |
|---|---|---|---|---|---|
| none | 0.042 | 0.014 | 0.008 | 0.005 | 0.004 |
| light | 0.026 | 0.007 | 0.004 | 0.004 | 0.002 |
| medium | 0.016 | 0.002 | 0.002 | 0.001 | 0.001 |
| heavy | 0.009 | 0.002 | 0.002 | 0.002 | 0.002 |
| **overall** | **0.034** | 0.011 | 0.007 | 0.004 | **0.003** |

Balanced accuracy over the same runs: 0.626 (N=1) → **0.642** (N=15).

**Interpretation.** Voting cuts label changes by **91.4%**, monotonically in
every occlusion band, while balanced accuracy rises slightly rather than
degrading. Temporal smoothing therefore buys output stability at no accuracy
cost. This matters for automated enforcement: a flickering class label means a
wrongly issued ticket. Part of the flip reduction is mechanical — averaging
smooths any signal — so the honest claim is *stability gained without accuracy
loss*, not that voting is a large accuracy win.

## Motorcycle data: the HELMET dataset

UA-DETRAC cannot support this project's core subject matter — it contains **no
two- or three-wheelers at all**. The HELMET dataset (910 clips × 100 frames,
tracked motorcycles, per-rider helmet annotation) supplies them.

Labels are compositional, encoding every occupant and their helmet state
(`DHelmetP1NoHelmetP2NoHelmet`), so one parse yields the 2-wheeler class, rider
occupancy, and per-seat compliance.

Statistics over all 910 clips / 283,377 annotations / 10,006 tracks:

| Riders per motorcycle | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Count | 163,435 | 101,906 | 15,854 | 2,094 | 88 |

- ≥1 helmet violation: **111,244 (39%)**
- Triple riding (>2 riders): **18,036 (6.36%)**
- Bare-headed by seat: driver 90,342 · P1 55,237 · P2 9,906 · P0 6,608 · P3 361

The loader recovers exactly **10,006** unique tracks, matching the figure
published with the dataset — a correctness check on the parsing. The 18,036
triple-riding instances give the risk indexer's overload rule real ground truth.

## Risk indexer validated against real ground truth

`scripts/validate_risk_indexer.py` feeds every one of the 10,006 real HELMET
tracks' true occupancy and helmet-use into `RiskIndexer` (no detector run —
this validates the *rule logic* against real-world prevalence, independent of
detection accuracy).

| Rule | Trigger rate on real data |
|---|---|
| overloaded (>2 riders) | 7.0% |
| helmet_misuse (≥1 bare head) | 34.2% |

Boundary check passes: 2 riders → not overloaded, 3 riders → overloaded.

**Finding: HIGH risk level never triggers on real motorcycle data (0.0%).**
`w2 (helmet_misuse) + w3 (overloaded) = 2.5 + 1.5 = 4.0`, but the HIGH threshold
is 6.0 — even the worst realistic motorcycle case (unhelmeted **and**
triple-riding) tops out at MEDIUM. Resulting distribution: LOW 93.5%,
MEDIUM 6.5%, HIGH 0.0%. Whether an overloaded, unhelmeted motorcycle should
register as HIGH is a policy call, not a code bug — flagged here rather than
silently re-tuning the weights to produce a more dramatic-looking distribution.
Wheel-count violation (w1=2.0) or wrong-way (w4=3.0) are what actually reach
HIGH; motorcycles rarely trigger either.

> **Image download deliberately not pursued.** The HELMET image archives are
> ~29 GB across 7 throttled OSF parts (one part stalled at 118 MB under both
> curl and wget). Before spending hours on that, a learning-curve gate
> (`scripts/gate_a_learning_curve.py`) retrained the helmet detector on 25/50/
> 75/100% of the existing 368-image train split: test mAP@50 went
> 0.494 → 0.619 → 0.700 → 0.718. The final-quarter gain (+0.017) is below the
> +0.02 threshold, so the curve has **plateaued** — more images would help only
> marginally; the detector's ceiling here is set by label quality and small
> object sizes, not data volume. All 910 HELMET annotation files (no images)
> are used for the risk-indexer validation above and the demographic/occupancy
> statistics; pixel-based analyses use UA-DETRAC. Full result:
> `results/phase0_gates.json`.

## Cross-camera re-identification (VeRi-776)

`scripts/run_reid_benchmark.py` evaluates the trained OSNet backend
(`src/tracking/multi_camera_reid.py`) on VeRi-776 (776 vehicles, 20 real
cameras, 51k images) using the standard protocol: rank gallery images by cosine
distance to each query embedding, excluding gallery images from the query's own
camera (same-camera retrieval is trivial and isn't cross-camera ReID).

| Metric | Value |
|---|---|
| Rank-1 | 0.310 |
| Rank-5 | 0.510 |
| Rank-10 | 0.615 |
| mAP | 0.095 |
| Embed latency | 8.7 ms/image (CPU) |

**Honest interpretation.** These numbers are well below published VeRi-776 SOTA
(fine-tuned ReID models reach Rank-1 ≈ 90%, mAP ≈ 70%). That's expected, not a
bug: `weights/osnet_x0_25.pth` was never supplied, so the backend runs
**off-the-shelf ImageNet-pretrained OSNet** with the classification head
discarded (`torchreid` prints "discarded due to unmatched keys:
classifier.weight, classifier.bias" at load time) — i.e. generic image
features, zero-shot, with no vehicle-ID metric learning at all. Rank-1 31% from
generic features alone confirms the embeddings carry real appearance signal
(chance level for 776 identities is ~0.1%), which is the correct thing to claim
here. Closing the gap to SOTA needs triplet/ID-loss fine-tuning on VeRi's own
37,778-image training split — already downloaded at
`data/raw/VeRi/VeRi/image_train/`, not yet used. Recorded as the clear next
step rather than a currently-claimed result.

See `results/phase3/reid_benchmark.json` for the full output.

## Demographics — a real hardware constraint, worked around

Gate B measured 62% of face-exposed rider detections yield a crop above the
32px minimum (median 38px), which justified installing DeepFace. The install
then surfaced a genuine environment issue: this machine's torch build targets a
newer NVIDIA driver than is installed, and loading a **second** Ultralytics/
torch model in a process that has also imported TensorFlow segfaults —
confirmed by bisecting `SurveillancePipeline` construction step by step
(`HelmetDetector`'s YOLO load is fine; `WheelClassifier`'s second YOLO load is
not, reproducibly, regardless of import order or `CUDA_VISIBLE_DEVICES`).

Rather than ship a pipeline that can crash, or silently drop the demographics
deliverable, `DemographicsEstimator` now detects when torch is already loaded
and queues face crops to `outputs/demographics_queue/` instead of calling
DeepFace in-process. `scripts/run_demographics.py` — which imports only cv2 and
deepface, never torch — processes that queue afterward. Verified end-to-end on
real pipeline output: **19/19** queued crops processed successfully with no
crash. A regression test (`test_demographics_refuses_inprocess_when_torch_loaded`)
guards the crash-prevention check itself.

Per Gate B, age estimates should be read as indicative rather than reliable
(median face crop 38px vs DeepFace's ~64px comfort zone); gender is more robust
at low resolution. See `results/phase3/demographics.json`.

## End-to-end pipeline run

`python main.py --source data/raw/video_demo.mp4 --camera cam_A`

300 frames (UA-DETRAC MVI_20032, 960×540), CPU only:

- **6.84 FPS mean, 9.5 FPS steady-state**
- Annotated output with telemetry HUD, tracks, and risk banner → `outputs/annotated.mp4`

**Honest caveat:** the helmet detector is trained on motorcycle riders, while
UA-DETRAC is car-only highway footage. Helmet detections in this demo are
therefore low-confidence false positives (e.g. a car boxed as "Driver With
Helmet 0.39") and are **not** meaningful. The run proves the pipeline plumbing
and measures CPU throughput; it is not evidence of helmet-detection quality.

## Finding: scene classification is unstable per-frame

On the demo clip the learned scene classifier returns DAY at only **0.42**
confidence (FOG 0.29, RAIN 0.29). Because the label is recomputed per frame, the
chosen enhancement can flicker between frames, producing visible colour shifts.

This is the same instability that voting fixes for wheel-count, so the obvious
remedy is to apply the same temporal smoothing to the scene label. Recorded as
future work rather than claimed as implemented.

## Deliverable checklist

- [x] DeepSORT temporal tracking (verified running, not fallback)
- [x] Multi-frame information used to stabilise classification (voting ablation)
- [x] Spatio-temporal analytics + risk assessment wired end-to-end
- [x] Working demo on real video with measured CPU throughput
- [x] Motorcycle-domain ground truth for occupancy/overload (HELMET)
- [x] Risk-indexer rules validated against real ground truth (10,006 tracks)
- [x] Cross-camera ReID benchmarked on VeRi-776
- [x] Demographics (DeepFace installed; process-isolated to prevent a
      confirmed torch+TensorFlow segfault; verified end-to-end on real output)
