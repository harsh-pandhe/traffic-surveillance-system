# Project Completion Plan — All Phases

Adaptive Spatio-Temporal Traffic Surveillance
Granular Helmet Compliance · Multi-Frame Wheel-Count Classification · Real-Time Risk Indexing

This plan takes the project from its current verified state to full delivery of
all seven contract deliverables.

---

## 0. Verified current state

Assessed by running the code, not by reading notes.

| Phase | Status | Evidence |
|---|---|---|
| 1 — Dataset & adaptive preprocessing | **Done** | Scene classifier 79.1% 4-class; CLAHE/DCP verified on real DAWN/ExDark; builders run; `docs/Phase1_Report.pdf` |
| 2 — Model training & helmet compliance | **Models done, hardening in progress** | Wheel 3-arm: SmallCNN 0.751 / RTMDet-CSPNeXt 0.683 / YOLOv8-cls 0.867. Helmet mAP@50 0.731. Two integration bugs found and fixed on `phase-2-hardening` |
| 3 — Tracking, ReID, analytics | **Not started** | Modules are scaffold only; `deep_sort_realtime`, `torchreid`, `deepface` not installed; never run on real data |
| 4 — Optimization & documentation | **Partial** | Prune + INT8 run (0.93→0.92 MB); ONNX export fails (`No module named 'onnxscript'`); no before/after benchmark |

### Known gaps carried forward

1. **No video data.** Phases 1–2 used still images. Phase 3 (tracking, multi-frame
   voting, occlusion handling) is impossible without video. **This is the single
   largest blocker.**
2. **No multi-camera data.** Cross-camera ReID needs overlapping/sequential camera
   feeds (CityFlow / VeRi-776).
3. **Granular helmet sub-classes absent.** Contract lists strap-unfastened,
   helmet-on-handlebar, helmet-on-arm. Public data for these does not exist
   (verified by search). Current model uses the AI City Track-5 taxonomy
   (driver/passenger × helmet/no-helmet + bike).
4. **Passenger-helmet AP low** (0.56–0.58) — few instances (issue #8).

---

## 1. Data acquisition (unblocks Phase 3)

Do this first; everything in Phase 3 depends on it.

| Need | Source | Action |
|---|---|---|
| Traffic **video** with riders | AI City Challenge Track 5 (100 train videos, 10 fps, 1080p) | Submit dataset request at aicitychallenge.org — **requires user to sign the data agreement** |
| Fallback video | Public dashcam / traffic footage (CC-licensed), or phone-recorded local traffic | Download 3–5 clips, 20–60 s each |
| Multi-camera ReID | VeRi-776 (776 vehicles, 20 cameras) or CityFlow | Request/download |
| Granular helmet classes | No public source | Decide: (a) document as out-of-scope, (b) annotate ~200 images locally with LabelImg/CVAT |

**Deliverable:** `data/raw/video/` populated; `DATASETS.md` updated with sources,
licenses, and placement.

---

## 2. Phase 2 — close out

1. Merge `phase-2-hardening` (compliance-inversion fix + wheel-weights wiring) via PR.
2. Rebuild Phase 2 report PDF with the 3-arm benchmark and the pretraining caveat.
3. Address issue #8: rebalance or augment passenger-helmet classes; retrain; report delta.
4. Decide granular-class strategy and record the decision in `docs/PHASE2.md`.
5. Close issues #7, #8.

**Deliverable:** `docs/Phase2_Report.pdf` (final), all Phase 2 issues closed.

---

## 3. Phase 3 — tracking, ReID, analytics

**Prerequisite:** video data from step 1.

### 3.1 Environment
```
pip install deep-sort-realtime torchreid deepface albumentations
```
Pin versions in `requirements.txt`. Verify each import replaces its fallback
(the pipeline currently prints a fallback warning for all three).

### 3.2 Tracking
- Wire DeepSORT properly; confirm stable IDs across ≥100 frames.
- Metric: ID switches, MOTA/IDF1 on a hand-labelled clip (or qualitative if no GT).

### 3.3 Multi-frame occlusion voting
- Feed per-frame wheel-count predictions into the existing `Track.votes` buffer.
- **Experiment to run:** single-frame accuracy vs N-frame majority-vote accuracy
  under occlusion. This is a headline research result — quantify the gain.

### 3.4 Cross-camera ReID
- OSNet embeddings + gallery matching on VeRi-776.
- Metric: Rank-1 accuracy, mAP.

### 3.5 Demographics
- DeepFace age/gender on exposed-face rider crops.
- Report coverage (% of riders with usable face crop) and confidence distribution.

### 3.6 Risk indexing
- Run the weighted risk equation on real video; produce annotated output with
  LOW/MEDIUM/HIGH banners.
- Validate the overload rule (rider count > 2) using the driver/passenger classes.

**Deliverables:** annotated demo video, `docs/Phase3_Report.pdf`,
`results/phase3/*.json`, occlusion-voting ablation table.

---

## 4. Phase 4 — optimization & benchmarking

### 4.1 Fix the export path
```
pip install onnx onnxruntime onnxscript
```
Then export the **real trained models** (helmet YOLOv8, wheel YOLOv8-cls, CSPNeXt),
not just the demo SmallCNN.

### 4.2 Optimization matrix
Run each model in four configurations:

| Config | Description |
|---|---|
| Baseline | FP32 PyTorch |
| Pruned | 30% structured sparsity |
| INT8 | Dynamic quantization |
| ONNX | ONNX Runtime, INT8 |

### 4.3 Metrics per configuration
mAP@50 / accuracy · inference latency (ms) · FPS · peak RAM (MB) · CPU utilisation ·
model size (MB). Use `utils/metrics.py`; report accuracy **retention**, not just speed.

**Deliverables:** `results/phase4/benchmark.json`, before/after comparison charts,
optimized weights in `weights/`, `docs/Phase4_Report.pdf`.

---

## 5. End products

### 5.1 Repository
```
traffic_surveillance/
├── README.md                 # install + quickstart + results summary
├── docs/
│   ├── HOW_TO_RUN.md         # step-by-step execution guide
│   ├── PHASE1..4.md          # per-phase technical notes
│   ├── Phase1..4_Report.pdf  # per-milestone demo reports
│   ├── Final_Report.pdf      # 25-page consolidated document
│   └── paper/                # research paper sources
├── src/, scripts/, utils/, config/
├── results/phase1..4/*.json  # all metrics, reproducible
└── weights/                  # PyTorch + ONNX (or download script if large)
```
Clean history, all issues closed, tagged release `v1.0`.

### 5.2 How-to-run document (`docs/HOW_TO_RUN.md`)
1. Prerequisites (Python 3.10+, CPU-only note)
2. Installation (venv, requirements, verifying optional deps)
3. Dataset acquisition and placement (per dataset, with links)
4. Running each phase — exact commands, expected output, runtime
5. Running the full pipeline on a video
6. Reproducing every table and figure in the reports
7. Troubleshooting (known dependency conflicts, fallback behaviour)

### 5.3 Final document — 25 pages

| Pages | Section |
|---|---|
| 1 | Title, author, declaration |
| 2 | Abstract + keywords |
| 3 | Table of contents, figures, tables |
| 4–5 | Introduction — motivation, problem statement, objectives, contributions |
| 6–8 | Literature review — helmet detection, wheel classification, tracking/ReID, adaptive preprocessing |
| 9–10 | System architecture — block diagram, module breakdown, data flow |
| 11–13 | Phase 1 — scene classification, adaptive enhancement, results |
| 14–17 | Phase 2 — datasets, CNN vs RTMDet vs YOLO benchmark, helmet detector, results |
| 18–20 | Phase 3 — tracking, occlusion voting ablation, ReID, demographics, risk index |
| 21–22 | Phase 4 — pruning, quantization, ONNX, before/after benchmark |
| 23 | Discussion — limitations, threats to validity, ethical considerations |
| 24 | Conclusion and future work |
| 25 | References + appendix (repo link, reproduction commands) |

Generated by `scripts/build_final_report.py` from the same `results/*.json` used
by the phase reports, so numbers cannot drift between documents.

### 5.4 Research paper

Generic conference/journal format now; retarget when the venue is chosen.

**Working title:** *Adaptive Spatio-Temporal Traffic Surveillance for Granular
Helmet-Compliance and Multi-Frame Vehicle Classification on Commodity CPUs*

**Structure (8–10 pages, two-column):**
1. Abstract
2. Introduction — enforcement gap, why CPU-only matters for deployment
3. Related work
4. Proposed method — adaptive preprocessing → detection → tracking → risk index
5. Experimental setup — datasets, splits, hardware, protocol
6. Results — scene classification, 3-way architecture benchmark, helmet mAP,
   occlusion-voting ablation, optimization trade-off
7. Discussion and limitations
8. Conclusion
9. References

**Claimed contributions (must be defensible by our numbers):**
- C1: Scene-adaptive preprocessing with a learned condition classifier
  (rule baseline 0.49 → learned 0.79 on real adverse-weather data).
- C2: Three-way CPU-oriented architecture benchmark for wheel-count
  classification with an explicit pretraining-fairness analysis.
- C3: Multi-frame voting for occlusion-robust classification — *pending Phase 3
  ablation; this is the strongest novelty claim and must be measured.*
- C4: End-to-end real-time risk indexing on commodity CPU with a quantified
  accuracy/latency trade-off.

**Honesty constraints:** report the pretraining asymmetry; state that granular
helmet sub-classes were unavailable; do not claim multi-camera results without
running VeRi-776.

---

## 6. Sequencing

| Step | Work | Blocked by |
|---|---|---|
| 1 | Acquire video + ReID data | user action (AI City agreement) |
| 2 | Close out Phase 2 (merge, rebuild PDF, issue #8) | — |
| 3 | Phase 3 implementation + experiments | step 1 |
| 4 | Phase 4 optimization + benchmark | step 3 (needs final models) |
| 5 | `HOW_TO_RUN.md` + README polish | step 4 |
| 6 | 25-page final report | steps 2–4 (all numbers) |
| 7 | Research paper | step 6 |
| 8 | Tag `v1.0`, final handover | all |

Steps 2 and 4-prep can proceed in parallel with step 1.

---

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| AI City data request denied/slow | Phase 3 blocked | Fall back to public dashcam clips or self-recorded footage |
| No multi-camera data | C-contribution weakened | Report single-camera ReID only; state the limitation |
| Granular helmet classes unavailable | Contract shortfall | Document explicitly; offer local annotation as scoped add-on |
| mmdetection never installable | RTMDet arm is a reimplementation | Already documented; optionally retry in an isolated Python 3.11 venv |
| CPU training time | Schedule slip | Cap epochs, cache datasets, run overnight in background |
