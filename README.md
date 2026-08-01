# Adaptive Spatio-Temporal Traffic Surveillance

Granular Helmet Compliance · Multi-Frame Wheel-Count Classification · Real-Time Risk Indexing.

CPU-optimized, modular Python pipeline. All results below are measured on real
data and tracked in `results/`, not hand-entered — the report PDFs and this
README are generated from the same JSON files, so numbers cannot drift.

## Results at a glance

| Phase | Headline result | Report |
|---|---|---|
| 1 — Scene &amp; preprocessing | Learned scene classifier: **0.79** 4-class accuracy (rules: 0.49) | [Phase1_Report.pdf](docs/Phase1_Report.pdf) |
| 2 — Detection models | Wheel-count: SmallCNN 0.787 / RTMDet-CSPNeXt 0.695 / **YOLOv8-cls 0.861** (leak-free split). Helmet: **mAP@50 0.764** (held-out test) | [Phase2_Report.pdf](docs/Phase2_Report.pdf) |
| 3 — Tracking &amp; analytics | Multi-frame voting: **91.4% fewer prediction flips**. ReID: Rank-1 0.310 (off-the-shelf). Risk rules validated on 10,006 real motorcycle tracks | [Phase3_Report.pdf](docs/Phase3_Report.pdf) |
| 4 — Optimization | Pruning collapses accuracy without fine-tuning (0.31 retention) — fine-tuned recovery restores it (0.98) — then ONNX INT8 shrinks models **3.8–3.9×** at no further cost | [Phase4_Report.pdf](docs/Phase4_Report.pdf) |

Full step-by-step reproduction: **[docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md)**.

## Pipeline (per frame)

| # | Stage | Module | Output |
|---|-------|--------|--------|
| 1 | Scene classify | `src/preprocessing/scene_classifier_ml.py` | DAY / NIGHT / FOG / RAIN |
| 2 | Adaptive enhance | `src/preprocessing/enhancements.py` | CLAHE / DCP dehaze / denoise |
| 3 | Helmet compliance | `src/models/helmet_detector.py` | 7-class YOLOv8 detections |
| 4 | Wheel classify | `src/models/benchmark_classifier.py` | 2/3/4/6+ wheeler (CNN vs RTMDet vs YOLO) |
| 5 | Track + vote | `src/tracking/deepsort_tracker.py` | stable IDs, temporal wheel vote |
| 6 | Cross-cam ReID | `src/tracking/multi_camera_reid.py` | OSNet global IDs |
| 7 | Demographics | `src/analytics/demographics.py` | age group + gender (process-isolated) |
| 8 | Risk index | `src/analytics/risk_indexer.py` | LOW / MEDIUM / HIGH banner |
| 9 | Optimize (offline) | `src/optimization/model_quantizer.py` | prune + fine-tune + INT8 + ONNX |

## Install

```bash
git clone https://github.com/harsh-pandhe/traffic-surveillance-system.git
cd traffic-surveillance-system
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools    # see docs/HOW_TO_RUN.md - do not skip
pip install -r requirements.txt
```

Heavy backends (Ultralytics, deep-sort-realtime, torchreid, DeepFace) have
fallbacks so the pipeline never crashes on a missing dependency — but a
missing backend is easy not to notice. **Always verify explicitly:**

```bash
python -c "
from src.tracking.deepsort_tracker import VehicleTracker
from src.tracking.multi_camera_reid import VehicleReID
print('tracker:', VehicleTracker()._backend)   # want: deepsort
print('reid   :', VehicleReID()._backend)      # want: osnet
"
```

## Datasets

Every dataset used is listed with placement in **[DATASETS.md](DATASETS.md)**:
DAWN, ExDark, COCO (Phase 1), a 7-class helmet set + auto-rickshaw crops
(Phase 2), UA-DETRAC + VeRi-776 + HELMET (Phase 3).

## Weights

Trained weights load from `weights/` (paths in `config/settings.yaml`):

- `helmet_yolov8.pt` — 7-class helmet compliance detector (mAP@50 0.764; ONNX FP32/INT8 export retains 99%, see `docs/PHASE4.md`)
- `wheel_yolov8_cls.pt` / `wheel_cnn.pt` / `wheel_cspnext.pt` — wheel-count models
- `scene_classifier.joblib` — learned scene classifier
- `osnet_x0_25.pth` — OSNet ReID backbone (off-the-shelf; not fine-tuned — see [issue #14](https://github.com/harsh-pandhe/traffic-surveillance-system/issues/14))

Retrain any of these with the scripts in `scripts/` — see
[docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for exact commands and expected output.

## Run

```bash
python main.py --source data/raw/clip.mp4 --camera cam_A   # process a clip
python main.py --source 0 --show                           # live webcam
python main.py --optimize                                  # Phase 4 prune+quantize+export
python scripts/run_demographics.py                         # offline demographics (see below)
```

Annotated video → `outputs/annotated.mp4`.

> **Demographics runs in a separate process.** On hardware where torch's CUDA
> build outpaces the installed driver, loading torch and TensorFlow together
> segfaults — confirmed and worked around here (see `docs/PHASE3.md`). The
> live pipeline queues face crops; `scripts/run_demographics.py` processes
> them afterward. This is a deliberate architectural choice, not a bug.

## Tests

```bash
pip install pytest && pytest tests/ -v
```

Regression tests for defects that actually shipped here: inverted helmet
compliance, config/weights mismatch, train/val leakage, and silent
DeepSORT/OSNet fallbacks. Runs in CI on every push (`.github/workflows/tests.yml`).

## Config

Everything (paths, thresholds, weights, risk weights) lives in
`config/settings.yaml`. No hard-coded constants in code.

## Risk model

```
R = w1·WheelViolation + w2·HelmetMisuse + w3·(Riders>2) + w4·WrongWay
LOW: R<3   MEDIUM: 3≤R<6   HIGH: R≥6
```

Validated against 10,006 real motorcycle tracks (HELMET dataset) — see
[docs/PHASE3.md](docs/PHASE3.md) for the calibration finding (HIGH never
triggers from helmet-misuse + overload alone; a policy question, not a bug).

## Metrics

`utils/metrics.py` — mAP@50, precision, recall, latency (ms), FPS, RAM, CPU%.
`utils.BenchmarkReport` tabulates before/after optimization.

## Documentation

- [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) — step-by-step execution guide
- [docs/PHASE1.md](docs/PHASE1.md) · [PHASE2.md](docs/PHASE2.md) · [PHASE3.md](docs/PHASE3.md) · [PHASE4.md](docs/PHASE4.md) — per-phase technical notes
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — full completion plan and status
- [DATASETS.md](DATASETS.md) — dataset sources and placement
