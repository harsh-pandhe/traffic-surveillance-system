# How to Run

Step-by-step execution guide for the Adaptive Spatio-Temporal Traffic
Surveillance system. Every command here has been run on the reference machine
(Linux, Python 3.12, CPU-only).

---

## 1. Prerequisites

- Python 3.10+ (reference environment: 3.12)
- ~15 GB free disk for datasets, ~2 GB for models
- No GPU required — the whole system is CPU-optimized

## 2. Installation

```bash
git clone https://github.com/harsh-pandhe/traffic-surveillance-system.git
cd traffic-surveillance-system
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools        # see note below - do not skip
pip install -r requirements.txt
```

> **Do not skip `--upgrade setuptools`.** On Python 3.12 an old setuptools makes
> `pkg_resources` raise `module 'pkgutil' has no attribute 'ImpImporter'`, which
> silently degrades DeepSORT to a fallback IoU tracker instead of failing loudly.

### Verifying the optional backends

Every heavy backend has a graceful fallback, so the pipeline runs even when one
is missing — which means a missing backend is easy to *not notice*. Check
explicitly:

```bash
python -c "
from src.tracking.deepsort_tracker import VehicleTracker
from src.tracking.multi_camera_reid import VehicleReID
print('tracker:', VehicleTracker()._backend)   # want: deepsort  (not: iou)
print('reid   :', VehicleReID()._backend)      # want: osnet     (not: histogram)
"
```

If you see `iou` or `histogram`, a dependency is missing or broken — see
§7 Troubleshooting.

## 3. Datasets

Download and place as below. Full links and licences are in `DATASETS.md`.

| Dataset | Used by | Placement |
|---|---|---|
| DAWN (fog/rain) | Phase 1 | `data/raw/DAWN/{Fog,Rain}/` |
| ExDark | Phase 1 (night) | `data/raw/ExDark/` |
| COCO val2017 | Phase 1 (day), Phase 2 (wheels) | `data/raw/COCO/val2017/` |
| Helmet (YOLO format) | Phase 2 | `data/raw/helmet_raw/` |
| AUTO-RICKSHAW-DETECTION | Phase 2 (3-wheeler) | `data/raw/auto_rickshaw/` |
| UA-DETRAC | Phase 3 (tracking + occlusion GT) | `data/raw/UA-DETRAC/` |
| VeRi-776 | Phase 3 (cross-camera ReID) | `data/raw/VeRi/` |

## 4. Running each phase

### Phase 1 — scene classification and adaptive enhancement

```bash
# Synthetic sanity check + before/after images (no datasets needed, ~10 s)
python demo_phase1.py

# Real-data evaluation: enhances the corpus and prints a confusion matrix
python -m src.preprocessing.scene_dataset

# Train the learned scene classifier (~2 min on CPU)
python train_scene_classifier.py --limit 300
```

Expected: 4-class accuracy ≈ **0.79**; model written to
`weights/scene_classifier.joblib`; enhanced frames under `data/processed/<COND>/`.

### Phase 2 — wheel-count benchmark and helmet detector

```bash
# Build the 4-class wheel dataset from COCO + auto-rickshaw crops
python scripts/build_wheel_dataset.py

# Benchmark SmallCNN vs RTMDet(CSPNeXt) vs YOLOv8-cls  (~25 min on CPU)
python scripts/train_wheel_classifier.py

# Train the 7-class helmet detector  (~7 min on CPU)
python scripts/train_helmet_detector.py \
    --data data/raw/helmet_raw/data.yaml --epochs 25 --imgsz 416 --batch 8
```

Expected: YOLOv8-cls selected at ≈ **0.867** accuracy; helmet
**mAP@50 ≈ 0.731**. Metrics land in `results/phase2/*.json`.

### Phase 3 — tracking, occlusion voting, ReID

```bash
# Headline experiment: does multi-frame voting help under occlusion?
python scripts/run_occlusion_ablation.py \
    --root data/raw/UA-DETRAC --sequences 5 --max-frames 300
```

Expected: an accuracy table stratified by occlusion band
(none / light / medium / heavy) for vote windows N = 1, 5, 10, 15, 30, written
to `results/phase3/occlusion_ablation.json`.

### Phase 4 — optimization and benchmarking

```bash
python main.py --optimize        # prune -> ONNX export -> INT8
```

Expected: `weights/model.onnx` (FP32) and `weights/model_int8.onnx` (INT8,
roughly 3.8× smaller).

## 5. Running the full pipeline

```bash
# On a video
python main.py --source data/raw/UA-DETRAC/<sequence>.mp4 --camera cam_A

# On a webcam, with a live preview window
python main.py --source 0 --show
```

Annotated output is written to `outputs/annotated.mp4`. Each frame is
scene-classified, enhanced, run through helmet and wheel models, tracked, ReID'd,
and scored for risk (LOW / MEDIUM / HIGH banner).

## 6. Reproducing the reports

All figures and tables regenerate from the tracked JSON in `results/`, so the
numbers in the PDFs cannot drift from the numbers in the code.

```bash
python scripts/make_report_assets.py    && python scripts/build_phase1_pdf.py
python scripts/make_phase2_assets.py    && python scripts/build_phase2_pdf.py
```

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `tracker: iou` instead of `deepsort` | Old setuptools on Python 3.12 (`pkgutil.ImpImporter`). Run `pip install --upgrade setuptools`. |
| `reid: histogram` instead of `osnet` | torchreid ships under two layouts; the code tries both `torchreid.reid.utils` and `torchreid.utils`. Also needs `gdown` and `tensorboard`. |
| `cannot import name 'AsyncFileLock' from 'filelock'` | Something downgraded `filelock`. Run `pip install --upgrade filelock`. |
| `module 'pkgutil' has no attribute 'ImpImporter'` | Python 3.12 with old setuptools — upgrade it. Commonly caused by installing `openmim`. |
| ONNX export fails with `__obj_flatten__` | You are exporting a *quantized* model. Export FP32 first, then quantize the ONNX graph (this is what `ModelOptimizer.optimize()` now does). |
| ONNX INT8 fails with `Inferred shape ... (128) vs (4)` | torch ≥ 2.5 uses the dynamo exporter, whose graphs break ONNX Runtime's quantizer. Export with `dynamo=False`. |
| `HelmetDetector` prints a class-mismatch warning | `config/settings.yaml → helmet_detector.class_names` does not match the loaded weights. Fix the config, or compliance logic will be wrong. |

> **Do not install `openmim` / `mmcv` in this environment.** openmim pulls an old
> setuptools that breaks Python 3.12 and downgrades `filelock` (breaking
> Ultralytics), and mmcv publishes no wheels for torch 2.12. RTMDet is provided
> via `src/models/cspnext.py` instead.

## 8. Repository layout

```
config/settings.yaml     single source of truth for paths, thresholds, classes
src/preprocessing/       scene classification + adaptive enhancement (Phase 1)
src/models/              helmet detector, wheel classifiers, CSPNeXt (Phase 2)
src/tracking/            DeepSORT, OSNet ReID, UA-DETRAC loader (Phase 3)
src/analytics/           risk indexer, demographics (Phase 3)
src/optimization/        pruning, quantization, ONNX export (Phase 4)
scripts/                 dataset builders, trainers, experiments, report builders
results/phase1..4/       tracked metrics -- the source of every reported number
docs/                    per-phase notes and report PDFs
main.py                  end-to-end pipeline orchestrator
```
