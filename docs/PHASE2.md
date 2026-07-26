# Phase 2 — Model Training & Granular Helmet Compliance

**Days 6–12 milestone.** Deliverables: (a) baseline CNN vs YOLO wheel-count
classification with a Precision/Recall/mAP comparison and a selected model, and
(b) a granular helmet-compliance detector.

## Components

| Component | File | Role |
|---|---|---|
| Wheel dataset builder | `scripts/build_wheel_dataset.py` | Crop COCO instances → 2/3/4/6+ wheel taxonomy (ImageFolder) |
| Wheel benchmark | `scripts/train_wheel_classifier.py` | Train + compare SmallCNN vs YOLOv8-cls |
| Custom CNN | `src/models/benchmark_classifier.py` | `SmallCNN` baseline |
| Helmet trainer | `scripts/train_helmet_detector.py` | Fine-tune YOLOv8 + report mAP |
| Helmet detector | `src/models/helmet_detector.py` | 7-class inference wrapper |

## Part A — Wheel-count classification (CNN vs YOLO)

All **four** classes represented: **2-Wheeler** (bicycle, motorcycle from COCO),
**3-Wheeler** (auto-rickshaw crops from the AUTO-RICKSHAW-DETECTION set),
**4-Wheeler** (car), **6+ Wheeler** (bus, truck). Balanced ~308–350 train per
class. Trained on ImageFolder crops (`data/wheels/{train,val}`), 96×96, CPU.

Three architectures benchmarked under identical data and evaluation protocol:

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Latency (ms) | Params | Epochs | Pretrained |
|---|---|---|---|---|---|---|---|---|
| SmallCNN (custom baseline) | 0.751 | 0.752 | 0.750 | 0.750 | 1.8 | 0.24M | 12 | none |
| RTMDet (CSPNeXt) | 0.683 | 0.688 | 0.686 | 0.683 | 0.6 | 2.35M | 40 | none |
| **YOLOv8-cls (selected)** | **0.867** | 0.865 | 0.867 | 0.865 | 3.4 | — | 8 | ImageNet |

Per-class F1 (YOLOv8-cls): 2-Wheeler 0.91, 3-Wheeler 0.96, 4-Wheeler 0.81,
6+ Wheeler 0.78.

**Selected model: YOLOv8-cls.**

### Methodological note (important for the paper)

The comparison is **not pretraining-neutral**. YOLOv8-cls starts from ImageNet
weights; SmallCNN and CSPNeXt are trained from scratch on ~1,350 crops. Two
consequences:

1. YOLOv8-cls's margin partly reflects transfer learning, not just architecture.
2. CSPNeXt (2.35M params) is the largest model but has the least data per
   parameter, so it underfits this dataset. At 12 epochs it scored only 0.562;
   extending to 40 epochs (training loss 0.95 → 0.26, converged) lifted it to
   0.683. It is the **fastest** model at inference (0.6 ms) despite being the
   largest — CSPNeXt's depthwise 5×5 design is efficient on CPU.

Conclusion: for this dataset size, ImageNet-pretrained YOLOv8-cls is the right
production choice; CSPNeXt would be expected to close the gap given either
pretrained weights or an order of magnitude more data.

### RTMDet implementation note

`mmdetection`/`mmcv` (the official RTMDet home) cannot be installed in this
environment: `openmim` fails on Python 3.12 (`AttributeError: module 'pkgutil'
has no attribute 'ImpImporter'`) and `mmcv` publishes no wheels for
torch 2.12. RTMDet's architectural contribution — the **CSPNeXt** backbone — is
therefore reimplemented directly in PyTorch in `src/models/cspnext.py`,
following the RTMDet paper (depthwise 5×5 CSP blocks, channel attention, SiLU,
SPPF, RTMDet-tiny scaling). This keeps the contract's CNN-vs-RTMDet comparison
honest and runnable on CPU.

Weights: `weights/wheel_cnn.pt`, `weights/wheel_cspnext.pt`,
`weights/wheel_yolov8_cls.pt`.

## Part B — Granular helmet compliance (YOLOv8)

Dataset: 7-class rider/helmet detection (368 train / 65 val / 52 test), YOLO
format. Fine-tuned YOLOv8n, 25 epochs, 416×416, CPU.

**Validation:** mAP@50 = **0.731**, mAP@50-95 = 0.385, Precision = 0.741,
Recall = 0.668.

| Class | AP@50 |
|---|---|
| bike | 0.863 |
| driver | 0.841 |
| passenger | 0.812 |
| driver_with_helmet | 0.784 |
| driver_without_helmet | 0.680 |
| passenger_without_helmet | 0.577 |
| passenger_with_helmet | 0.560 |

Weights: `weights/helmet_yolov8.pt` (auto-loaded by `helmet_detector.py`).

Passenger-helmet classes are weakest — fewest instances (passenger_with_helmet
n=65) and small objects. More passenger-side data would lift these.

## Reproduce

```bash
# Part A
python scripts/build_wheel_dataset.py                 # after COCO in data/raw
python scripts/train_wheel_classifier.py

# Part B (after a YOLO-format helmet set in data/raw/helmet_raw)
python scripts/train_helmet_detector.py \
    --data data/raw/helmet_raw/data.yaml --epochs 25 --imgsz 416 --batch 8
```

Metrics persisted in `results/phase2/*.json`.

## Known gaps

1. Helmet dataset taxonomy (driver/passenger × helmet/no-helmet + bike) maps to
   the project's 7-class compliance scheme; strap/hanging/arm sub-classes need a
   dedicated dataset (future data collection).
2. Passenger-helmet classes have the fewest instances — targeted collection or
   augmentation would raise their AP (issue #8).

## Deliverable checklist

- [x] Baseline CNN trained (SmallCNN)
- [x] YOLO model trained (YOLOv8-cls)
- [x] Precision/Recall/mAP comparison + selected model
- [x] Granular helmet-compliance detector (7 classes, mAP reported)
- [x] Reproduction scripts + persisted metrics
- [x] 3-Wheeler class (auto-rickshaw crops) — all 4 wheel classes active
