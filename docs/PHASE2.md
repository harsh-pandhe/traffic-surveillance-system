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

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Latency (ms) |
|---|---|---|---|---|---|
| SmallCNN (custom baseline) | 0.639 | 0.640 | 0.637 | 0.627 | 1.9 |
| **YOLOv8-cls (selected)** | **0.867** | 0.865 | 0.867 | 0.865 | 3.2 |

Per-class F1 (YOLOv8-cls): 2-Wheeler 0.91, 3-Wheeler 0.96, 4-Wheeler 0.81,
6+ Wheeler 0.78.

**Selected model: YOLOv8-cls** — +23 points accuracy over the CNN baseline at a
small latency cost. Weights: `weights/wheel_cnn.pt`, YOLO run under `runs/`.

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
