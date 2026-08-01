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
| SmallCNN (custom baseline) | 0.787 | 0.756 | 0.753 | 0.753 | 1.1 | 0.24M | 12 | none |
| RTMDet (CSPNeXt) | 0.695 | 0.638 | 0.645 | 0.639 | 1.0 | 2.35M | 12 | none |
| **YOLOv8-cls (selected)** | **0.861** | 0.829 | 0.826 | 0.827 | 2.9 | — | 8 | ImageNet |

> **These numbers supersede an earlier, invalid run.** The first benchmark
> reported 0.751 / 0.683 / 0.867 on a split that leaked: `build_wheel_dataset.py`
> shuffled *crops* and sliced by index, so several vehicles from the same COCO
> photo landed on both sides of the split (measured: 76% / 91% / 65% of 2-, 4-
> and 6+-wheeler crops came from such photos). Separately, the 3-wheeler class
> was built from **whole photos** while every other class was a tight crop,
> handing the model a possible shape shortcut. Both defects are fixed: the split
> is now by source image and all classes are crops. Because both changed at once,
> the difference between old and new numbers is **not** attributable to leakage
> alone. The model ranking is unchanged, so the selection conclusion is robust.
>
> One hypothesis was **disproved** by the rerun: the 3-wheeler class was
> suspected of scoring 0.96 F1 because of the whole-photo shortcut, but after
> cropping it still scores 0.96 — auto-rickshaws are simply visually distinctive.
> The classes that actually fell are 4-wheeler (0.81 → 0.70) and 6+ wheeler
> (0.78 → 0.72), and 4-wheeler was the most leak-prone class at 91%. That is the
> pattern leakage predicts.

Per-class F1 (YOLOv8-cls, leak-free): 2-Wheeler 0.93, 3-Wheeler 0.96,
4-Wheeler 0.70, 6+ Wheeler 0.72. Car-vs-bus/truck is now the hard pair, which is
expected once same-scene crops no longer span the split.

**Selected model: YOLOv8-cls.**

### Methodological note (important for the paper)

The comparison is **not pretraining-neutral**. YOLOv8-cls starts from ImageNet
weights; SmallCNN and CSPNeXt are trained from scratch on ~1,350 crops. Two
consequences:

1. YOLOv8-cls's margin partly reflects transfer learning, not just architecture.
2. CSPNeXt (2.35M params) is the largest model but has the least data per
   parameter, so it underfits this dataset. It is nonetheless competitive on
   **latency** (1.0 ms, essentially tied with the 10× smaller SmallCNN and ~3×
   faster than YOLOv8-cls) — CSPNeXt's depthwise 5×5 design is efficient on CPU.
   On the earlier dataset it was still improving with longer training (0.562 at
   12 epochs → 0.683 at 40), so its reported figure is a floor, not a ceiling.

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

**Held-out test set (52 images, 321 instances) — the headline number:**
mAP@50 = **0.764**, mAP@50-95 = 0.372, Precision = 0.732, Recall = 0.746.

**Validation set (65 images):** mAP@50 = 0.731, mAP@50-95 = 0.385,
Precision = 0.741, Recall = 0.668.

The validation split was used to select the best epoch, so it is optimistically
biased; the test split was never touched until final evaluation. Test scoring
slightly *higher* than validation is small-split variance, not a better model —
both splits are small, so each carries a wide confidence interval. They
corroborate each other at roughly 0.73–0.76.

| Class | AP@50 (test) |
|---|---|
| driver | 0.935 |
| bike | 0.881 |
| passenger | 0.821 |
| driver_without_helmet | 0.808 |
| driver_with_helmet | 0.795 |
| passenger_without_helmet | 0.565 |
| passenger_with_helmet | 0.545 |

The passenger-helmet classes are weakest on **both** splits, so this is a real
data limitation (few passenger instances, small objects) rather than a split
artifact — tracked as issue #8.

Weights: `weights/helmet_yolov8.pt` (auto-loaded by `helmet_detector.py`).

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
