# Datasets

Datasets actually used, by phase, with the placement each script expects. All
were downloaded via Kaggle mirrors unless noted; `✓ used` means present in
`data/raw/` and exercised by a script in this repo, not just linked.

| Dataset | Purpose | Phase | Placement | Link |
|---|---|---|---|---|
| **DAWN** ✓ used | Fog / Rain driving images | 1 | `data/raw/DAWN/{Fog,Rain}/` | kaggle.com/datasets/orvile/dawn-detection-in-adverse-weather-nature |
| **ExDark** ✓ used | Low-light / night images | 1 | `data/raw/ExDark/` | github.com/cs-chan/Exclusively-Dark-Image-Dataset |
| **COCO val2017** ✓ used | Day scenes (Phase 1) + wheel crops (Phase 2) | 1, 2 | `data/raw/COCO/val2017/` | cocodataset.org |
| **Helmet (7-class YOLO)** ✓ used | Granular helmet compliance | 2 | `data/raw/helmet_raw/` | AI City Track-5 taxonomy, Kaggle YOLO mirror |
| **AUTO-RICKSHAW-DETECTION** ✓ used | 3-wheeler crops | 2 | `data/raw/auto_rickshaw/` | kaggle.com/datasets/uzairahmad1434/auto-rickshaw-detection |
| **UA-DETRAC** ✓ used | Tracking video + occlusion GT | 3 | `data/raw/UA-DETRAC/` | kaggle.com/datasets/bratjay/ua-detrac-orig (mirror of the official set) |
| **VeRi-776** ✓ used | Cross-camera vehicle ReID | 3 | `data/raw/VeRi/VeRi/` | kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset |
| **HELMET** ✓ used (annotations only) | Motorcycle occupancy + helmet-use ground truth | 3 | `data/raw/HELMET/annotation/` | osf.io/4pwj8 — 910 clips, 283,377 instances, 10,006 tracks. Images (~29 GB, 7 parts) not downloaded: a learning-curve gate showed the detector has plateaued at 368 images, so the download would not have paid off — see `docs/PHASE3.md`. |

BDD100K, CityFlow, FairFace, and UTKFace were considered but not used: COCO/
UA-DETRAC/HELMET covered the same needs without a registration wall
(BDD100K, CityFlow) or a second demographics dataset once DeepFace's own
pretrained models proved sufficient (FairFace, UTKFace).

## Phase 1 placement

Download and lay out under `data/raw/` exactly like this so the scene builder
finds them automatically:

```
data/raw/
├── DAWN/
│   ├── Fog/        # -> FOG
│   └── Rain/       # -> RAIN
├── ExDark/         # -> NIGHT   (any nested folders are walked)
└── COCO/
    └── day/        # -> DAY     (or a BDD100K daytime subset)
```

The condition→label mapping lives in
`src/preprocessing/scene_dataset.py::_default_sources`. Edit it if your folder
names differ.

## Build the Phase 1 dataset + evaluate the scene classifier

```bash
python -m src.preprocessing.scene_dataset
```

Outputs:
- **Enhanced corpus** → `data/processed/<CONDITION>/` (CLAHE/dehaze applied).
- **Env-detection metrics** → confusion matrix + per-class Precision/Recall/F1
  printed to console (the Phase 1 milestone metric).

## Build a detection/classification dataset (for Phase 2 training)

`config/settings.yaml → dataset_builder` lists sources (YOLO / COCO / folder
formats) with a per-source `class_map` into the 7-class helmet or 4-class wheel
taxonomy. Then:

```bash
python -m src.preprocessing.dataset_builder
```

Produces a unified YOLO dataset + `data.yaml` under `data/unified/`, ready for
`yolo train`.
