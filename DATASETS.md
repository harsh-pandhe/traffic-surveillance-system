# Datasets

Official sources for every phase. Phase 1 uses only the **weather / lighting**
sets (for scene detection + enhancement). The rest are Phase 2–3.

| Dataset | Purpose | Phase | Link |
|---|---|---|---|
| **DAWN** | Fog / Rain / Sand / Snow driving images | 1 | https://www.kaggle.com/datasets/orvile/dawn-detection-in-adverse-weather-nature |
| **ExDark** | Low-light / night images | 1 | https://github.com/cs-chan/Exclusively-Dark-Image-Dataset |
| **BDD100K** | Day/night + weather attributes, driving | 1 | https://bdd-data.berkeley.edu/ |
| **COCO** | General daytime objects | 1 | https://cocodataset.org/ |
| CityFlow (AI City) | Multi-camera vehicle | 3 | https://www.aicitychallenge.org/ |
| VeRi-776 | Vehicle ReID | 3 | https://github.com/VehicleReId/VeRi |
| FairFace | Demographics (age/gender) | 3 | https://github.com/dchen236/FairFace |
| UTKFace | Age / gender | 3 | https://www.kaggle.com/datasets/jangedoo/utkface-new |

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
