# Phase 1 — Dataset Preparation & Adaptive Preprocessing

**Days 1–5 milestone.** Deliverable: a working pipeline that automatically
detects environmental conditions and enhances images before further processing,
plus a unified dataset built from benchmark datasets.

## What was built

| Component | File | Role |
|---|---|---|
| Environment detection (rules) | `src/preprocessing/scene_classifier.py` | Rule-based DAY / NIGHT / FOG / RAIN baseline |
| Environment detection (learned) | `src/preprocessing/scene_classifier_ml.py` + `scene_features.py` | RandomForest over 15 image features; drop-in, rule fallback |
| Model training | `train_scene_classifier.py` | Trains + evaluates the learned classifier, saves to `weights/` |
| Adaptive enhancement | `src/preprocessing/enhancements.py` | CLAHE (night), DCP dehaze (fog), dehaze+denoise (rain), pass-through (day) |
| Scene dataset builder + evaluator | `src/preprocessing/scene_dataset.py` | Merge weather/lighting sets, enhance, score env-detection accuracy |
| Detection/cls dataset unifier | `src/preprocessing/dataset_builder.py` | Merge YOLO/COCO/folder sources → one taxonomy → `data.yaml` |
| Config | `config/settings.yaml` | All thresholds/paths/sources |

## Environment detection — how it decides

Cheap, explainable per-frame statistics, fused by priority rules (no training):

- **mean luminance** → darkness ⇒ NIGHT
- **Laplacian variance** → global sharpness; haze smears edges ⇒ FOG
- **HSV saturation** → colour washout reinforces FOG
- **blue/red channel ratio** → cool cast ⇒ RAIN
- **FFT high-frequency energy** → rain streak texture ⇒ RAIN

## Adaptive enhancement — applied only when needed

- **NIGHT** → CLAHE on the LAB L-channel (recovers shadow detail).
- **FOG** → Dark Channel Prior dehazing, transmission refined by a fast guided filter.
- **RAIN** → DCP dehaze + non-local-means denoise.
- **DAY** → untouched (no needless processing).

## Verification (reproducible)

Synthetic 4-condition test — `python demo_phase1.py`:

```
expected | predicted | match
DAY      | DAY       | YES
NIGHT    | NIGHT     | YES
FOG      | FOG       | YES
RAIN     | RAIN      | YES
scene accuracy: 4/4
```

Before/after images written to `outputs/phase1/`.

Real-data evaluation — `python -m src.preprocessing.scene_dataset` (after
placing DAWN/ExDark/COCO per `DATASETS.md`) prints a confusion matrix + per-class
Precision/Recall/F1 over the benchmark images.

### Real-data results (DAWN + ExDark + COCO)

Datasets used: **DAWN** Fog (300) / Rain (200), **ExDark** night (300),
**COCO** val2017 daytime (300). Balanced at 300/class, 80/20 stratified split.

Rule-based baseline on the confusable Fog/Rain pair was only **~49%** (rain
collapsed into fog: 14% rain recall). The **learned RandomForest** lifts this
sharply:

| Model | Accuracy | Notes |
|---|---|---|
| Rules (Fog/Rain only) | 0.49 | rain recall 0.14 |
| Rules tuned (Fog/Rain) | 0.59 | rain recall 0.38 |
| Learned, 2-class (Fog/Rain) | 0.68 | rain recall 0.60 |
| Learned, 3-class (+Night) | 0.83 | night recall 1.00 |
| **Learned, 4-class (final)** | **0.79** | see per-class below |

### Addressing the dataset-provenance confound

The 4-class result mixes three source datasets (DAWN for fog/rain, ExDark for
night, COCO for day), so in principle the classifier could be learning "which
dataset is this" (camera, compression, resolution) rather than weather itself —
a real confound worth ruling out rather than assuming away.

The **2-class Fog/Rain row above is the unconfounded check**: both classes come
from the *same* source (DAWN), so camera and dataset provenance are held
constant and only the weather condition differs. The classifier still reaches
0.68 accuracy / 0.60 rain recall on this pair — well above chance (0.50) and
far above the same-source rule baseline (0.49) — which is direct evidence the
model is picking up genuine weather signal (haze/edge/texture statistics), not
just dataset fingerprints. This is the honest evidence for the claim, not the
inflated 4-class number.

Final 4-class validation (n=220):

| Class | Precision | Recall |
|---|---|---|
| DAY | 0.83 | 0.87 |
| NIGHT | 0.90 | 0.88 |
| FOG | 0.77 | 0.72 |
| RAIN | 0.62 | 0.65 |

Top features: `luminance_mean`, `dark_channel_mean`, `log_laplacian`,
`laplacian_var`, `value_mean`, `edge_density`. Residual error is FOG↔RAIN, which
physically overlap (overcast rain vs light fog). Model saved to
`weights/scene_classifier.joblib`; `main.py` loads it automatically.

Retrain: `python train_scene_classifier.py --limit 300`.

### Bug fixed during verification
DCP dehaze whited-out fog frames due to a double `/255` on the atmospheric-light
term (A≈0.003 → ~10× amplification). Fixed in `enhancements.py`; caught only by
visually inspecting the enhanced output, not by compilation.

## Datasets used (Phase 1)

DAWN (fog/rain), ExDark (night), COCO / BDD100K (day). Links + placement in
`DATASETS.md`.

## How to reproduce end-to-end

```bash
pip install -r requirements.txt
python demo_phase1.py                        # synthetic proof + before/after images
# after downloading datasets (see DATASETS.md):
python -m src.preprocessing.scene_dataset    # enhance corpus + accuracy metrics
python -m src.preprocessing.dataset_builder  # unified YOLO dataset for Phase 2
```

## Deliverable checklist

- [x] Environment detection module (Day/Night/Rain/Fog)
- [x] Adaptive enhancement applied only when required
- [x] Unified dataset builder from benchmark datasets
- [x] Dataset preprocessing scripts
- [x] Verified pipeline (4/4 synthetic; confusion-matrix eval for real data)
- [x] Reproduction + dataset docs
