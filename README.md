# Adaptive Spatio-Temporal Traffic Surveillance

Granular Helmet Compliance · Multi-Frame Wheel-Count Classification · Real-Time Risk Indexing.

CPU-optimized, modular Python pipeline for M.Tech research.

## Pipeline (per frame)

| # | Stage | Module | Output |
|---|-------|--------|--------|
| 1 | Scene classify | `src/preprocessing/scene_classifier.py` | DAY / NIGHT / FOG / RAIN |
| 2 | Adaptive enhance | `src/preprocessing/enhancements.py` | CLAHE / DCP dehaze / denoise |
| 3 | Helmet compliance | `src/models/helmet_detector.py` | 7-class YOLOv8 detections |
| 4 | Wheel classify | `src/models/benchmark_classifier.py` | 2/3/4/6+ wheeler (CNN vs YOLO) |
| 5 | Track + vote | `src/tracking/deepsort_tracker.py` | stable IDs, temporal wheel vote |
| 6 | Cross-cam ReID | `src/tracking/multi_camera_reid.py` | OSNet global IDs |
| 7 | Demographics | `src/analytics/demographics.py` | age group + gender |
| 8 | Risk index | `src/analytics/risk_indexer.py` | LOW / MEDIUM / HIGH banner |
| 9 | Optimize (offline) | `src/optimization/model_quantizer.py` | INT8 + 30% prune + ONNX |

## Install

```bash
cd traffic_surveillance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Heavy backends (Ultralytics, deep-sort-realtime, torchreid, DeepFace) are all
**optional at runtime** — every stage has a graceful fallback so the pipeline
runs out-of-the-box, then upgrades automatically once real weights are present.

## Weights

Drop fine-tuned weights in `weights/` (paths in `config/settings.yaml`):

- `helmet_yolov8.pt` — 7-class helmet compliance detector
- `wheel_yolov8_cls.pt` / `wheel_cnn.pt` — wheel-count classifiers
- `osnet_x0_25.pth` — OSNet ReID backbone

Missing files → stock/pretrained fallbacks (demo mode).

## Run

```bash
python main.py --source data/raw/clip.mp4 --camera cam_A   # process a clip
python main.py --source 0 --show                           # live webcam
python main.py --optimize                                  # Phase 4 demo
```

Annotated video → `outputs/annotated.mp4`.

## Config

Everything (paths, thresholds, weights, risk weights) lives in
`config/settings.yaml`. No hard-coded constants in code.

## Risk model

```
R = w1·WheelViolation + w2·HelmetMisuse + w3·(Riders>2) + w4·WrongWay
LOW: R<3   MEDIUM: 3≤R<6   HIGH: R≥6
```

## Metrics

`utils/metrics.py` — mAP@50, precision, recall, latency (ms), FPS, RAM, CPU%.
Use `BenchmarkReport` to tabulate before/after optimization.
