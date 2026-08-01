# Phase 4 — Optimization & Benchmarking

**Days 18–21 milestone.** Deliverables: pruning + INT8 quantization for faster
CPU inference, and a before/after benchmark (mAP, FPS, memory, latency).

## Components

| Component | File | Role |
|---|---|---|
| Optimizer | `src/optimization/model_quantizer.py` | Prune, fine-tune, quantize, ONNX export |
| Benchmark | `scripts/run_phase4_benchmark.py` | Runs every model through 4 configs, records accuracy retention |

## Method

Each wheel-count model (SmallCNN, RTMDet-CSPNeXt) runs through:

| Config | What it is |
|---|---|
| `fp32` | Baseline PyTorch |
| `pruned_naive` | 30% L1 structured pruning, **no fine-tuning** — diagnostic only |
| `pruned` | Same pruning + 3-epoch fine-tune recovery — the deployment candidate |
| `int8` | PyTorch dynamic INT8 quantization on the recovered pruned model |
| `onnx_int8` | ONNX export + ONNX Runtime INT8 quantization |

YOLOv8-cls runs FP32 PyTorch vs its ONNX export (Ultralytics handles pruning/
quantization internally; not re-implemented here).

Evaluated on 200 leak-free wheel validation crops (`data/wheels/val`).

## Finding: naive pruning collapses accuracy — fine-tuning is not optional

| Model | FP32 | Pruned (naive) | Pruned (+fine-tune) |
|---|---|---|---|
| SmallCNN | 0.860 | **0.270** (ret. 0.31) | **0.840** (ret. 0.98) |
| RTMDet-CSPNeXt | 0.785 | **0.465** (ret. 0.59) | **0.820** (ret. 1.04) |

30% structured pruning zeroes whole output channels in every conv/linear layer
independently; because layers feed each other the effect compounds, and with no
retraining the network never recovers. This is expected behaviour for naive
structured pruning, not a code bug — the standard fix is exactly what's applied
here: prune, then fine-tune. Three epochs at a low learning rate (1e-4) fully
recovers SmallCNN and, notably, pushes CSPNeXt **above** its FP32 baseline
(1.04× — consistent with the Phase 2 finding that CSPNeXt was still improving
with more training on this dataset size).

## Full results

| Model | Config | Accuracy | Retention | Latency (ms) | FPS | Size (MB) |
|---|---|---|---|---|---|---|
| SmallCNN | fp32 | 0.860 | 1.00 | 0.90 | 1110 | 0.94 |
| SmallCNN | pruned_naive | 0.270 | 0.31 | 0.91 | 1098 | 0.94 |
| SmallCNN | pruned | 0.840 | 0.98 | 0.98 | 1025 | 0.94 |
| SmallCNN | int8 | 0.840 | 0.98 | 1.04 | 958 | 0.94 |
| SmallCNN | onnx_int8 | 0.840 | 0.98 | 2.08 | 481 | **0.24** |
| RTMDet-CSPNeXt | fp32 | 0.785 | 1.00 | 3.12 | 320 | 9.07 |
| RTMDet-CSPNeXt | pruned_naive | 0.465 | 0.59 | 3.21 | 312 | 9.07 |
| RTMDet-CSPNeXt | pruned | 0.820 | 1.04 | 3.09 | 324 | 9.07 |
| RTMDet-CSPNeXt | int8 | 0.820 | 1.04 | 3.13 | 319 | 9.06 |
| RTMDet-CSPNeXt | onnx_int8 | 0.820 | 1.04 | 3.66 | 273 | **2.37** |
| YOLOv8-cls | fp32 | 0.960 | 1.00 | 2.29 | 436 | 2.83 |
| YOLOv8-cls | onnx_fp32 | 0.980 | 1.02 | 0.49 | 2035 | 5.52 |

## Interpretation

- **INT8 quantization is essentially free** once the model is fine-tuned:
  `int8` accuracy exactly matches `pruned` for both models (0.840 vs 0.840,
  0.820 vs 0.820) — the accuracy cost in this pipeline is entirely from
  pruning, not from quantization.
- **ONNX Runtime INT8 shrinks the models 3.8–3.9×** at no further accuracy
  cost: SmallCNN 0.94 MB → 0.24 MB, CSPNeXt 9.07 MB → 2.37 MB.
- **ONNX latency is not uniformly faster on CPU at this scale.** ONNX INT8 is
  *slower* per-image than native PyTorch for the small models here (SmallCNN:
  0.98 ms native vs 2.08 ms ONNX) — session/dispatch overhead dominates at
  batch-1 on a sub-millisecond model. YOLOv8-cls's ONNX export is the exception
  (0.49 ms vs 2.29 ms), because Ultralytics' own inference path carries more
  Python-level overhead than its exported graph. **The size win is real and
  unconditional; the latency win depends on the model and is not assumed.**
- Model **size**, not latency, is the deliverable that matters most for the
  ONNX artifacts here — relevant for deployment footprint, not necessarily
  real-time throughput on this hardware.

## Reproduce

```bash
python scripts/run_phase4_benchmark.py --limit 200 --ft-epochs 3
```

Weights: `weights/wheel_cnn.pt` (recovered pruned), `weights/wheel_cnn_int8.pt`
saved as intermediate artifacts under `results/phase4/artifacts/`; ONNX graphs
alongside. Metrics: `results/phase4/benchmark.json`.

## Deliverable checklist

- [x] 30% structured pruning implemented and benchmarked
- [x] INT8 quantization (PyTorch dynamic + ONNX Runtime)
- [x] ONNX export for the real trained models (not just a demo net)
- [x] Before/after accuracy, latency, FPS, size — reported as **retention**,
      not just speedup
- [x] Real failure mode found and fixed (naive pruning collapse) rather than
      hidden behind a favourable-looking number
