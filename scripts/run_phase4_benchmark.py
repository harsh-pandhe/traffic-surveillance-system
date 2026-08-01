"""
scripts/run_phase4_benchmark.py
--------------------------------
Phase 4 - optimization benchmark.

Runs each trained model through four deployment configurations and records what
each one costs and saves:

    FP32        baseline PyTorch
    pruned      30% L1 structured sparsity
    int8        PyTorch dynamic INT8 quantization
    onnx_int8   ONNX Runtime, INT8-quantized graph

For every configuration we record accuracy (so a speedup that destroys the model
is visible), latency, throughput, peak RAM, and on-disk size. The headline
number is **accuracy retention**, not raw speedup: halving latency while losing
15 points of accuracy is a finding, not a win.

Evaluated on the leak-free wheel validation split.

Usage:
    python scripts/run_phase4_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.benchmark_classifier import SmallCNN
from src.optimization.model_quantizer import ModelOptimizer
from utils.metrics import system_stats

OUT_DIR = "results/phase4"
WHEELS_VAL = "data/wheels/val"
IMGSZ = 96


# --------------------------------------------------------------------------- #
def load_val(imgsz: int, limit: int | None = None):
    """Load the wheel validation split as tensors (shared by every config)."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((imgsz, imgsz)),
                             transforms.ToTensor()])
    ds = datasets.ImageFolder(WHEELS_VAL, tf)
    xs, ys = [], []
    for i in range(len(ds)):
        if limit and i >= limit:
            break
        x, y = ds[i]
        xs.append(x); ys.append(y)
    return torch.stack(xs), np.array(ys), ds.classes


def torch_accuracy_latency(model, X, y, warmup: int = 5):
    """Per-sample latency and accuracy for a torch module."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(X[:1])
        preds, lat = [], []
        for i in range(len(X)):
            t0 = time.perf_counter()
            out = model(X[i:i + 1])
            lat.append((time.perf_counter() - t0) * 1000)
            preds.append(int(out.argmax(1).item()))
    acc = float((np.array(preds) == y).mean())
    return acc, float(np.mean(lat))


def onnx_accuracy_latency(path: str, X, y, warmup: int = 5):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    arr = X.numpy()
    for _ in range(warmup):
        sess.run(None, {name: arr[:1]})
    preds, lat = [], []
    for i in range(len(arr)):
        t0 = time.perf_counter()
        out = sess.run(None, {name: arr[i:i + 1]})[0]
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(int(np.argmax(out, axis=1)[0]))
    acc = float((np.array(preds) == y).mean())
    return acc, float(np.mean(lat))


def file_size_mb(p: str) -> float:
    total = os.path.getsize(p) if os.path.isfile(p) else 0.0
    ext = p + ".data"                      # ONNX external weights
    if os.path.isfile(ext):
        total += os.path.getsize(ext)
    return total / (1024 ** 2)


# --------------------------------------------------------------------------- #
def bench_torch_model(name: str, build, weights: str, X, y, n_classes: int,
                      workdir: str, train_loader=None,
                      ft_epochs: int = 3) -> list[dict]:
    """Run one torch classifier through all four configurations."""
    rows = []

    def record(config, acc, lat, size_mb, note=""):
        stats = system_stats()
        rows.append({"model": name, "config": config, "accuracy": acc,
                     "latency_ms": lat, "fps": 1000.0 / lat if lat else None,
                     "size_mb": size_mb, "ram_mb": stats["ram_mb"],
                     "cpu_percent": stats["cpu_percent"], "note": note})
        print(f"  {config:10s} acc={acc:.3f} lat={lat:6.2f}ms "
              f"size={size_mb:6.2f}MB")

    # --- FP32 baseline ---
    model = build(n_classes)
    model.load_state_dict(torch.load(weights, map_location="cpu",
                                     weights_only=True))
    acc, lat = torch_accuracy_latency(model, X, y)
    record("fp32", acc, lat, file_size_mb(weights))
    baseline_acc = acc

    opt = ModelOptimizer()

    # --- pruned (30% structured), BEFORE recovery ---
    # Reported to make the failure mode visible: naive structured pruning with
    # no retraining collapses accuracy (measured: SmallCNN 0.86 -> 0.27). This
    # config is diagnostic, not a deployment candidate.
    pruned = build(n_classes)
    pruned.load_state_dict(torch.load(weights, map_location="cpu",
                                      weights_only=True))
    pruned = opt.prune_structured(pruned)
    p_path = os.path.join(workdir, f"{name}_pruned_naive.pt")
    torch.save(pruned.state_dict(), p_path)
    acc, lat = torch_accuracy_latency(pruned, X, y)
    record("pruned_naive", acc, lat, file_size_mb(p_path),
          note="no fine-tune after pruning -- diagnostic, not deployable")

    # --- pruned + fine-tuned: the actual deployment candidate ---
    if train_loader is not None:
        pruned = opt.finetune(pruned, train_loader, epochs=ft_epochs)
    pf_path = os.path.join(workdir, f"{name}_pruned.pt")
    torch.save(pruned.state_dict(), pf_path)
    acc, lat = torch_accuracy_latency(pruned, X, y)
    record("pruned", acc, lat, file_size_mb(pf_path),
          note=f"pruned + {ft_epochs}-epoch fine-tune recovery")

    # --- INT8 dynamic (on the recovered pruned model) ---
    q = opt.quantize_dynamic(pruned)
    q_path = os.path.join(workdir, f"{name}_int8.pt")
    torch.save(q.state_dict(), q_path)
    acc, lat = torch_accuracy_latency(q, X, y)
    record("int8", acc, lat, file_size_mb(q_path))

    # --- ONNX Runtime INT8 ---
    try:
        onnx_fp32 = os.path.join(workdir, f"{name}.onnx")
        opt.export_onnx(pruned, (1, 3, IMGSZ, IMGSZ), onnx_fp32)
        onnx_int8 = opt.quantize_onnx(onnx_fp32)
        if onnx_int8:
            acc, lat = onnx_accuracy_latency(onnx_int8, X, y)
            record("onnx_int8", acc, lat, file_size_mb(onnx_int8))
    except Exception as exc:
        print(f"  onnx_int8  FAILED: {str(exc)[:100]}")

    for r in rows:
        r["accuracy_retention"] = (r["accuracy"] / baseline_acc
                                   if baseline_acc else None)
    return rows


def bench_yolo_cls(X, y, classes, workdir: str) -> list[dict]:
    """YOLOv8-cls: PyTorch vs exported ONNX."""
    from ultralytics import YOLO
    w = "weights/wheel_yolov8_cls.pt"
    if not os.path.isfile(w):
        return []
    rows = []
    model = YOLO(w)

    # Map YOLO's own class order onto the ImageFolder order.
    yolo_names = [model.names[i] for i in sorted(model.names)]
    idx_map = {i: (classes.index(n) if n in classes else i)
               for i, n in enumerate(yolo_names)}

    arr = (X.numpy().transpose(0, 2, 3, 1) * 255).astype("uint8")
    preds, lat = [], []
    for i in range(len(arr)):
        t0 = time.perf_counter()
        r = model.predict(arr[i], imgsz=IMGSZ, device="cpu", verbose=False)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(idx_map.get(int(r.probs.top1), int(r.probs.top1)))
    acc = float((np.array(preds) == y).mean())
    stats = system_stats()
    rows.append({"model": "YOLOv8-cls", "config": "fp32", "accuracy": acc,
                 "latency_ms": float(np.mean(lat)),
                 "fps": 1000.0 / float(np.mean(lat)),
                 "size_mb": file_size_mb(w), "ram_mb": stats["ram_mb"],
                 "cpu_percent": stats["cpu_percent"],
                 "accuracy_retention": 1.0, "note": "ultralytics"})
    print(f"  {'fp32':10s} acc={acc:.3f} lat={np.mean(lat):6.2f}ms")

    try:
        onnx_path = model.export(format="onnx", imgsz=IMGSZ, device="cpu",
                                 verbose=False)
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        sess = ort.InferenceSession(str(onnx_path), so,
                                    providers=["CPUExecutionProvider"])
        nm = sess.get_inputs()[0].name
        xin = X.numpy()
        p2, l2 = [], []
        for i in range(len(xin)):
            t0 = time.perf_counter()
            out = sess.run(None, {nm: xin[i:i + 1]})[0]
            l2.append((time.perf_counter() - t0) * 1000)
            p2.append(idx_map.get(int(np.argmax(out, axis=1)[0]),
                                  int(np.argmax(out, axis=1)[0])))
        acc2 = float((np.array(p2) == y).mean())
        stats = system_stats()
        rows.append({"model": "YOLOv8-cls", "config": "onnx_fp32",
                     "accuracy": acc2, "latency_ms": float(np.mean(l2)),
                     "fps": 1000.0 / float(np.mean(l2)),
                     "size_mb": file_size_mb(str(onnx_path)),
                     "ram_mb": stats["ram_mb"],
                     "cpu_percent": stats["cpu_percent"],
                     "accuracy_retention": acc2 / acc if acc else None,
                     "note": "ultralytics export"})
        print(f"  {'onnx_fp32':10s} acc={acc2:.3f} lat={np.mean(l2):6.2f}ms")
    except Exception as exc:
        print(f"  onnx       FAILED: {str(exc)[:100]}")
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200,
                    help="validation images to evaluate per config")
    ap.add_argument("--ft-epochs", type=int, default=3,
                    help="fine-tune epochs to recover pruned accuracy")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    workdir = os.path.join(OUT_DIR, "artifacts")
    os.makedirs(workdir, exist_ok=True)

    X, y, classes = load_val(IMGSZ, args.limit)
    print(f"evaluating on {len(X)} validation crops, classes={classes}\n")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    train_tf = transforms.Compose([transforms.Resize((IMGSZ, IMGSZ)),
                                   transforms.ToTensor()])
    train_loader = DataLoader(
        datasets.ImageFolder("data/wheels/train", train_tf),
        batch_size=32, shuffle=True, num_workers=2)

    rows: list[dict] = []

    if os.path.isfile("weights/wheel_cnn.pt"):
        print("=== SmallCNN ===")
        rows += bench_torch_model("SmallCNN", lambda n: SmallCNN(num_classes=n),
                                  "weights/wheel_cnn.pt", X, y, len(classes),
                                  workdir, train_loader, args.ft_epochs)

    if os.path.isfile("weights/wheel_cspnext.pt"):
        print("\n=== RTMDet-CSPNeXt ===")
        from src.models.cspnext import CSPNeXtClassifier
        rows += bench_torch_model("RTMDet-CSPNeXt",
                                  lambda n: CSPNeXtClassifier(num_classes=n),
                                  "weights/wheel_cspnext.pt", X, y,
                                  len(classes), workdir, train_loader,
                                  args.ft_epochs)

    print("\n=== YOLOv8-cls ===")
    rows += bench_yolo_cls(X, y, classes, workdir)

    with open(f"{OUT_DIR}/benchmark.json", "w") as fh:
        json.dump({"n_eval_images": len(X), "classes": classes,
                   "rows": rows}, fh, indent=2)

    print("\n=== Phase 4 summary (accuracy retention vs FP32) ===")
    print(f"{'model':16s}{'config':11s}{'acc':>7s}{'ret':>7s}"
          f"{'ms':>8s}{'FPS':>7s}{'MB':>8s}")
    for r in rows:
        ret = r.get("accuracy_retention")
        print(f"{r['model']:16s}{r['config']:11s}{r['accuracy']:7.3f}"
              f"{(ret if ret else 0):7.2f}{r['latency_ms']:8.2f}"
              f"{(r['fps'] or 0):7.1f}{r['size_mb']:8.2f}")
    print(f"\nsaved -> {OUT_DIR}/benchmark.json")


if __name__ == "__main__":
    main()
