"""
scripts/train_wheel_classifier.py
----------------------------------
Phase 2 - wheel-count classification: train + benchmark two models.

    A) SmallCNN  (custom PyTorch baseline, src/models/benchmark_classifier.py)
    B) YOLOv8-cls (Ultralytics, fine-tuned)

Both consume the ImageFolder layout produced by build_wheel_dataset.py
(data/wheels/{train,val}/<class>/*.jpg). Reports accuracy, macro
Precision/Recall/F1, per-class metrics, and mean inference latency for each,
then names the better model. Writes weights + metrics.json.

CPU-friendly: small net, few epochs, capped image size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.benchmark_classifier import SmallCNN

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report)

OUT = "outputs/phase2"
os.makedirs(OUT, exist_ok=True)


def loaders(root, imgsz, batch):
    tf_train = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])
    tr = datasets.ImageFolder(os.path.join(root, "train"), tf_train)
    va = datasets.ImageFolder(os.path.join(root, "val"), tf_val)
    return (DataLoader(tr, batch, shuffle=True, num_workers=2),
            DataLoader(va, batch, shuffle=False, num_workers=2),
            tr.classes)


def train_cnn(root, imgsz, epochs, batch, lr):
    dev = torch.device("cpu")
    dl_tr, dl_va, classes = loaders(root, imgsz, batch)
    model = SmallCNN(num_classes=len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        print(f"  [CNN] epoch {ep+1}/{epochs}  loss={tot/len(dl_tr.dataset):.3f}")

    # Evaluate
    model.eval()
    ys, ps, lat = [], [], []
    with torch.no_grad():
        for x, y in dl_va:
            t0 = time.perf_counter()
            out = model(x)
            lat.append((time.perf_counter() - t0) / x.size(0) * 1000)
            ps.extend(out.argmax(1).tolist()); ys.extend(y.tolist())
    torch.save(model.state_dict(), "weights/wheel_cnn.pt")
    return _metrics("SmallCNN", ys, ps, classes, float(np.mean(lat)))


def train_yolo(root, imgsz, epochs, batch):
    from ultralytics import YOLO
    model = YOLO("yolov8n-cls.pt")
    model.train(data=os.path.abspath(root), epochs=epochs, imgsz=imgsz,
                batch=batch, device="cpu", verbose=False,
                project=OUT, name="yolo_cls", exist_ok=True)

    # Evaluate on val split with the same protocol.
    classes = sorted(os.listdir(os.path.join(root, "val")))
    ys, ps, lat = [], [], []
    for ci, c in enumerate(classes):
        cdir = os.path.join(root, "val", c)
        for f in os.listdir(cdir):
            t0 = time.perf_counter()
            r = model.predict(os.path.join(cdir, f), imgsz=imgsz,
                              device="cpu", verbose=False)
            lat.append((time.perf_counter() - t0) * 1000)
            ys.append(ci); ps.append(int(r[0].probs.top1))
    return _metrics("YOLOv8-cls", ys, ps, classes, float(np.mean(lat)))


def _metrics(name, ys, ps, classes, latency_ms):
    acc = accuracy_score(ys, ps)
    p, r, f1, _ = precision_recall_fscore_support(
        ys, ps, average="macro", zero_division=0)
    print(f"\n[{name}] acc={acc:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} "
          f"lat={latency_ms:.1f}ms")
    print(classification_report(ys, ps, target_names=classes, digits=3,
                                zero_division=0))
    return {"model": name, "accuracy": acc, "precision": p, "recall": r,
            "f1": f1, "latency_ms": latency_ms, "classes": classes,
            "report": classification_report(ys, ps, target_names=classes,
                                             output_dict=True, zero_division=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/wheels")
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--yolo-epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs("weights", exist_ok=True)
    print("=== Training SmallCNN baseline ===")
    cnn = train_cnn(args.root, args.imgsz, args.epochs, args.batch, args.lr)

    print("\n=== Training YOLOv8-cls ===")
    try:
        yolo = train_yolo(args.root, args.imgsz, args.yolo_epochs, args.batch)
    except Exception as exc:
        print(f"[YOLOv8-cls] training failed: {exc}")
        yolo = None

    results = {"cnn": cnn, "yolo": yolo}
    if yolo:
        winner = "SmallCNN" if cnn["accuracy"] >= yolo["accuracy"] else "YOLOv8-cls"
        results["selected"] = winner
        print(f"\n=== Selected model: {winner} ===")
    with open(f"{OUT}/wheel_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"metrics -> {OUT}/wheel_metrics.json")


if __name__ == "__main__":
    main()
