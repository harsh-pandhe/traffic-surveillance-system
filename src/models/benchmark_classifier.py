"""
src/models/benchmark_classifier.py
-----------------------------------
Phase 2 - Wheel-count classification with a benchmarking harness.

Two interchangeable backends classify a cropped vehicle into:
    0: 2-Wheeler   1: 3-Wheeler   2: 4-Wheeler   3: 6+ Wheeler

    * `SmallCNN`         - a compact custom PyTorch CNN baseline.
    * `YoloClassifier`   - Ultralytics YOLOv8-cls / RTMDet-cls wrapper.

`WheelClassifier` is the unified front-end selected via config
(`wheel_classifier.backend`). `benchmark()` runs both backends on the same batch
and reports accuracy + latency so the research report can compare them head-to-head.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import load_config, resolve_path

WHEEL_CLASS_NAMES = {0: "2-Wheeler", 1: "3-Wheeler", 2: "4-Wheeler", 3: "6+ Wheeler"}
NUM_CLASSES = 4


# ========================================================================== #
# Custom CNN baseline                                                        #
# ========================================================================== #
class SmallCNN(nn.Module):
    """Lightweight 4-block CNN for wheel-count classification (CPU-friendly)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # 128 -> 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # 64 -> 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # 32 -> 16
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                             # -> 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class WheelPrediction:
    cls_id: int
    cls_name: str
    conf: float


# ========================================================================== #
# CNN backend wrapper                                                        #
# ========================================================================== #
class CNNBackend:
    def __init__(self, weights: str, imgsz: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.model = SmallCNN().to(self.device)
        if weights and os.path.isfile(weights):
            state = torch.load(weights, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"[CNNBackend] loaded weights: {weights}")
        else:
            print("[CNNBackend] no weights found; using randomly-initialised net "
                  "(demo mode - train before real use).")
        self.model.eval()

    def _preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        batch = []
        for c in crops:
            img = cv2.resize(c, (self.imgsz, self.imgsz))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch.append(img.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(batch)).to(self.device)

    @torch.no_grad()
    def predict(self, crops: List[np.ndarray]) -> List[WheelPrediction]:
        if not crops:
            return []
        x = self._preprocess(crops)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        confs, ids = probs.max(dim=1)
        return [
            WheelPrediction(int(i), WHEEL_CLASS_NAMES[int(i)], float(c))
            for i, c in zip(ids.tolist(), confs.tolist())
        ]


# ========================================================================== #
# YOLO-cls / RTMDet backend wrapper                                          #
# ========================================================================== #
class YoloBackend:
    def __init__(self, weights: str, imgsz: int, device: str = "cpu"):
        from ultralytics import YOLO
        self.imgsz = imgsz
        self.device = device
        if weights and os.path.isfile(weights):
            self.model = YOLO(weights)
            print(f"[YoloBackend] loaded weights: {weights}")
        else:
            # yolov8n-cls.pt is auto-downloaded; it is ImageNet-pretrained.
            self.model = YOLO("yolov8n-cls.pt")
            print("[YoloBackend] custom weights missing; using yolov8n-cls.pt "
                  "(demo mode - fine-tune on wheel dataset for real use).")

    def predict(self, crops: List[np.ndarray]) -> List[WheelPrediction]:
        if not crops:
            return []
        preds: List[WheelPrediction] = []
        results = self.model.predict(
            crops, imgsz=self.imgsz, device=self.device, verbose=False
        )
        for r in results:
            probs = r.probs
            if probs is None:
                preds.append(WheelPrediction(0, WHEEL_CLASS_NAMES[0], 0.0))
                continue
            raw_id = int(probs.top1)
            # Map arbitrary backbone index into our 4-class space.
            cls_id = raw_id % NUM_CLASSES
            preds.append(
                WheelPrediction(cls_id, WHEEL_CLASS_NAMES[cls_id],
                                float(probs.top1conf))
            )
        return preds


# ========================================================================== #
# Unified front-end + benchmark harness                                      #
# ========================================================================== #
class WheelClassifier:
    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        w = cfg["wheel_classifier"]
        self.imgsz = int(w["imgsz"])
        self.device = cfg["runtime"]["device"]
        self.backend_name = w["backend"]
        self._cfg = w

        if self.backend_name == "cnn":
            self.backend = CNNBackend(resolve_path(w["cnn_weights"]),
                                      self.imgsz, self.device)
        else:
            self.backend = YoloBackend(resolve_path(w["yolo_weights"]),
                                       self.imgsz, self.device)

    def classify(self, crops: List[np.ndarray]) -> List[WheelPrediction]:
        return self.backend.predict(crops)

    def classify_one(self, crop: np.ndarray) -> WheelPrediction:
        out = self.backend.predict([crop])
        return out[0] if out else WheelPrediction(0, WHEEL_CLASS_NAMES[0], 0.0)

    # ------------------------------------------------------------------ #
    def benchmark(self, crops: List[np.ndarray],
                  labels: List[int] | None = None) -> Dict[str, Dict[str, float]]:
        """
        Run CNN and YOLO backends on the same crops, timing each and (if ground
        truth labels are supplied) computing accuracy. Returns a nested report.
        """
        cnn = CNNBackend(resolve_path(self._cfg["cnn_weights"]),
                         self.imgsz, self.device)
        yolo = YoloBackend(resolve_path(self._cfg["yolo_weights"]),
                           self.imgsz, self.device)

        report: Dict[str, Dict[str, float]] = {}
        for name, backend in (("cnn", cnn), ("yolo", yolo)):
            t0 = time.perf_counter()
            preds = backend.predict(crops)
            dt = time.perf_counter() - t0
            n = max(len(crops), 1)
            entry = {
                "total_ms": dt * 1000.0,
                "ms_per_image": dt * 1000.0 / n,
                "fps": n / dt if dt > 0 else 0.0,
            }
            if labels is not None and preds:
                correct = sum(1 for p, y in zip(preds, labels) if p.cls_id == y)
                entry["accuracy"] = correct / len(labels)
            report[name] = entry
        return report


if __name__ == "__main__":
    clf = WheelClassifier()
    fake = [(np.random.rand(128, 128, 3) * 255).astype(np.uint8) for _ in range(4)]
    print("single:", clf.classify_one(fake[0]))
    print("benchmark:", clf.benchmark(fake, labels=[0, 1, 2, 3]))
