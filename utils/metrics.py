"""
utils/metrics.py
----------------
Phase 4 - Benchmarking utilities.

Provides:
    * detection metrics : precision, recall, mAP@50 from predictions vs GT boxes
    * runtime metrics   : inference latency (ms), FPS via a rolling timer
    * system metrics    : RAM (RSS) and CPU utilisation via psutil
    * a comparison table (before vs after optimization)

Everything is dependency-light; psutil is optional (guarded).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    import psutil
    _PSUTIL = True
except Exception:  # pragma: no cover
    _PSUTIL = False


# ========================================================================== #
# Runtime timing                                                             #
# ========================================================================== #
class LatencyMeter:
    """Rolling latency / FPS meter."""

    def __init__(self, window: int = 100):
        self._samples: List[float] = []
        self._window = window
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Record and return the last interval in milliseconds."""
        if self._t0 is None:
            return 0.0
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._samples.append(dt_ms)
        if len(self._samples) > self._window:
            self._samples.pop(0)
        self._t0 = None
        return dt_ms

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0

    @property
    def fps(self) -> float:
        avg = self.avg_latency_ms
        return 1000.0 / avg if avg > 0 else 0.0


# ========================================================================== #
# System telemetry                                                           #
# ========================================================================== #
def system_stats() -> Dict[str, float]:
    """Return current RAM (MB) and CPU (%) for this process."""
    if not _PSUTIL:
        return {"ram_mb": 0.0, "cpu_percent": 0.0}
    proc = psutil.Process()
    ram_mb = proc.memory_info().rss / (1024 ** 2)
    cpu = proc.cpu_percent(interval=0.1)
    return {"ram_mb": ram_mb, "cpu_percent": cpu}


# ========================================================================== #
# Detection accuracy: precision / recall / mAP@50                            #
# ========================================================================== #
def _iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6)
    return inter / ua


@dataclass
class Detection:
    box: List[float]
    cls_id: int
    score: float


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated Average Precision (COCO-style)."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def compute_map50(
    predictions: Dict[int, List[Detection]],
    ground_truth: Dict[int, List[Tuple[List[float], int]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Precision, Recall and mAP@50 over a set of images.

    Args:
        predictions:  {image_id: [Detection, ...]}
        ground_truth: {image_id: [(box, cls_id), ...]}
    Returns:
        {"precision", "recall", "mAP@50"}
    """
    # Group by class.
    class_ids = set()
    for dets in predictions.values():
        class_ids.update(d.cls_id for d in dets)
    for gts in ground_truth.values():
        class_ids.update(c for _b, c in gts)

    aps, total_tp, total_fp, total_fn = [], 0, 0, 0

    for cls in sorted(class_ids):
        # Collect predictions for this class across all images, sorted by score.
        preds: List[Tuple[int, Detection]] = []
        for img_id, dets in predictions.items():
            for d in dets:
                if d.cls_id == cls:
                    preds.append((img_id, d))
        preds.sort(key=lambda p: p[1].score, reverse=True)

        # Ground-truth boxes for this class, with a matched flag per image.
        gt_boxes: Dict[int, List[List[float]]] = defaultdict(list)
        n_gt = 0
        for img_id, gts in ground_truth.items():
            for box, c in gts:
                if c == cls:
                    gt_boxes[img_id].append(box)
                    n_gt += 1
        matched: Dict[int, set] = defaultdict(set)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for i, (img_id, det) in enumerate(preds):
            best_iou, best_j = 0.0, -1
            for j, gbox in enumerate(gt_boxes.get(img_id, [])):
                if j in matched[img_id]:
                    continue
                iou = _iou(det.box, gbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j >= 0:
                tp[i] = 1
                matched[img_id].add(best_j)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (n_gt + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps.append(_ap_from_pr(recalls, precisions) if len(preds) else 0.0)

        total_tp += int(tp.sum())
        total_fp += int(fp.sum())
        total_fn += n_gt - int(tp.sum())

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    map50 = float(np.mean(aps)) if aps else 0.0
    return {"precision": precision, "recall": recall, "mAP@50": map50}


# ========================================================================== #
# Before/after comparison report                                            #
# ========================================================================== #
@dataclass
class BenchmarkReport:
    rows: List[Dict[str, float]] = field(default_factory=list)

    def add(self, label: str, latency_ms: float, fps: float,
            ram_mb: float, cpu_percent: float,
            map50: float | None = None) -> None:
        self.rows.append({
            "config": label,
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 2),
            "ram_mb": round(ram_mb, 1),
            "cpu_percent": round(cpu_percent, 1),
            "mAP@50": round(map50, 4) if map50 is not None else None,
        })

    def render(self) -> str:
        try:
            from tabulate import tabulate
            return tabulate(self.rows, headers="keys", tablefmt="github")
        except Exception:  # pragma: no cover
            lines = [str(self.rows[0].keys())] + [str(r.values()) for r in self.rows]
            return "\n".join(lines)


if __name__ == "__main__":
    # mAP smoke test: perfect prediction => mAP 1.0
    gt = {0: [([10, 10, 50, 50], 0)]}
    pred = {0: [Detection([12, 12, 52, 52], 0, 0.9)]}
    print(compute_map50(pred, gt))
    print("system:", system_stats())
