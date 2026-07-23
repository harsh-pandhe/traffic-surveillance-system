"""
src/tracking/deepsort_tracker.py
---------------------------------
Phase 3 - Multi-object tracking across consecutive frames.

Wraps `deep-sort-realtime` to maintain stable track IDs through occlusion, and
adds a temporal voting buffer so a vehicle's wheel-count class is smoothed over
`vote_window` frames (a single-frame misclassification cannot flip the verdict).

If `deep-sort-realtime` is unavailable, a minimal IoU-based tracker is used as a
fallback so the pipeline still runs end-to-end.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.config import load_config


@dataclass
class Track:
    """A confirmed track with its temporal wheel-count vote buffer."""
    track_id: int
    bbox: List[float]                       # [x1, y1, x2, y2]
    cls_id: int
    votes: Deque[int] = field(default_factory=lambda: deque(maxlen=15))

    def majority_wheel_class(self) -> Optional[int]:
        """Return the most-voted wheel class across the temporal window."""
        if not self.votes:
            return None
        counts = np.bincount(np.asarray(self.votes))
        return int(np.argmax(counts))


# -------------------------------------------------------------------------- #
# Fallback IoU tracker (used only if deep-sort-realtime is missing)          #
# -------------------------------------------------------------------------- #
def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


class _SimpleIoUTracker:
    def __init__(self, iou_th: float = 0.3, max_age: int = 30):
        self.iou_th = iou_th
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}

    def update(self, detections: List[Tuple[List[float], float, int]]):
        assigned = {}
        used = set()
        for tid, tr in self.tracks.items():
            best_iou, best_j = 0.0, -1
            for j, (bbox, _c, _cls) in enumerate(detections):
                if j in used:
                    continue
                i = _iou(tr["bbox"], bbox)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= self.iou_th and best_j >= 0:
                used.add(best_j)
                assigned[tid] = detections[best_j]

        # age / update existing tracks
        for tid in list(self.tracks.keys()):
            if tid in assigned:
                bbox, _c, cls = assigned[tid]
                self.tracks[tid].update(bbox=bbox, cls=cls, age=0)
            else:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]

        # spawn new tracks
        for j, (bbox, _c, cls) in enumerate(detections):
            if j not in used:
                self.tracks[self.next_id] = {"bbox": bbox, "cls": cls, "age": 0}
                self.next_id += 1

        return [(tid, tr["bbox"], tr["cls"]) for tid, tr in self.tracks.items()]


# -------------------------------------------------------------------------- #
# Public tracker                                                             #
# -------------------------------------------------------------------------- #
class VehicleTracker:
    """DeepSORT tracker with temporal wheel-count voting."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        t = cfg["tracker"]
        self.vote_window = int(t["vote_window"])
        self.max_age = int(t["max_age"])
        self.n_init = int(t["n_init"])
        self.max_iou_distance = float(t["max_iou_distance"])
        self.embedder = t["embedder"]

        self._backend, self._impl = self._init_backend()
        # Persistent vote buffers keyed by track id.
        self._vote_buffers: Dict[int, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.vote_window)
        )

    def _init_backend(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            impl = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                embedder=self.embedder,
                half=False,
                bgr=True,
            )
            return "deepsort", impl
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[VehicleTracker] deep-sort-realtime unavailable ({exc}); "
                  f"using fallback IoU tracker.")
            return "iou", _SimpleIoUTracker(
                iou_th=1.0 - self.max_iou_distance, max_age=self.max_age
            )

    # ------------------------------------------------------------------ #
    def update(self, frame_bgr: np.ndarray,
               detections: List[Tuple[List[float], float, int]]) -> List[Track]:
        """
        Args:
            frame_bgr:  current frame (needed by DeepSORT embedder).
            detections: list of (bbox[x1,y1,x2,y2], conf, wheel_cls_id).
        Returns:
            list of confirmed Track objects with smoothed wheel votes.
        """
        if self._backend == "deepsort":
            confirmed = self._update_deepsort(frame_bgr, detections)
        else:
            confirmed = self._update_iou(detections)
        return confirmed

    def _update_deepsort(self, frame, detections) -> List[Track]:
        # DeepSORT expects [ [x, y, w, h], conf, class ] tuples.
        ds_dets = []
        det_cls = []
        for bbox, conf, cls in detections:
            x1, y1, x2, y2 = bbox
            ds_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, str(cls)))
            det_cls.append(cls)

        tracks = self._impl.update_tracks(ds_dets, frame=frame)
        out: List[Track] = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = int(tr.track_id)
            l, t, r, b = tr.to_ltrb()
            # DeepSORT carries our class string through det_class.
            cls_id = self._safe_int(tr.get_det_class(), default=0)
            self._vote_buffers[tid].append(cls_id)
            out.append(Track(track_id=tid, bbox=[l, t, r, b],
                              cls_id=cls_id, votes=self._vote_buffers[tid]))
        return out

    def _update_iou(self, detections) -> List[Track]:
        results = self._impl.update(detections)
        out: List[Track] = []
        for tid, bbox, cls_id in results:
            self._vote_buffers[tid].append(int(cls_id))
            out.append(Track(track_id=tid, bbox=bbox,
                              cls_id=int(cls_id), votes=self._vote_buffers[tid]))
        return out

    @staticmethod
    def _safe_int(val, default: int = 0) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return default


if __name__ == "__main__":
    trk = VehicleTracker()
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    dets = [([100, 100, 200, 260], 0.9, 0), ([300, 120, 420, 300], 0.8, 2)]
    for step in range(3):
        tracks = trk.update(frame, dets)
        print(f"step {step}: {[(t.track_id, t.majority_wheel_class()) for t in tracks]}")
