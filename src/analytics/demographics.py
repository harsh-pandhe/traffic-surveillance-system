"""
src/analytics/demographics.py
------------------------------
Phase 3 - Rider demographics (age group + gender).

For riders whose face is exposed (no helmet / half-face / strap unfastened), a
face crop is extracted and passed to DeepFace (which wraps FairFace-style age &
gender models). This supports the research goal of profiling non-compliant riders.

Graceful degradation: if DeepFace is not installed, `estimate()` returns an
`UNKNOWN` result instead of crashing the pipeline.

IMPORTANT - process isolation. On this hardware, torch's bundled CUDA runtime
targets a newer NVIDIA driver than is installed ("CUDA initialization: The
NVIDIA driver on your system is too old"). Loading a *second* Ultralytics/torch
model in a process that has also imported TensorFlow (DeepFace's backend)
segfaults -- confirmed by bisection: HelmetDetector's YOLO load is fine,
WheelClassifier's second YOLO load is not, every time, regardless of import
order or CUDA_VISIBLE_DEVICES. This is a native-library conflict below what
Python-level fixes can safely patch. DeepFace works perfectly in a process that
never imports torch (verified). So `estimate()` in-process is guarded off
whenever torch is already loaded; the live pipeline instead calls
`save_for_offline_analysis()` to queue face crops, and
`scripts/run_demographics.py` (which never imports torch) processes the queue
afterward. This keeps demographics honest and working without risking the
pipeline that helmet/wheel detection and tracking depend on.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from utils.config import load_config

# Coarse age buckets for reporting.
AGE_BUCKETS = [(0, 12, "Child"), (13, 19, "Teen"),
               (20, 39, "Adult"), (40, 59, "Middle-Aged"), (60, 200, "Senior")]


@dataclass
class Demographic:
    age: Optional[int]
    age_group: str
    gender: str
    confidence: float

    @classmethod
    def unknown(cls) -> "Demographic":
        return cls(age=None, age_group="UNKNOWN", gender="UNKNOWN", confidence=0.0)


def _age_to_group(age: Optional[float]) -> str:
    if age is None:
        return "UNKNOWN"
    for lo, hi, name in AGE_BUCKETS:
        if lo <= age <= hi:
            return name
    return "UNKNOWN"


class DemographicsEstimator:
    """Age / gender estimation for exposed-face rider crops."""

    def __init__(self, config: dict | None = None, queue_dir: str | None = None):
        cfg = config or load_config()
        d = cfg["demographics"]
        self.enabled = bool(d["enabled"])
        self.detector_backend = d["detector_backend"]
        self.min_face_size = int(d["min_face_size"])
        self.queue_dir = queue_dir or d.get("queue_dir", "outputs/demographics_queue")
        self._deepface = self._init_backend() if self.enabled else None

    def _init_backend(self):
        if "torch" in sys.modules:
            # See the module docstring: torch + TensorFlow in one process
            # segfaults on this hardware once a second torch model loads. Queue
            # crops for the standalone script instead of risking the pipeline.
            print("[DemographicsEstimator] torch is already loaded in this "
                 "process; running DeepFace here would risk a segfault "
                 "(confirmed on this hardware). Queuing face crops to "
                 f"'{self.queue_dir}/' instead -- run "
                 "`python scripts/run_demographics.py` afterward.")
            return None
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            from deepface import DeepFace
            print("[DemographicsEstimator] DeepFace backend ready (CPU-only).")
            return DeepFace
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[DemographicsEstimator] DeepFace unavailable ({exc}); "
                  f"demographics disabled.")
            return None

    # ------------------------------------------------------------------ #
    def save_for_offline_analysis(self, face_crop_bgr: np.ndarray,
                                  track_id: int | str = "na") -> Optional[str]:
        """
        Queue a face crop for the standalone script instead of analysing it
        in-process. Used automatically when in-process DeepFace is unavailable
        (see `_init_backend`); safe to call directly too.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        h, w = face_crop_bgr.shape[:2]
        if min(h, w) < self.min_face_size:
            return None
        os.makedirs(self.queue_dir, exist_ok=True)
        fname = f"{int(time.time() * 1000)}_{track_id}.jpg"
        path = os.path.join(self.queue_dir, fname)
        cv2.imwrite(path, face_crop_bgr)
        return path

    # ------------------------------------------------------------------ #
    def estimate(self, face_crop_bgr: np.ndarray,
                track_id: int | str = "na") -> Demographic:
        """
        Estimate age group + gender for a single face crop (BGR).

        If DeepFace could not be loaded in-process (see `_init_backend`), the
        crop is transparently queued for `scripts/run_demographics.py` instead
        of being silently dropped, and this returns UNKNOWN for the live frame.
        """
        if not self.enabled or face_crop_bgr is None or face_crop_bgr.size == 0:
            return Demographic.unknown()
        if self._deepface is None:
            self.save_for_offline_analysis(face_crop_bgr, track_id)
            return Demographic.unknown()

        h, w = face_crop_bgr.shape[:2]
        if min(h, w) < self.min_face_size:
            return Demographic.unknown()

        try:
            res = self._deepface.analyze(
                img_path=face_crop_bgr,
                actions=["age", "gender"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )
            r = res[0] if isinstance(res, list) else res
            age = float(r.get("age", None)) if r.get("age") is not None else None
            gender = r.get("dominant_gender", "UNKNOWN")
            gconf = 0.0
            if isinstance(r.get("gender"), dict) and gender in r["gender"]:
                gconf = float(r["gender"][gender]) / 100.0
            return Demographic(
                age=int(age) if age is not None else None,
                age_group=_age_to_group(age),
                gender=gender,
                confidence=gconf,
            )
        except Exception as exc:  # pragma: no cover - runtime robustness
            print(f"[DemographicsEstimator] analyze failed: {exc}")
            return Demographic.unknown()

    # ------------------------------------------------------------------ #
    @staticmethod
    def crop_face_region(frame_bgr: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract the upper portion of a rider bbox as an approximate face ROI
        (top ~35% of the body box), clamped to frame bounds.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        face_h = int((y2 - y1) * 0.35)
        return frame_bgr[y1:y1 + max(face_h, 1), x1:x2].copy()


if __name__ == "__main__":
    est = DemographicsEstimator()
    dummy_face = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    print(est.estimate(dummy_face))
