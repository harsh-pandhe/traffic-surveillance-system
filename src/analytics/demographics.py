"""
src/analytics/demographics.py
------------------------------
Phase 3 - Rider demographics (age group + gender).

For riders whose face is exposed (no helmet / half-face / strap unfastened), a
face crop is extracted and passed to DeepFace (which wraps FairFace-style age &
gender models). This supports the research goal of profiling non-compliant riders.

Graceful degradation: if DeepFace is not installed, `estimate()` returns an
`UNKNOWN` result instead of crashing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        d = cfg["demographics"]
        self.enabled = bool(d["enabled"])
        self.detector_backend = d["detector_backend"]
        self.min_face_size = int(d["min_face_size"])
        self._deepface = self._init_backend() if self.enabled else None

    def _init_backend(self):
        try:
            from deepface import DeepFace
            print("[DemographicsEstimator] DeepFace backend ready.")
            return DeepFace
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[DemographicsEstimator] DeepFace unavailable ({exc}); "
                  f"demographics disabled.")
            return None

    # ------------------------------------------------------------------ #
    def estimate(self, face_crop_bgr: np.ndarray) -> Demographic:
        """Estimate age group + gender for a single face crop (BGR)."""
        if (not self.enabled or self._deepface is None
                or face_crop_bgr is None or face_crop_bgr.size == 0):
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
