"""
src/models/helmet_detector.py
------------------------------
Phase 2 - Granular helmet-compliance detection.

Wraps an Ultralytics YOLOv8/YOLOv11 detector configured for 7 classes:

    0: No Helmet
    1: Full-Face Helmet
    2: Half-Face Helmet
    3: Strap Unfastened
    4: Helmet on Handlebar
    5: Helmet on Arm
    6: Rider

If the fine-tuned 7-class weights are missing, the detector transparently falls
back to a generic COCO YOLOv8n model so the rest of the pipeline still runs (the
class map is then a best-effort remap that flags every person as a Rider). This
keeps the codebase runnable out-of-the-box while remaining production-shaped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from utils.config import load_config, resolve_path

# Compliance semantics ------------------------------------------------------
NO_HELMET = 0
FULL_FACE = 1
HALF_FACE = 2
STRAP_UNFASTENED = 3
HELMET_ON_HANDLEBAR = 4
HELMET_ON_ARM = 5
RIDER = 6

# Which classes count as a *violation* for the risk indexer.
VIOLATION_CLASSES = {
    NO_HELMET,
    STRAP_UNFASTENED,
    HELMET_ON_HANDLEBAR,
    HELMET_ON_ARM,
}
# Classes where the face is exposed -> demographics can run.
FACE_EXPOSED_CLASSES = {NO_HELMET, HALF_FACE, STRAP_UNFASTENED}


@dataclass
class HelmetDetection:
    """One detected object with compliance semantics attached."""
    bbox: List[float]          # [x1, y1, x2, y2] in pixel coords
    cls_id: int
    cls_name: str
    conf: float

    @property
    def is_violation(self) -> bool:
        return self.cls_id in VIOLATION_CLASSES

    @property
    def face_exposed(self) -> bool:
        return self.cls_id in FACE_EXPOSED_CLASSES


class HelmetDetector:
    """YOLOv8-based granular helmet compliance detector."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        h = cfg["helmet_detector"]
        self.conf_th = float(h["conf_threshold"])
        self.iou_th = float(h["iou_threshold"])
        self.imgsz = int(h["imgsz"])
        self.class_names = {int(k): v for k, v in h["class_names"].items()}
        self.device = cfg["runtime"]["device"]

        self._using_fallback = False
        self.model = self._load_model(h)

    # ------------------------------------------------------------------ #
    def _load_model(self, h: dict):
        from ultralytics import YOLO  # local import -> optional dependency

        custom = resolve_path(h["weights"])
        if os.path.isfile(custom):
            return YOLO(custom)

        # Fall back to a stock model (auto-downloaded by ultralytics).
        self._using_fallback = True
        fallback = h.get("fallback_weights", "yolov8n.pt")
        print(
            f"[HelmetDetector] custom weights '{custom}' not found; "
            f"falling back to '{fallback}' (person->Rider remap)."
        )
        return YOLO(fallback)

    # ------------------------------------------------------------------ #
    def _remap_fallback_cls(self, coco_cls: int) -> int | None:
        """
        Map generic COCO classes to our schema when running the fallback model.
        COCO 0 == 'person' -> treat as Rider so the pipeline still produces
        tracks and risk scores. All other COCO classes are ignored.
        """
        if coco_cls == 0:
            return RIDER
        return None

    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> List[HelmetDetection]:
        """Run detection on a BGR frame and return HelmetDetection objects."""
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_th,
            iou=self.iou_th,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: List[HelmetDetection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        for box in r.boxes:
            raw_cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy.squeeze().tolist()

            if self._using_fallback:
                mapped = self._remap_fallback_cls(raw_cls)
                if mapped is None:
                    continue
                cls_id = mapped
            else:
                cls_id = raw_cls

            cls_name = self.class_names.get(cls_id, str(cls_id))
            detections.append(
                HelmetDetection(bbox=xyxy, cls_id=cls_id,
                                cls_name=cls_name, conf=conf)
            )
        return detections

    # ------------------------------------------------------------------ #
    @staticmethod
    def count_violations(dets: List[HelmetDetection]) -> int:
        return sum(1 for d in dets if d.is_violation)

    @staticmethod
    def count_riders(dets: List[HelmetDetection]) -> int:
        return sum(1 for d in dets if d.cls_id == RIDER)


if __name__ == "__main__":
    det = HelmetDetector()
    dummy = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    out = det.detect(dummy)
    print(f"detections: {len(out)} (fallback={det._using_fallback})")
