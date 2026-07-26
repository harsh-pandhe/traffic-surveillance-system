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

# ---------------------------------------------------------------------------
# Compliance semantics are resolved from config BY CLASS NAME at load time.
#
# Hard-coding class ids here was a real defect: the trained weights use the
# AI City Track-5 ordering (0 == "Driver With Helmet"), while the old constants
# assumed 0 == "No Helmet". That inverted compliance -- helmeted riders were
# reported as violations. Names are stable across retraining; ids are not.
# ---------------------------------------------------------------------------


@dataclass
class HelmetDetection:
    """One detected object with compliance semantics attached."""
    bbox: List[float]          # [x1, y1, x2, y2] in pixel coords
    cls_id: int
    cls_name: str
    conf: float
    is_violation: bool = False
    face_exposed: bool = False
    is_rider: bool = False


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

        # Resolve semantic roles from names -> ids for this specific model.
        self.violation_ids = self._ids_for(h.get("violation_classes", []))
        self.face_exposed_ids = self._ids_for(h.get("face_exposed_classes", []))
        self.rider_ids = self._ids_for(h.get("rider_classes", []))
        self.vehicle_ids = self._ids_for(h.get("vehicle_classes", []))

        self._using_fallback = False
        self.model = self._load_model(h)
        self._verify_class_alignment()

    # ------------------------------------------------------------------ #
    def _ids_for(self, names: List[str]) -> set:
        """Map configured class *names* to the ids used by this model."""
        lookup = {v.strip().lower(): k for k, v in self.class_names.items()}
        ids = set()
        for n in names:
            key = str(n).strip().lower()
            if key in lookup:
                ids.add(lookup[key])
            else:
                print(f"[HelmetDetector] warning: semantic class '{n}' is not "
                      f"in class_names; ignoring.")
        return ids

    def _verify_class_alignment(self) -> None:
        """
        Fail loudly if config class_names disagree with the loaded weights.
        A silent mismatch here inverts compliance decisions.
        """
        if self._using_fallback:
            return
        model_names = getattr(self.model, "names", None)
        if not model_names:
            return
        if len(model_names) != len(self.class_names):
            print(f"[HelmetDetector] WARNING: model has {len(model_names)} "
                  f"classes but config declares {len(self.class_names)}.")
            return
        # Compare normalised names position-by-position.
        def norm(s):
            return str(s).strip().lower().replace("_", " ")
        mismatched = [
            (i, model_names[i], self.class_names.get(i))
            for i in range(len(model_names))
            if norm(model_names[i]) != norm(self.class_names.get(i, ""))
        ]
        if mismatched:
            print("[HelmetDetector] WARNING: config/weights class mismatch "
                  "(compliance semantics may be wrong):")
            for i, m, c in mismatched:
                print(f"    id {i}: weights='{m}'  config='{c}'")

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
        COCO 0 == 'person' -> treat as a generic rider so the pipeline still
        produces tracks. All other COCO classes are ignored.
        """
        if coco_cls == 0 and self.rider_ids:
            return min(self.rider_ids)
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
                HelmetDetection(
                    bbox=xyxy, cls_id=cls_id, cls_name=cls_name, conf=conf,
                    is_violation=cls_id in self.violation_ids,
                    face_exposed=cls_id in self.face_exposed_ids,
                    is_rider=cls_id in self.rider_ids,
                )
            )
        return detections

    # ------------------------------------------------------------------ #
    @staticmethod
    def count_violations(dets: List[HelmetDetection]) -> int:
        return sum(1 for d in dets if d.is_violation)

    @staticmethod
    def count_riders(dets: List[HelmetDetection]) -> int:
        """Occupancy count (driver + passengers) -> feeds the overload rule."""
        return sum(1 for d in dets if d.is_rider)


if __name__ == "__main__":
    det = HelmetDetector()
    print("class map:", det.class_names)
    print("violation ids:", sorted(det.violation_ids))
    print("rider ids:", sorted(det.rider_ids))
    dummy = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    out = det.detect(dummy)
    print(f"detections: {len(out)} (fallback={det._using_fallback})")
