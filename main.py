"""
main.py
-------
Pipeline orchestrator / entry point for the Adaptive Spatio-Temporal Traffic
Surveillance system.

Per-frame flow:
    1. Scene classify (DAY/NIGHT/FOG/RAIN)      [Phase 1]
    2. Adaptive enhance (CLAHE / dehaze / denoise)
    3. Detect vehicles + riders in one pass (7-class detector) [Phase 2]
    4. Wheel-count classification per vehicle crop
    5. Track VEHICLES + temporal wheel voting   [Phase 3]
    6. Helmet compliance, checked only within 2-wheeler tracks
       (this detector's driver/passenger x helmet-status taxonomy
       only applies to motorcycles; other wheel classes skip this check)
    7. Cross-camera ReID (OSNet global IDs)
    8. Demographics for exposed-face riders
    9. Risk index + banner                       [Phase 3]
    10. Visual overlays + telemetry, write output

Usage:
    python main.py --source data/raw/clip.mp4 --camera cam_A
    python main.py --source 0                       # webcam
    python main.py --source data/raw/clip.mp4 --optimize   # run Phase 4 demo first
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

from utils.config import load_config, resolve_path
from utils.metrics import LatencyMeter, system_stats
from utils import visualization as viz

from src.preprocessing.scene_classifier import SceneClassifier
from src.preprocessing.enhancements import FrameEnhancer
from src.models.helmet_detector import HelmetDetector
from src.models.benchmark_classifier import WheelClassifier, WHEEL_CLASS_NAMES
from src.tracking.deepsort_tracker import VehicleTracker
from src.tracking.multi_camera_reid import VehicleReID
from src.analytics.demographics import DemographicsEstimator
from src.analytics.risk_indexer import RiskIndexer


class SurveillancePipeline:
    """Wires every stage together and runs on a video source."""

    def __init__(self, config: dict | None = None, camera_id: str = "cam_A"):
        self.cfg = config or load_config()
        self.camera_id = camera_id

        # Instantiate every stage once (models loaded here).
        # Prefer the trained scene classifier when available; it internally
        # falls back to the rule-based one if no model file is present.
        self.scene = self._build_scene_classifier()
        self.enhancer = FrameEnhancer(self.cfg)
        self.helmet = HelmetDetector(self.cfg)
        self.wheels = WheelClassifier(self.cfg)
        self.tracker = VehicleTracker(self.cfg)
        self.reid = VehicleReID(self.cfg)
        self.demographics = DemographicsEstimator(self.cfg)
        self.risk = RiskIndexer(self.cfg)

        self.meter = LatencyMeter()
        # Centroid history per track id (for wrong-way detection).
        self._history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=30)
        )

    # ------------------------------------------------------------------ #
    def _build_scene_classifier(self):
        """Use the learned scene classifier if enabled, else the rule-based one."""
        ml_cfg = self.cfg.get("scene_classifier_ml", {})
        if ml_cfg.get("use_learned", True):
            from src.preprocessing.scene_classifier_ml import LearnedSceneClassifier
            clf = LearnedSceneClassifier(self.cfg)
            if clf.is_trained:
                print("[Pipeline] using learned scene classifier.")
                return clf
            print("[Pipeline] learned model absent; using rule-based scene classifier.")
        return SceneClassifier(self.cfg)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _crop(frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _bbox_center_in(outer: List[float], inner: List[float]) -> bool:
        """True if inner's box center falls within outer -- used to associate
        a rider detection with the vehicle track it is riding on."""
        ox1, oy1, ox2, oy2 = outer
        icx = (inner[0] + inner[2]) / 2.0
        icy = (inner[1] + inner[3]) / 2.0
        return ox1 <= icx <= ox2 and oy1 <= icy <= oy2

    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        self.meter.start()

        # --- Phase 1: scene + enhancement ---
        scene_res = self.scene.classify(frame)
        enhanced = self.enhancer.enhance(frame, scene_res.label)

        # --- Phase 2: single detector call yields vehicles + riders together ---
        dets = self.helmet.detect(enhanced)
        vehicle_dets = [d for d in dets if d.is_vehicle]
        rider_dets = [d for d in dets if d.is_rider]

        # --- Phase 3: track the VEHICLE boxes (not rider boxes) ---
        # Wheel-count classification has to run per-frame, before the box is
        # handed to the tracker, since the tracker's vote buffer only smooths
        # a classification already made -- it cannot classify after the fact.
        track_inputs: List[Tuple[List[float], float, int]] = []
        for d in vehicle_dets:
            crop = self._crop(enhanced, d.bbox)
            wp = self.wheels.classify_one(crop)
            track_inputs.append((d.bbox, d.conf, wp.cls_id))

        tracks = self.tracker.update(enhanced, track_inputs)

        annotated = viz.draw_helmet_detections(enhanced, dets)

        # Aggregate scene-level risk from worst per-track risk.
        worst = None
        for tr in tracks:
            cx = (tr.bbox[0] + tr.bbox[2]) / 2.0
            cy = (tr.bbox[1] + tr.bbox[3]) / 2.0
            self._history[tr.track_id].append((cx, cy))

            wheel_cls = tr.majority_wheel_class()
            wheel_name = WHEEL_CLASS_NAMES.get(wheel_cls, "?")

            # --- Helmet compliance is only checked within 2-wheeler tracks ---
            # (the taxonomy this detector was trained on -- driver/passenger x
            # helmet-status -- only applies to motorcycles in the first place).
            if wheel_cls == 0:
                vehicle_riders = [
                    d for d in rider_dets if self._bbox_center_in(tr.bbox, d.bbox)
                ]
                helmet_violations = self.helmet.count_violations(vehicle_riders)
                rider_count = self.helmet.count_riders(vehicle_riders)
            else:
                helmet_violations = 0
                rider_count = 0

            # --- Phase 3: cross-camera ReID ---
            crop = self._crop(enhanced, tr.bbox)
            global_id = self.reid.match(crop, self.camera_id)

            # Wrong-way heuristic (disabled unless a lane axis is configured).
            wrong_way = self.risk.detect_wrong_way(
                list(self._history[tr.track_id]), allowed_axis="none"
            )

            factors = self.risk.factors_from_context(
                wheel_class=wheel_cls,
                helmet_violation=helmet_violations > 0,
                rider_count=rider_count,
                wrong_way=wrong_way,
            )
            r = self.risk.compute(factors)
            if worst is None or r.score > worst.score:
                worst = r

            annotated = viz.draw_track(
                annotated, tr.bbox, tr.track_id, wheel_name,
                global_id=global_id, risk_level=r.level,
            )

        # --- Phase 3: demographics for exposed-face riders ---
        if self.demographics.enabled:
            for d in dets:
                if d.face_exposed:
                    face = self.demographics.crop_face_region(enhanced, d.bbox)
                    demo = self.demographics.estimate(face)
                    annotated = viz.draw_demographics(annotated, d.bbox, demo)

        # --- overlays ---
        latency_ms = self.meter.stop()
        sysm = system_stats()
        if worst is not None:
            annotated = viz.draw_risk_banner(
                annotated, worst.level, worst.score, worst.reasons
            )
        annotated = viz.draw_telemetry(
            annotated, scene_res.label, self.meter.fps,
            sysm["ram_mb"], sysm["cpu_percent"], frame_idx,
        )
        return annotated

    # ------------------------------------------------------------------ #
    def run(self, source: str, save: bool | None = None,
            show: bool | None = None) -> None:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        save = self.cfg["runtime"]["save_annotated"] if save is None else save
        show = self.cfg["runtime"]["show_window"] if show is None else show

        writer = None
        if save:
            out_dir = resolve_path(self.cfg["paths"]["output_dir"])
            os.makedirs(out_dir, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg["runtime"]["target_fps"]
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = os.path.join(out_dir, "annotated.mp4")
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            print(f"[Pipeline] writing annotated video -> {out_path}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = self.process_frame(frame, frame_idx)
                if writer is not None:
                    writer.write(annotated)
                if show:
                    cv2.imshow("Traffic Surveillance", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"[Pipeline] frame {frame_idx}  "
                          f"fps={self.meter.fps:.1f}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        print(f"[Pipeline] done. processed {frame_idx} frames.")


# ---------------------------------------------------------------------- #
def _run_optimization_demo(cfg: dict) -> None:
    """Phase 4 demo: prune+quantize+export the wheel CNN and report sizes."""
    from src.optimization.model_quantizer import ModelOptimizer
    from src.models.benchmark_classifier import SmallCNN

    net = SmallCNN()
    opt = ModelOptimizer(cfg)
    before = ModelOptimizer.model_size_mb(net)
    net_opt, onnx_path = opt.optimize(net, export=True)
    after = ModelOptimizer.model_size_mb(net_opt)
    print(f"[Optimize] size before={before:.3f} MB after={after:.3f} MB "
          f"onnx={onnx_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Traffic Surveillance pipeline")
    ap.add_argument("--source", default="0",
                    help="video path or webcam index (default 0)")
    ap.add_argument("--camera", default="cam_A", help="camera id for ReID")
    ap.add_argument("--config", default=None, help="path to settings.yaml")
    ap.add_argument("--optimize", action="store_true",
                    help="run the Phase 4 optimization demo and exit")
    ap.add_argument("--no-save", action="store_true", help="do not write output")
    ap.add_argument("--show", action="store_true", help="show a live window")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.optimize:
        _run_optimization_demo(cfg)
        return

    pipe = SurveillancePipeline(cfg, camera_id=args.camera)
    pipe.run(args.source, save=not args.no_save, show=args.show)


if __name__ == "__main__":
    main()
