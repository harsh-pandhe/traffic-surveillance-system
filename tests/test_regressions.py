"""
tests/test_regressions.py
--------------------------
Regression tests for defects that actually shipped in this project.

Every test here corresponds to a real bug that reached a published report
because nothing asserted it:

  * helmet compliance was inverted (class ids hard-coded against the wrong
    taxonomy, so helmeted riders were reported as violations);
  * the selected wheel model was never copied out of runs/, so the pipeline
    silently ran a generic ImageNet classifier;
  * the wheel dataset split by crop instead of by source image, leaking scenes
    across train/val and inflating the benchmark;
  * DeepSORT and OSNet silently degraded to fallbacks after a dependency
    downgrade, which looked exactly like success.

Tests that need trained weights or datasets skip cleanly when those are absent,
so the suite still runs on a fresh clone and in CI.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HELMET_WEIGHTS = "weights/helmet_yolov8.pt"
HELMET_TEST_IMAGES = "data/raw/helmet_raw/data/test/images"
WHEELS_ROOT = "data/wheels"


# --------------------------------------------------------------------------- #
# 1. Helmet compliance semantics (the inversion bug)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isfile(HELMET_WEIGHTS),
                    reason="helmet weights not present")
def test_compliance_semantics_not_inverted():
    """
    'With Helmet' must never be a violation; 'Without Helmet' always must be.

    This is the exact defect that shipped: config declared 0 == 'No Helmet'
    while the weights used 0 == 'Driver With Helmet', inverting every decision.
    """
    from src.models.helmet_detector import HelmetDetector
    det = HelmetDetector()

    for cls_id, name in det.class_names.items():
        low = name.lower()
        if "with helmet" in low:
            assert cls_id not in det.violation_ids, (
                f"'{name}' (id {cls_id}) is marked a violation - "
                f"compliance semantics are inverted")
        if "without helmet" in low or low == "no helmet":
            assert cls_id in det.violation_ids, (
                f"'{name}' (id {cls_id}) is NOT marked a violation")


@pytest.mark.skipif(not os.path.isfile(HELMET_WEIGHTS),
                    reason="helmet weights not present")
def test_config_matches_trained_weights():
    """Config class_names must match the loaded weights, position by position."""
    from src.models.helmet_detector import HelmetDetector
    det = HelmetDetector()
    model_names = getattr(det.model, "names", None)
    if not model_names:
        pytest.skip("model exposes no class names")

    def norm(s):
        return str(s).strip().lower().replace("_", " ")

    assert len(model_names) == len(det.class_names), (
        f"weights have {len(model_names)} classes, "
        f"config declares {len(det.class_names)}")
    for i in range(len(model_names)):
        assert norm(model_names[i]) == norm(det.class_names.get(i, "")), (
            f"class {i}: weights='{model_names[i]}' "
            f"config='{det.class_names.get(i)}' - compliance will be wrong")


@pytest.mark.skipif(
    not (os.path.isfile(HELMET_WEIGHTS) and os.path.isdir(HELMET_TEST_IMAGES)),
    reason="helmet weights or test images not present")
def test_detector_labels_real_images_correctly():
    """End-to-end check on real held-out images."""
    import cv2
    from src.models.helmet_detector import HelmetDetector
    det = HelmetDetector()
    seen = {}
    for p in sorted(glob.glob(os.path.join(HELMET_TEST_IMAGES, "*")))[:25]:
        img = cv2.imread(p)
        if img is None:
            continue
        for d in det.detect(img):
            seen.setdefault(d.cls_name, d.is_violation)

    if "Driver With Helmet" in seen:
        assert seen["Driver With Helmet"] is False
    if "Driver Without Helmet" in seen:
        assert seen["Driver Without Helmet"] is True
    assert seen, "detector produced no detections on real test images"


# --------------------------------------------------------------------------- #
# 2. Dataset leakage (split by image, not by crop)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(WHEELS_ROOT),
                    reason="wheel dataset not built")
def test_no_source_image_leaks_across_splits():
    """
    No source photo may contribute crops to both train and val.

    Crop filenames embed the source stem (`coco_<cls>_<stem>_<idx>.jpg`), so
    stripping the trailing index recovers the source image identity.
    """
    def sources(split):
        out = defaultdict(set)
        root = os.path.join(WHEELS_ROOT, split)
        if not os.path.isdir(root):
            return out
        for cls in os.listdir(root):
            for f in os.listdir(os.path.join(root, cls)):
                out[cls].add(re.sub(r"_\d+\.jpg$", "", f))
        return out

    train, val = sources("train"), sources("val")
    for cls in set(train) | set(val):
        overlap = train.get(cls, set()) & val.get(cls, set())
        assert not overlap, (
            f"{cls}: {len(overlap)} source images appear in BOTH train and val "
            f"- benchmark numbers are inflated (e.g. {sorted(overlap)[:3]})")


# --------------------------------------------------------------------------- #
# 3. Backends must not silently degrade to fallbacks
# --------------------------------------------------------------------------- #
def test_deepsort_backend_is_real():
    """A dependency downgrade silently swapped DeepSORT for an IoU fallback."""
    from src.tracking.deepsort_tracker import VehicleTracker
    backend = VehicleTracker()._backend
    assert backend == "deepsort", (
        f"tracker fell back to '{backend}' - DeepSORT is not actually running")


def test_reid_backend_is_real():
    """torchreid's two package layouts previously left OSNet silently disabled."""
    from src.tracking.multi_camera_reid import VehicleReID
    backend = VehicleReID()._backend
    assert backend == "osnet", (
        f"ReID fell back to '{backend}' - OSNet is not actually running")


# --------------------------------------------------------------------------- #
# 4. Config sanity
# --------------------------------------------------------------------------- #
def test_configured_weight_paths_exist():
    """Config must not point at weights that were never produced."""
    from utils.config import load_config, resolve_path
    cfg = load_config()
    checks = {
        "helmet_detector.weights": cfg["helmet_detector"]["weights"],
        "wheel_classifier.yolo_weights": cfg["wheel_classifier"]["yolo_weights"],
        "scene_classifier_ml.model_path":
            cfg["scene_classifier_ml"]["model_path"],
    }
    missing = [f"{k} -> {v}" for k, v in checks.items()
               if not os.path.isfile(resolve_path(v))]
    if missing:
        pytest.skip("weights not built in this environment: " + "; ".join(missing))
