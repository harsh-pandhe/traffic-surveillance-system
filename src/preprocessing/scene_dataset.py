"""
src/preprocessing/scene_dataset.py
-----------------------------------
Phase 1 (deliverable): condition-labeled scene dataset builder + evaluator.

Combines the weather/lighting benchmark datasets used in Phase 1 into ONE
condition-labeled corpus (DAY / NIGHT / FOG / RAIN), then:

    1. Ingests each source folder and assigns its ground-truth condition
       (e.g. DAWN/Fog -> FOG, ExDark -> NIGHT, COCO/BDD day -> DAY).
    2. Runs the rule-based SceneClassifier on every image.
    3. Optionally writes the scene-adaptive *enhanced* image into
       data/processed/<CONDITION>/ (the "enhance before further processing" step).
    4. Reports env-detection accuracy: confusion matrix + per-class P/R/F1.

This produces the measurable metric for the Phase 1 milestone demo, and the
enhanced corpus that Phase 2 trains on.

Relevant Phase-1 datasets (see DATASETS.md for links/placement):
    DAWN   -> Fog/Rain folders   -> FOG / RAIN
    ExDark -> low-light images   -> NIGHT
    COCO / BDD100K daytime       -> DAY
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2

from utils.config import load_config, resolve_path
from src.preprocessing.scene_classifier import (
    SceneClassifier, SCENE_LABELS, DAY, NIGHT, FOG, RAIN,
)
from src.preprocessing.enhancements import FrameEnhancer

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class SceneEvalReport:
    """Confusion matrix + derived metrics for env-detection accuracy."""
    labels: Tuple[str, ...] = SCENE_LABELS
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total: int = 0

    def __post_init__(self):
        if not self.confusion:
            self.confusion = {gt: {pr: 0 for pr in self.labels}
                              for gt in self.labels}

    def add(self, gt: str, pred: str) -> None:
        self.confusion[gt][pred] += 1
        self.total += 1

    def accuracy(self) -> float:
        correct = sum(self.confusion[l][l] for l in self.labels)
        return correct / self.total if self.total else 0.0

    def per_class(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for l in self.labels:
            tp = self.confusion[l][l]
            fp = sum(self.confusion[g][l] for g in self.labels if g != l)
            fn = sum(self.confusion[l][p] for p in self.labels if p != l)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            out[l] = {"precision": prec, "recall": rec, "f1": f1,
                      "support": tp + fn}
        return out

    def render(self) -> str:
        lines = ["Confusion matrix (rows=truth, cols=pred):",
                 "        " + "".join(f"{l:>8s}" for l in self.labels)]
        for gt in self.labels:
            row = "".join(f"{self.confusion[gt][pr]:>8d}" for pr in self.labels)
            lines.append(f"{gt:>8s}{row}")
        lines.append(f"\nOverall accuracy: {self.accuracy():.3f} "
                     f"({self.total} images)")
        lines.append("\nPer-class:")
        lines.append(f"{'label':>8s} {'prec':>6s} {'rec':>6s} "
                     f"{'f1':>6s} {'n':>6s}")
        for l, m in self.per_class().items():
            lines.append(f"{l:>8s} {m['precision']:6.3f} {m['recall']:6.3f} "
                         f"{m['f1']:6.3f} {int(m['support']):6d}")
        return "\n".join(lines)


class SceneDatasetBuilder:
    """Unify condition-labeled scene sources; enhance + evaluate the classifier."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.processed_dir = resolve_path(self.cfg["paths"]["processed_dir"])
        self.scene = SceneClassifier(self.cfg)
        self.enhancer = FrameEnhancer(self.cfg)

    @staticmethod
    def _list_images(folder: str) -> List[str]:
        if not os.path.isdir(folder):
            return []
        out = []
        for dirpath, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(_IMG_EXTS):
                    out.append(os.path.join(dirpath, f))
        return sorted(out)

    def build_and_evaluate(
        self,
        sources: List[Tuple[str, str]],
        write_enhanced: bool = True,
        limit_per_source: int | None = None,
    ) -> SceneEvalReport:
        """
        Args:
            sources: list of (folder_path, ground_truth_condition).
            write_enhanced: also write enhanced images into data/processed/<cond>/.
            limit_per_source: cap images per source (useful for quick demos).
        Returns:
            SceneEvalReport with confusion matrix + metrics.
        """
        report = SceneEvalReport()
        for folder, gt in sources:
            gt = gt.upper()
            if gt not in SCENE_LABELS:
                raise ValueError(f"unknown ground-truth condition: {gt}")
            imgs = self._list_images(resolve_path(folder))
            if limit_per_source:
                imgs = imgs[:limit_per_source]
            if write_enhanced:
                os.makedirs(os.path.join(self.processed_dir, gt), exist_ok=True)

            for i, path in enumerate(imgs):
                img = cv2.imread(path)
                if img is None:
                    continue
                pred = self.scene.classify(img).label
                report.add(gt, pred)
                if write_enhanced:
                    enh = self.enhancer.enhance(img, pred)
                    out = os.path.join(self.processed_dir, gt,
                                       f"{gt.lower()}_{i:06d}.jpg")
                    cv2.imwrite(out, enh)
        return report


def _default_sources(cfg: dict) -> List[Tuple[str, str]]:
    """Map the Phase-1 datasets on disk to their ground-truth conditions."""
    raw = cfg["paths"]["raw_dir"]
    # These paths are the expected placement documented in DATASETS.md.
    return [
        (f"{raw}/DAWN/Fog", FOG),
        (f"{raw}/DAWN/Rain", RAIN),
        (f"{raw}/ExDark", NIGHT),
        (f"{raw}/COCO/val2017", DAY),    # COCO daytime (or BDD100K daytime subset)
    ]


if __name__ == "__main__":
    cfg = load_config()
    builder = SceneDatasetBuilder(cfg)
    sources = _default_sources(cfg)
    present = [(p, g) for p, g in sources if os.path.isdir(resolve_path(p))]
    if not present:
        print("No Phase-1 datasets found on disk. Expected folders:")
        for p, g in sources:
            print(f"  [{g:5s}] {p}")
        print("\nSee DATASETS.md for download links + placement, then rerun:")
        print("  python -m src.preprocessing.scene_dataset")
    else:
        report = builder.build_and_evaluate(present, write_enhanced=True)
        print(report.render())
