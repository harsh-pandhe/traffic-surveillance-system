"""
src/preprocessing/dataset_builder.py
-------------------------------------
Phase 1 (deliverable): Unified dataset builder + preprocessing script.

Combines several heterogeneous benchmark datasets into ONE dataset with a single
common label taxonomy (the project's helmet or wheel class map), then:

    1. remaps each source's class ids/names into the unified taxonomy,
    2. copies/normalises images + labels into a YOLO-style layout,
    3. creates a reproducible train/val split,
    4. optionally runs scene-adaptive enhancement (Phase 1) on every image,
    5. writes a `data.yaml` ready for Ultralytics training in Phase 2.

Supported source formats:
    * "yolo"   : images_dir + labels_dir with `<stem>.txt` (class cx cy w h)
    * "coco"   : images_dir + a COCO-style instances json (labels_dir = json path)
    * "folder" : classification layout, one sub-folder per class name

Everything is config-driven via `config/settings.yaml -> dataset_builder`.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2

from utils.config import load_config, resolve_path
from src.preprocessing.scene_classifier import SceneClassifier
from src.preprocessing.enhancements import FrameEnhancer

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class BuildStats:
    total_images: int = 0
    total_labels: int = 0
    per_source: Dict[str, int] = None
    train: int = 0
    val: int = 0
    dropped: int = 0

    def __post_init__(self):
        if self.per_source is None:
            self.per_source = {}


class DatasetBuilder:
    """Merge multiple source datasets into one unified YOLO-format dataset."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        d = self.cfg["dataset_builder"]
        self.task = d["task"]
        self.out_dir = resolve_path(d["output_dir"])
        self.val_split = float(d["val_split"])
        self.seed = int(d["seed"])
        self.apply_enh = bool(d["apply_enhancement"])
        self.sources = d["sources"]

        # Target taxonomy comes from the relevant model's class_names.
        if self.task == "helmet":
            self.class_names = {int(k): v for k, v in
                                self.cfg["helmet_detector"]["class_names"].items()}
        else:
            self.class_names = {int(k): v for k, v in
                                self.cfg["wheel_classifier"]["class_names"].items()}

        self._enh: Optional[FrameEnhancer] = None
        self._scene: Optional[SceneClassifier] = None
        if self.apply_enh:
            self._enh = FrameEnhancer(self.cfg)
            self._scene = SceneClassifier(self.cfg)

        random.seed(self.seed)

    # ------------------------------------------------------------------ #
    # Layout helpers                                                     #
    # ------------------------------------------------------------------ #
    def _make_dirs(self) -> None:
        for split in ("train", "val"):
            os.makedirs(os.path.join(self.out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "labels", split), exist_ok=True)

    @staticmethod
    def _list_images(folder: str) -> List[str]:
        if not os.path.isdir(folder):
            return []
        return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith(_IMG_EXTS)]

    # ------------------------------------------------------------------ #
    # Per-format readers -> normalised list of (image_path, yolo_lines)  #
    # yolo_lines already remapped to the unified taxonomy.               #
    # ------------------------------------------------------------------ #
    def _read_yolo(self, src: dict) -> List[Tuple[str, List[str]]]:
        cmap = {int(k): int(v) for k, v in src["class_map"].items()}
        images = self._list_images(resolve_path(src["images_dir"]))
        labels_dir = resolve_path(src["labels_dir"])
        out = []
        for img in images:
            stem = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(labels_dir, stem + ".txt")
            lines: List[str] = []
            if os.path.isfile(lbl):
                with open(lbl, "r", encoding="utf-8") as fh:
                    for row in fh:
                        parts = row.split()
                        if len(parts) < 5:
                            continue
                        src_cls = int(float(parts[0]))
                        if src_cls not in cmap:      # class not in taxonomy -> drop box
                            continue
                        parts[0] = str(cmap[src_cls])
                        lines.append(" ".join(parts))
            out.append((img, lines))
        return out

    def _read_coco(self, src: dict) -> List[Tuple[str, List[str]]]:
        cmap = {int(k): int(v) for k, v in src["class_map"].items()}
        images_dir = resolve_path(src["images_dir"])
        with open(resolve_path(src["labels_dir"]), "r", encoding="utf-8") as fh:
            coco = json.load(fh)
        img_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_img: Dict[int, List[dict]] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        out = []
        for img_id, im in img_by_id.items():
            path = os.path.join(images_dir, im["file_name"])
            if not os.path.isfile(path):
                continue
            w, h = im["width"], im["height"]
            lines: List[str] = []
            for a in anns_by_img.get(img_id, []):
                src_cls = int(a["category_id"])
                if src_cls not in cmap:
                    continue
                x, y, bw, bh = a["bbox"]           # COCO xywh (top-left)
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                lines.append(f"{cmap[src_cls]} {cx:.6f} {cy:.6f} "
                             f"{bw / w:.6f} {bh / h:.6f}")
            out.append((path, lines))
        return out

    def _read_folder(self, src: dict) -> List[Tuple[str, List[str]]]:
        """
        Classification layout -> one full-image box per class (cx,cy,w,h = centre,
        full frame). Lets a detection trainer consume classification data.
        """
        cmap = {str(k): int(v) for k, v in src["class_map"].items()}
        root = resolve_path(src["images_dir"])
        out = []
        for cls_name, uni_id in cmap.items():
            for img in self._list_images(os.path.join(root, cls_name)):
                out.append((img, [f"{uni_id} 0.5 0.5 1.0 1.0"]))
        return out

    def _read_source(self, src: dict) -> List[Tuple[str, List[str]]]:
        fmt = src["format"]
        if fmt == "yolo":
            return self._read_yolo(src)
        if fmt == "coco":
            return self._read_coco(src)
        if fmt == "folder":
            return self._read_folder(src)
        raise ValueError(f"unknown source format: {fmt}")

    # ------------------------------------------------------------------ #
    # Image writer (with optional adaptive enhancement)                  #
    # ------------------------------------------------------------------ #
    def _write_image(self, src_path: str, dst_path: str) -> bool:
        img = cv2.imread(src_path)
        if img is None:
            return False
        if self.apply_enh and self._scene is not None and self._enh is not None:
            label = self._scene.classify(img).label
            img = self._enh.enhance(img, label)
        cv2.imwrite(dst_path, img)
        return True

    # ------------------------------------------------------------------ #
    def build(self) -> BuildStats:
        """Run the full unify+split+preprocess pipeline. Returns BuildStats."""
        self._make_dirs()
        stats = BuildStats()

        # 1) gather remapped samples from all sources
        samples: List[Tuple[str, List[str], str]] = []   # (img, lines, src_name)
        for src in self.sources:
            got = self._read_source(src)
            stats.per_source[src["name"]] = len(got)
            for img, lines in got:
                samples.append((img, lines, src["name"]))

        # 2) deterministic shuffle + split
        random.shuffle(samples)
        n_val = int(len(samples) * self.val_split)
        split_of = {i: ("val" if i < n_val else "train")
                    for i in range(len(samples))}

        # 3) copy images (+enhance) and write labels
        for i, (img, lines, src_name) in enumerate(samples):
            split = split_of[i]
            stem = f"{src_name}_{i:06d}"
            dst_img = os.path.join(self.out_dir, "images", split, stem + ".jpg")
            dst_lbl = os.path.join(self.out_dir, "labels", split, stem + ".txt")
            if not self._write_image(img, dst_img):
                stats.dropped += 1
                continue
            with open(dst_lbl, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            stats.total_images += 1
            stats.total_labels += len(lines)
            stats.train += int(split == "train")
            stats.val += int(split == "val")

        # 4) emit Ultralytics data.yaml
        self._write_data_yaml()
        return stats

    def _write_data_yaml(self) -> str:
        names = [self.class_names[i] for i in sorted(self.class_names)]
        content = (
            f"# Auto-generated by dataset_builder.py (task={self.task})\n"
            f"path: {os.path.abspath(self.out_dir)}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(names)}\n"
            f"names: {names}\n"
        )
        path = os.path.join(self.out_dir, "data.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path


if __name__ == "__main__":
    builder = DatasetBuilder()
    stats = builder.build()
    print("=== Unified dataset build complete ===")
    print(f"task           : {builder.task}")
    print(f"output         : {builder.out_dir}")
    print(f"per source     : {stats.per_source}")
    print(f"images (train/val): {stats.train}/{stats.val}")
    print(f"labels total   : {stats.total_labels}")
    print(f"dropped        : {stats.dropped}")
