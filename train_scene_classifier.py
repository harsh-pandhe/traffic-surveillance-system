"""
train_scene_classifier.py
-------------------------
Phase 1 (learned classifier): train + evaluate the DAY/NIGHT/FOG/RAIN model.

Reads condition-labeled image folders, extracts `scene_features` vectors, trains
a RandomForest with a stratified train/val split, prints a confusion matrix +
per-class Precision/Recall/F1, and saves the model to weights/.

Usage:
    python train_scene_classifier.py                 # uses --sources defaults
    python train_scene_classifier.py \
        --source FOG   data/raw/DAWN/Fog \
        --source RAIN  data/raw/DAWN/Rain \
        --source NIGHT data/raw/ExDark \
        --source DAY   data/raw/COCO/day \
        --limit 300

The model bundle stores {model, labels, feature_names} so inference knows the
class order.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from utils.config import load_config, resolve_path
from src.preprocessing.scene_features import extract_features, FEATURE_NAMES
from src.preprocessing.scene_classifier import SCENE_LABELS

_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _list_images(folder: str, limit: int | None) -> List[str]:
    files: List[str] = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    files = sorted(files)
    return files[:limit] if limit else files


def build_dataset(sources: List[Tuple[str, str]],
                  limit: int | None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """sources: list of (LABEL, folder). Returns (X, y, label_names)."""
    label_names = list(SCENE_LABELS)
    label_to_idx = {l: i for i, l in enumerate(label_names)}

    X, y = [], []
    for label, folder in sources:
        label = label.upper()
        if label not in label_to_idx:
            raise ValueError(f"unknown label {label}")
        folder = resolve_path(folder)
        imgs = _list_images(folder, limit)
        kept = 0
        for p in imgs:
            im = cv2.imread(p)
            if im is None:
                continue
            X.append(extract_features(im))
            y.append(label_to_idx[label])
            kept += 1
        print(f"  [{label:5s}] {kept:4d} images from {folder}")
    return np.asarray(X, np.float32), np.asarray(y, np.int64), label_names


def _confusion(y_true, y_pred, labels) -> str:
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    lines = ["Confusion matrix (rows=truth, cols=pred):",
             "        " + "".join(f"{l:>8s}" for l in labels)]
    for i, l in enumerate(labels):
        lines.append(f"{l:>8s}" + "".join(f"{cm[i][j]:>8d}" for j in range(n)))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs=2, action="append",
                    metavar=("LABEL", "FOLDER"),
                    help="repeatable: --source FOG data/raw/DAWN/Fog")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images per source (balances classes)")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="model output path")
    args = ap.parse_args()

    cfg = load_config()
    out_path = resolve_path(
        args.out or cfg.get("scene_classifier_ml", {}).get(
            "model_path", "weights/scene_classifier.joblib")
    )

    # Default sources: whatever Phase-1 datasets are present on disk.
    if args.source:
        sources = [(lbl, folder) for lbl, folder in args.source]
    else:
        candidates = [
            ("FOG", "data/raw/DAWN/Fog"),
            ("RAIN", "data/raw/DAWN/Rain"),
            ("NIGHT", "data/raw/ExDark"),
            ("DAY", "data/raw/COCO/val2017"),
        ]
        sources = [(l, f) for l, f in candidates
                   if os.path.isdir(resolve_path(f))]
        if not sources:
            print("No dataset folders found. Pass --source LABEL FOLDER, or see DATASETS.md.")
            return

    print("Building feature dataset...")
    X, y, labels = build_dataset(sources, args.limit)
    print(f"Total: {len(X)} samples across {len(set(y))} classes "
          f"-> {Counter(labels[i] for i in y)}")
    if len(set(y)) < 2:
        print("Need >=2 classes to train. Download more Phase-1 datasets.")
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=args.seed,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    present = sorted(set(y))
    present_names = [labels[i] for i in present]
    print("\n=== Validation results ===")
    print(f"Accuracy: {accuracy_score(yte, pred):.3f}  (n_val={len(yte)})")
    print(_confusion(yte, pred, [labels[i] for i in range(len(labels))]))
    print("\n" + classification_report(
        yte, pred, labels=present, target_names=present_names, digits=3,
        zero_division=0))

    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                 key=lambda kv: kv[1], reverse=True)
    print("Top features:", ", ".join(f"{n}={v:.3f}" for n, v in imp[:6]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import joblib
    joblib.dump({"model": clf, "labels": labels,
                 "feature_names": FEATURE_NAMES}, out_path)
    print(f"\nSaved model -> {out_path}")


if __name__ == "__main__":
    main()
