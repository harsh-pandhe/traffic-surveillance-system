"""
scripts/build_wheel_dataset.py
------------------------------
Phase 2 - wheel-count classification dataset builder.

Crops vehicle instances from COCO val2017 using the instances annotations and
maps COCO categories to the project's wheel-count taxonomy:

    2-Wheeler (0)  <- bicycle, motorcycle
    3-Wheeler (1)  <- (not in COCO; supplied separately, see --extra3)
    4-Wheeler (2)  <- car
    6+ Wheeler (3) <- bus, truck

Optionally merges a supplementary 3-wheeler (auto-rickshaw) folder so all four
classes are represented. Writes an ImageFolder-style layout:

    data/wheels/train/<class>/*.jpg
    data/wheels/val/<class>/*.jpg

which both the SmallCNN trainer and YOLOv8-cls consume directly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Target taxonomy (index -> folder name).
CLASS_DIR = {0: "2wheeler", 1: "3wheeler", 2: "4wheeler", 3: "6plus_wheeler"}

# COCO category name -> our wheel class id.
COCO_MAP = {
    "bicycle": 0, "motorcycle": 0,
    "car": 2,
    "bus": 3, "truck": 3,
}

MIN_BOX = 48          # ignore tiny crops (px, min side)
MAX_PER_CLASS = 700   # cap to keep classes balanced-ish


def crop_from_coco(images_dir, ann_json, out_root, val_split, seed):
    with open(ann_json) as fh:
        coco = json.load(fh)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im for im in coco["images"]}

    # Gather (image_path, bbox, wheel_cls) for wanted categories.
    samples_by_cls = defaultdict(list)
    for a in coco["annotations"]:
        name = cat_id_to_name.get(a["category_id"])
        if name not in COCO_MAP:
            continue
        if a.get("iscrowd", 0) == 1:
            continue
        cls = COCO_MAP[name]
        x, y, w, h = a["bbox"]
        if min(w, h) < MIN_BOX:
            continue
        im = img_by_id.get(a["image_id"])
        if im is None:
            continue
        path = os.path.join(images_dir, im["file_name"])
        samples_by_cls[cls].append((path, (x, y, w, h)))

    random.seed(seed)
    counts = defaultdict(lambda: {"train": 0, "val": 0})
    for cls, samples in samples_by_cls.items():
        # Split by SOURCE IMAGE, not by crop.
        #
        # A single COCO photo often contains several vehicles of the same class
        # (measured: 76% / 91% / 65% of 2-, 4- and 6+-wheeler crops come from
        # such photos). Shuffling crops and slicing by index therefore put
        # different vehicles from the *same* scene on both sides of the split --
        # sharing background, lighting, camera and resolution -- which leaks and
        # inflates validation accuracy. Grouping by image keeps every crop of a
        # photo on one side.
        by_image = defaultdict(list)
        for path, box in samples:
            by_image[path].append(box)

        image_paths = sorted(by_image)          # sort first for determinism
        random.shuffle(image_paths)
        n_val_imgs = int(len(image_paths) * val_split)
        split_of = {p: ("val" if i < n_val_imgs else "train")
                    for i, p in enumerate(image_paths)}

        emitted = 0
        for i, path in enumerate(image_paths):
            if emitted >= MAX_PER_CLASS:
                break
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            split = split_of[path]
            stem = os.path.splitext(os.path.basename(path))[0]
            for j, (x, y, w, h) in enumerate(by_image[path]):
                if emitted >= MAX_PER_CLASS:
                    break
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(W, int(x + w)), min(H, int(y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                d = os.path.join(out_root, split, CLASS_DIR[cls])
                os.makedirs(d, exist_ok=True)
                # Filename carries the source image stem so leakage is auditable.
                cv2.imwrite(os.path.join(d, f"coco_{cls}_{stem}_{j:02d}.jpg"), crop)
                counts[cls][split] += 1
                emitted += 1
    return counts


def merge_extra_3wheeler(extra_dir, out_root, val_split, seed):
    """
    Crop auto-rickshaws into the 3-wheeler class.

    This must emit *crops*, not whole photos. The COCO arm produces tight
    vehicle crops, so copying full scenes here would give the classifier a
    trivial shortcut -- "full photo => 3-wheeler" -- rather than making it learn
    vehicle appearance. (That shortcut is the likely explanation for the
    implausibly high 3-wheeler F1 of 0.96 in the first benchmark run.)

    The source is a YOLO-format dataset, so bounding boxes come from the
    matching .txt label file. Splitting is by image, consistent with the COCO arm.
    """
    if not extra_dir or not os.path.isdir(extra_dir):
        return {"train": 0, "val": 0}

    # Pair each image with its YOLO label file.
    pairs = []
    for dp, _dn, fn in os.walk(extra_dir):
        for f in fn:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(dp, f)
            lbl_path = os.path.join(
                dp.replace(os.sep + "images", os.sep + "labels"),
                os.path.splitext(f)[0] + ".txt")
            if os.path.isfile(lbl_path):
                pairs.append((img_path, lbl_path))

    random.seed(seed)
    pairs.sort()
    random.shuffle(pairs)
    n_val_imgs = int(len(pairs) * val_split)

    c = {"train": 0, "val": 0}
    emitted = 0
    for i, (img_path, lbl_path) in enumerate(pairs):
        if emitted >= MAX_PER_CLASS:
            break
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        split = "val" if i < n_val_imgs else "train"
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for j, line in enumerate(open(lbl_path).read().splitlines()):
            if emitted >= MAX_PER_CLASS:
                break
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < MIN_BOX or y2 - y1 < MIN_BOX:
                continue
            d = os.path.join(out_root, split, CLASS_DIR[1])
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"auto_{stem}_{j:02d}.jpg"),
                        img[y1:y2, x1:x2])
            c[split] += 1
            emitted += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw/COCO/val2017")
    ap.add_argument("--ann", default="data/raw/COCO/annotations/instances_val2017.json")
    ap.add_argument("--out", default="data/wheels")
    ap.add_argument("--extra3", default="data/raw/auto_rickshaw",
                    help="optional 3-wheeler (auto-rickshaw) image folder")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Cropping vehicles from COCO...")
    counts = crop_from_coco(args.images, args.ann, args.out,
                            args.val_split, args.seed)
    c3 = merge_extra_3wheeler(args.extra3, args.out, args.val_split, args.seed)

    print("\n=== Wheel-count dataset ===")
    for cls in sorted(CLASS_DIR):
        if cls == 1:
            tr, va = c3["train"], c3["val"]
        else:
            tr, va = counts[cls]["train"], counts[cls]["val"]
        src = "auto-rickshaw" if cls == 1 else "COCO"
        print(f"  {CLASS_DIR[cls]:14s} train={tr:4d} val={va:4d}  ({src})")
    print(f"\nOutput -> {args.out}/(train|val)/<class>/")


if __name__ == "__main__":
    main()
