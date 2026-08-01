"""
scripts/gate_a_learning_curve.py
---------------------------------
Gate A: is the helmet detector still data-limited at 368 training images?

The HELMET dataset would raise the training set from 368 images to ~91,000
frames, but its images are ~29 GB behind a throttled OSF endpoint. Before
spending hours on that download, measure whether more data would actually help.

Method: retrain the detector on 25%, 50%, 75% and 100% of the training split
(validation and test held fixed) and evaluate each on the untouched test split.
If test mAP is still climbing steeply at 100%, the model is data-starved and the
download is justified. If it has flattened, the bottleneck is elsewhere (model
capacity, label quality, class imbalance) and more images will not pay.

Usage:
    python scripts/gate_a_learning_curve.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "data/raw/helmet_raw/data"
OUT_JSON = "results/phase0_gates.json"
CLASSES = ['driver_with_helmet', 'bike', 'driver', 'passenger_with_helmet',
           'passenger', 'driver_without_helmet', 'passenger_without_helmet']


def build_subset(fraction: float, workdir: str, seed: int = 42) -> str:
    """Materialise a YOLO dataset using `fraction` of the training images."""
    imgs = sorted(os.listdir(f"{SRC}/train/images"))
    random.Random(seed).shuffle(imgs)
    keep = imgs[:max(1, int(len(imgs) * fraction))]

    root = os.path.join(workdir, f"frac_{int(fraction*100)}")
    for split in ("train", "vaid", "test"):
        os.makedirs(f"{root}/{split}/images", exist_ok=True)
        os.makedirs(f"{root}/{split}/labels", exist_ok=True)

    # Train: only the sampled subset.
    for f in keep:
        stem = os.path.splitext(f)[0]
        shutil.copy(f"{SRC}/train/images/{f}", f"{root}/train/images/{f}")
        lbl = f"{SRC}/train/labels/{stem}.txt"
        if os.path.isfile(lbl):
            shutil.copy(lbl, f"{root}/train/labels/{stem}.txt")

    # Val and test: always the full, fixed splits.
    for split in ("vaid", "test"):
        for f in os.listdir(f"{SRC}/{split}/images"):
            stem = os.path.splitext(f)[0]
            shutil.copy(f"{SRC}/{split}/images/{f}", f"{root}/{split}/images/{f}")
            lbl = f"{SRC}/{split}/labels/{stem}.txt"
            if os.path.isfile(lbl):
                shutil.copy(lbl, f"{root}/{split}/labels/{stem}.txt")

    with open(f"{root}/data.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(root)}\ntrain: train/images\n"
                 f"val: vaid/images\ntest: test/images\n"
                 f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    return root, len(keep)


def run_point(fraction: float, epochs: int, imgsz: int, batch: int,
              workdir: str) -> dict:
    from ultralytics import YOLO
    root, n_train = build_subset(fraction, workdir)
    model = YOLO("yolov8n.pt")
    model.train(data=f"{root}/data.yaml", epochs=epochs, imgsz=imgsz,
                batch=batch, device="cpu", verbose=False,
                project=workdir, name=f"run_{int(fraction*100)}",
                exist_ok=True, plots=False)
    res = model.val(data=f"{root}/data.yaml", split="test", device="cpu",
                    project=workdir, name=f"val_{int(fraction*100)}",
                    exist_ok=True, verbose=False)
    return {"fraction": fraction, "train_images": n_train,
            "test_mAP50": float(res.box.map50),
            "test_mAP50_95": float(res.box.map)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="gate_a_")
    points = []
    for f in args.fractions:
        print(f"\n=== training on {int(f*100)}% of the train split ===")
        p = run_point(f, args.epochs, args.imgsz, args.batch, workdir)
        print(f"  {p['train_images']} imgs -> test mAP@50 {p['test_mAP50']:.3f}")
        points.append(p)

    # Slope over the final segment decides whether data is still the bottleneck.
    last_gain = points[-1]["test_mAP50"] - points[-2]["test_mAP50"]
    total_gain = points[-1]["test_mAP50"] - points[0]["test_mAP50"]
    verdict = ("DATA-LIMITED - more images should help"
               if last_gain > 0.02 else
               "PLATEAUED - more images unlikely to help materially")

    print("\n=== Gate A: helmet learning curve ===")
    for p in points:
        print(f"  {int(p['fraction']*100):3d}%  {p['train_images']:4d} imgs  "
              f"test mAP@50 {p['test_mAP50']:.3f}")
    print(f"\ngain over final quarter: {last_gain:+.3f}  "
          f"(total across curve: {total_gain:+.3f})")
    print(f"VERDICT: {verdict}")

    os.makedirs("results", exist_ok=True)
    gates = {}
    if os.path.isfile(OUT_JSON):
        gates = json.load(open(OUT_JSON))
    gates["gate_A_helmet_data"] = {
        "status": "COMPLETE",
        "question": "Is the helmet detector still data-limited at 368 images?",
        "points": points, "final_quarter_gain": last_gain,
        "total_gain": total_gain, "verdict": verdict,
        "decision": ("download HELMET images (~29GB)" if last_gain > 0.02
                     else "skip HELMET download; data is not the bottleneck"),
    }
    json.dump(gates, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved -> {OUT_JSON}")
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
