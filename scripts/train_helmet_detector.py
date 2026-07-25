"""
scripts/train_helmet_detector.py
---------------------------------
Phase 2 - granular helmet-compliance detector (YOLOv8).

Fine-tunes YOLOv8n on a YOLO-format helmet dataset and reports mAP@50, mAP@50-95,
Precision, and Recall on the validation split. The project's target taxonomy is
the 7-class granular scheme (No Helmet, Full-Face, Half-Face, Strap Unfastened,
Helmet on Handlebar, Helmet on Arm, Rider); the trainer adapts to whatever
classes the supplied data.yaml defines and records the mapping.

Usage:
    python scripts/train_helmet_detector.py --data data/helmet/data.yaml \
        --epochs 30 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "outputs/phase2"
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/helmet/data.yaml",
                    help="YOLO dataset yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        print(f"Dataset yaml not found: {args.data}")
        print("Place a YOLO-format helmet dataset and point --data at its yaml.")
        return

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(data=os.path.abspath(args.data), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=OUT, name="helmet_yolo", exist_ok=True, verbose=False)

    metrics = model.val(data=os.path.abspath(args.data), device=args.device,
                        project=OUT, name="helmet_val", exist_ok=True)
    box = metrics.box
    result = {
        "model": "YOLOv8n",
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_ap50": {metrics.names[i]: float(ap)
                           for i, ap in enumerate(box.ap50)}
        if hasattr(box, "ap50") else {},
        "classes": list(metrics.names.values()),
    }
    print("\n=== Helmet detector (val) ===")
    print(f"mAP@50={result['mAP50']:.3f}  mAP@50-95={result['mAP50_95']:.3f}  "
          f"P={result['precision']:.3f}  R={result['recall']:.3f}")

    # Copy best weights to the project weights/ dir.
    best = os.path.join(OUT, "helmet_yolo", "weights", "best.pt")
    if os.path.isfile(best):
        import shutil
        os.makedirs("weights", exist_ok=True)
        shutil.copy(best, "weights/helmet_yolov8.pt")
        print("saved -> weights/helmet_yolov8.pt")

    with open(f"{OUT}/helmet_metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"metrics -> {OUT}/helmet_metrics.json")


if __name__ == "__main__":
    main()
