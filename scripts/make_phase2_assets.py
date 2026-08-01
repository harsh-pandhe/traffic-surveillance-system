"""
scripts/make_phase2_assets.py
------------------------------
Generate figures for the Phase 2 report PDF from the persisted metrics
(results/phase2/*.json) plus one real helmet-detection sample image.

Outputs -> outputs/report_p2/*.png
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/report_p2"
os.makedirs(OUT, exist_ok=True)

WHEEL = json.load(open("results/phase2/wheel_metrics.json"))

def _normalise_helmet(h):
    """Accept both the old flat schema and the new {val,test} schema."""
    if "test" in h and isinstance(h["test"], dict):
        out = dict(h)
        out.update(h["test"])          # headline = held-out test numbers
        out["_split"] = "test"
        return out
    h = dict(h); h["_split"] = "val"
    return h

HELMET = json.load(open("results/phase2/helmet_metrics.json"))
HELMET = _normalise_helmet(HELMET)

NAVY, BLUE, RED, GREY = "#1b2a4a", "#2e86c1", "#c0392b", "#7fb3d5"


ARMS = [k for k in ("cnn", "cspnext", "yolo") if WHEEL.get(k)]


def fig_wheel_compare():
    models = [WHEEL[k]["model"] for k in ARMS]
    accs = [WHEEL[k]["accuracy"] for k in ARMS]
    f1s = [WHEEL[k]["f1"] for k in ARMS]
    x = np.arange(len(ARMS)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=NAVY)
    b2 = ax.bar(x + w/2, f1s, w, label="Macro-F1", color=BLUE)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Wheel-Count: CNN vs RTMDet(CSPNeXt) vs YOLO")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/wheel_compare.png", dpi=150); plt.close(fig)


def fig_wheel_perclass():
    classes = WHEEL["cnn"]["classes"]
    x = np.arange(len(classes)); w = 0.8 / len(ARMS)
    palette = {"cnn": GREY, "cspnext": NAVY, "yolo": BLUE}
    fig, ax = plt.subplots(figsize=(7.4, 4))
    for i, k in enumerate(ARMS):
        vals = [WHEEL[k]["report"][c]["f1-score"] for c in classes]
        off = (i - (len(ARMS) - 1) / 2) * w
        ax.bar(x + off, vals, w, label=WHEEL[k]["model"], color=palette[k])
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1 (wheel-count)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/wheel_perclass.png", dpi=150); plt.close(fig)


def fig_helmet_ap():
    ap = HELMET["per_class_ap50"]
    items = sorted(ap.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    colors = [RED if v < 0.6 else BLUE for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(names, vals, color=colors)
    ax.axvline(HELMET["mAP50"], color=NAVY, ls="--", lw=1.5,
               label=f"mAP@50 = {HELMET['mAP50']:.3f}")
    for i, v in enumerate(vals):
        ax.text(v+0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0); ax.set_xlabel("AP@50")
    ax.set_title("Helmet Detector — Per-Class AP@50"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/helmet_ap.png", dpi=150); plt.close(fig)


def fig_helmet_sample():
    """Run the trained helmet model on a real test image; save annotated."""
    w = "weights/helmet_yolov8.pt"
    imgs = glob.glob("data/raw/helmet_raw/data/test/images/*")
    if not os.path.isfile(w) or not imgs:
        return
    try:
        from ultralytics import YOLO
        model = YOLO(w)
        # pick an image with several detections
        best_img, best_res, best_n = imgs[0], None, -1
        for p in imgs[:25]:
            r = model.predict(p, imgsz=416, conf=0.3, device="cpu", verbose=False)[0]
            n = len(r.boxes) if r.boxes is not None else 0
            if n > best_n:
                best_n, best_img, best_res = n, p, r
        annotated = best_res.plot()  # BGR
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title(f"Helmet detector on a test image ({best_n} detections)")
        fig.tight_layout(); fig.savefig(f"{OUT}/helmet_sample.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("helmet sample failed:", e)


def fig_composition():
    labels = ["2wheeler", "3wheeler", "4wheeler", "6plus_wheeler"]
    counts = [len(glob.glob(f"data/wheels/train/{l}/*")) for l in labels]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(labels, counts, color=BLUE)
    for i, v in enumerate(counts):
        ax.text(i, v+3, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Train images"); ax.set_title("Wheel Dataset Composition (4 classes)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/wheel_composition.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_wheel_compare()
    fig_wheel_perclass()
    fig_helmet_ap()
    fig_composition()
    fig_helmet_sample()
    print(f"assets -> {OUT}/")
