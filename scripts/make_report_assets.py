"""
scripts/make_report_assets.py
------------------------------
Generate every figure + the metrics JSON used by the Phase 1 report PDF.

Reproduces the 4-class scene-classifier training (fixed seed) so the confusion
matrix, per-class metrics, and feature importances in the report are the real,
recomputed numbers -- not hand-copied. Also renders the enhancement montage and
dataset-composition / accuracy-progression charts.

Outputs -> outputs/report/*.png  and  outputs/report/metrics.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

# Allow running from anywhere: put the project root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score)

from src.preprocessing.scene_features import extract_features, FEATURE_NAMES
from src.preprocessing.scene_classifier import SceneClassifier

OUT = "outputs/report"
os.makedirs(OUT, exist_ok=True)

LABELS = ["DAY", "NIGHT", "FOG", "RAIN"]
SOURCES = [
    ("FOG", "data/raw/DAWN/Fog"),
    ("RAIN", "data/raw/DAWN/Rain"),
    ("NIGHT", "data/raw/ExDark"),
    ("DAY", "data/raw/COCO/val2017"),
]
LIMIT = 300
SEED = 42

# Palette (colour-blind friendly-ish).
C = {"DAY": "#f2b134", "NIGHT": "#2c3e6b", "FOG": "#8a9ba8", "RAIN": "#2e86c1",
     "accent": "#c0392b", "bar": "#34699a", "bar2": "#7fb3d5"}


def _imgs(folder, limit):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return sorted(files)[:limit]


# ---------------------------------------------------------------------- #
# 1) Build feature dataset                                               #
# ---------------------------------------------------------------------- #
def build_dataset():
    li = {l: i for i, l in enumerate(LABELS)}
    X, y, counts = [], [], {}
    for label, folder in SOURCES:
        n = 0
        for p in _imgs(folder, LIMIT):
            im = cv2.imread(p)
            if im is None:
                continue
            X.append(extract_features(im))
            y.append(li[label])
            n += 1
        counts[label] = n
        print(f"  [{label:5s}] {n} images")
    return np.asarray(X, np.float32), np.asarray(y, np.int64), counts


# ---------------------------------------------------------------------- #
# Figure helpers                                                         #
# ---------------------------------------------------------------------- #
def fig_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("4-Class Confusion Matrix (validation)")
    thr = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
    plt.close(fig)


def fig_per_class(report):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(LABELS)); w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, m in enumerate(metrics):
        vals = [report[l][m] for l in LABELS]
        ax.bar(x + (k - 1) * w, vals, w, label=m,
               color=[C["bar"], C["bar2"], C["accent"]][k])
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/per_class.png", dpi=150)
    plt.close(fig)


def fig_importance(importances):
    order = np.argsort(importances)
    names = [FEATURE_NAMES[i] for i in order]
    vals = importances[order]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(names, vals, color=C["bar"])
    ax.set_xlabel("Importance"); ax.set_title("RandomForest Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/feature_importance.png", dpi=150)
    plt.close(fig)


def fig_progression():
    stages = ["Rules\n(fog/rain)", "Rules tuned\n(fog/rain)",
              "Learned 2-cls\n(fog/rain)", "Learned 3-cls\n(+night)",
              "Learned 4-cls\n(final)"]
    acc = [0.49, 0.59, 0.68, 0.83, 0.79]
    colors = [C["fog"] if False else "#999999"] * 2 + [C["bar2"], C["bar"], C["accent"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(stages, acc, color=colors)
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Progression: Rules -> Learned Classifier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/accuracy_progression.png", dpi=150)
    plt.close(fig)


def fig_composition(counts):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    labs = list(counts.keys()); vals = [counts[l] for l in labs]
    ax.bar(labs, vals, color=[C[l] for l in labs])
    for i, v in enumerate(vals):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Images"); ax.set_title("Dataset Composition (balanced sample)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/dataset_composition.png", dpi=150)
    plt.close(fig)


def fig_fog_rain_metrics():
    """Median rule-features that separate fog vs rain (from real DAWN)."""
    feats = ["laplacian", "hf_energy(x100)", "saturation", "edge_density(x100)"]
    fog = [9, 49, 17, None]; rain = [30, 56, 26, None]
    # compute edge density medians live for honesty
    def med_edge(folder):
        vals = []
        for p in _imgs(folder, 200):
            im = cv2.imread(p)
            if im is None:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            vals.append(float((cv2.Canny(g, 50, 150) > 0).mean()) * 100)
        return float(np.median(vals))
    fog[3] = med_edge("data/raw/DAWN/Fog")
    rain[3] = med_edge("data/raw/DAWN/Rain")
    x = np.arange(len(feats)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - w / 2, fog, w, label="FOG", color=C["FOG"])
    ax.bar(x + w / 2, rain, w, label="RAIN", color=C["RAIN"])
    ax.set_xticks(x); ax.set_xticklabels(feats, fontsize=9)
    ax.set_title("FOG vs RAIN — Discriminative Features (medians)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fog_rain_features.png", dpi=150)
    plt.close(fig)


def fig_enhancement_montage():
    """Real-image before/after: classify each real frame, then enhance it."""
    from src.preprocessing.enhancements import FrameEnhancer
    from src.preprocessing.scene_classifier_ml import LearnedSceneClassifier
    enh = FrameEnhancer()
    clf = LearnedSceneClassifier()   # the final integrated (trained) classifier
    # one representative real image per condition
    picks = [
        ("NIGHT", "data/raw/ExDark"),
        ("FOG", "data/raw/DAWN/Fog"),
        ("RAIN", "data/raw/DAWN/Rain"),
    ]
    fig, axes = plt.subplots(len(picks), 2, figsize=(7.4, 8.4))
    for r, (cond, folder) in enumerate(picks):
        imgs = _imgs(folder, 40)
        img = None
        for p in imgs:
            cand = cv2.imread(p)
            if cand is not None:
                img = cand
                break
        if img is None:
            for c in range(2):
                axes[r][c].axis("off")
            continue
        label = clf.classify(img).label
        after = enh.enhance(img, label)
        for c, (im, kind) in enumerate([(img, "before"), (after, "after")]):
            ax = axes[r][c]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            title = f"{cond} — {kind}"
            if c == 1:
                title += f"  (detected {label})"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Adaptive Enhancement on Real Benchmark Images (before / after)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/enhancement_montage.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- #
def main():
    print("Extracting features from real datasets...")
    X, y, counts = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3])
    rep = classification_report(yte, pred, labels=[0, 1, 2, 3],
                                target_names=LABELS, output_dict=True,
                                zero_division=0)

    # Figures
    fig_confusion(cm)
    fig_per_class(rep)
    fig_importance(clf.feature_importances_)
    fig_progression()
    fig_composition(counts)
    fig_fog_rain_metrics()
    fig_enhancement_montage()

    metrics = {
        "accuracy": acc,
        "n_val": int(len(yte)),
        "n_total": int(len(y)),
        "counts": counts,
        "confusion": cm.tolist(),
        "labels": LABELS,
        "per_class": {l: {k: rep[l][k] for k in
                          ["precision", "recall", "f1-score", "support"]}
                      for l in LABELS},
        "feature_importance": dict(sorted(
            zip(FEATURE_NAMES, clf.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True)),
        "top_features": [n for n, _ in sorted(
            zip(FEATURE_NAMES, clf.feature_importances_),
            key=lambda kv: kv[1], reverse=True)[:6]],
    }
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nAccuracy {acc:.3f} | assets + metrics.json -> {OUT}/")


if __name__ == "__main__":
    main()
