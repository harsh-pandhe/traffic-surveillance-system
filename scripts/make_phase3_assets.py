"""
scripts/make_phase3_assets.py
-------------------------------
Generate figures for the Phase 3 report PDF from results/phase3/*.json.

Outputs -> outputs/report_p3/*.png
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/report_p3"
os.makedirs(OUT, exist_ok=True)

NAVY, BLUE, RED, GREY, GREEN = "#1b2a4a", "#2e86c1", "#c0392b", "#7fb3d5", "#1e8449"

OCC = json.load(open("results/phase3/occlusion_ablation.json"))
RISK = json.load(open("results/phase3/risk_indexer_validation.json"))
REID = json.load(open("results/phase3/reid_benchmark.json"))
HELMET_STATS = json.load(open("results/phase3/helmet_dataset_stats.json"))


def fig_flip_rate():
    ns = OCC["vote_windows"]
    vals = [OCC["flip_rate_overall"][str(n)] for n in ns]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(ns, vals, "o-", color=NAVY, linewidth=2, markersize=7)
    for n, v in zip(ns, vals):
        ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Vote window N"); ax.set_ylabel("Flip-rate")
    ax.set_title("Prediction Stability vs Vote Window (lower = more stable)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/flip_rate.png", dpi=150); plt.close(fig)


def fig_balanced_accuracy():
    ns = OCC["vote_windows"]
    bal = [OCC["overall"][str(n)] for n in ns]
    raw = [OCC["raw_accuracy"][str(n)] for n in ns]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(ns, bal, "o-", color=BLUE, label="Balanced accuracy", linewidth=2)
    ax.plot(ns, raw, "s--", color=GREY, label="Raw accuracy (97% one class)",
           linewidth=1.5)
    ax.set_xlabel("Vote window N"); ax.set_ylabel("Accuracy")
    ax.set_title("Wheel-Count Accuracy vs Vote Window")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/balanced_accuracy.png", dpi=150); plt.close(fig)


def fig_occlusion_bands():
    bands = ["none", "light", "medium", "heavy"]
    support = [OCC["band_support"][b] for b in bands]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bands, support, color=[GREEN, BLUE, "#f39c12", RED])
    for i, v in enumerate(support):
        ax.text(i, v + max(support) * 0.01, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Instances"); ax.set_title("UA-DETRAC Occlusion Band Support")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/occlusion_bands.png", dpi=150); plt.close(fig)


def fig_risk_distribution():
    dist = RISK["risk_level_distribution"]
    levels = ["LOW", "MEDIUM", "HIGH"]
    vals = [dist.get(l, 0) for l in levels]
    colors = [GREEN, "#f39c12", RED]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(levels, vals, color=colors)
    total = sum(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                f"{v}\n({v*100/total:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Motorcycle tracks"); ax.set_title("Risk Level Distribution (10,006 real tracks)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/risk_distribution.png", dpi=150); plt.close(fig)


def fig_occupancy():
    dist = HELMET_STATS["rider_count_distribution"]
    keys = sorted(dist, key=lambda k: int(k))
    vals = [dist[k] for k in keys]
    colors = [GREEN if int(k) <= 2 else RED for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(keys, vals, color=colors)
    total = sum(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                f"{v*100/total:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Riders per motorcycle"); ax.set_ylabel("Instances")
    ax.set_title("Real-World Occupancy (HELMET, 283,377 instances)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/occupancy.png", dpi=150); plt.close(fig)


def fig_reid():
    metrics = ["Rank-1", "Rank-5", "Rank-10", "mAP"]
    vals = [REID["rank1"], REID["rank5"], REID["rank10"], REID["mAP"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(metrics, vals, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("VeRi-776 Cross-Camera ReID (off-the-shelf OSNet)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/reid.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_flip_rate()
    fig_balanced_accuracy()
    fig_occlusion_bands()
    fig_risk_distribution()
    fig_occupancy()
    fig_reid()
    print(f"assets -> {OUT}/")
