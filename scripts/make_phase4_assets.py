"""
scripts/make_phase4_assets.py
-------------------------------
Generate figures for the Phase 4 report PDF from results/phase4/benchmark.json.

Outputs -> outputs/report_p4/*.png
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

OUT = "outputs/report_p4"
os.makedirs(OUT, exist_ok=True)

NAVY, BLUE, RED, GREY, GREEN, ORANGE = ("#1b2a4a", "#2e86c1", "#c0392b",
                                        "#7fb3d5", "#1e8449", "#f39c12")

D = json.load(open("results/phase4/benchmark.json"))
ROWS = D["rows"]


def rows_for(model):
    return [r for r in ROWS if r["model"] == model]


def fig_pruning_recovery():
    """The headline finding: naive prune collapse vs fine-tune recovery."""
    models = ["SmallCNN", "RTMDet-CSPNeXt"]
    configs = ["fp32", "pruned_naive", "pruned"]
    labels = ["FP32", "Pruned\n(no fine-tune)", "Pruned\n(+fine-tune)"]
    colors = [GREY, RED, GREEN]
    x = np.arange(len(models)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, cfg in enumerate(configs):
        vals = []
        for m in models:
            r = next((r for r in rows_for(m) if r["config"] == cfg), None)
            vals.append(r["accuracy"] if r else 0)
        offs = (i - 1) * w
        bars = ax.bar(x + offs, vals, w, label=labels[i], color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Naive Pruning Collapses Accuracy -- Fine-Tuning Recovers It")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/pruning_recovery.png", dpi=150); plt.close(fig)


def fig_size_reduction():
    models = ["SmallCNN", "RTMDet-CSPNeXt"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(models)); w = 0.35
    fp32_sizes, int8_sizes = [], []
    for m in models:
        fp32_sizes.append(next(r["size_mb"] for r in rows_for(m) if r["config"] == "fp32"))
        int8_sizes.append(next(r["size_mb"] for r in rows_for(m) if r["config"] == "onnx_int8"))
    b1 = ax.bar(x - w / 2, fp32_sizes, w, label="FP32", color=GREY)
    b2 = ax.bar(x + w / 2, int8_sizes, w, label="ONNX INT8", color=BLUE)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                    f"{b.get_height():.2f}MB", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Model size (MB)"); ax.set_title("Model Size: FP32 vs ONNX INT8")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/size_reduction.png", dpi=150); plt.close(fig)


def fig_latency_by_config():
    models = ["SmallCNN", "RTMDet-CSPNeXt"]
    configs = ["fp32", "pruned", "int8", "onnx_int8"]
    colors = [GREY, ORANGE, BLUE, GREEN]
    x = np.arange(len(configs)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(models):
        vals = [next((r["latency_ms"] for r in rows_for(m) if r["config"] == c), 0)
               for c in configs]
        ax.bar(x + (i - 0.5) * w, vals, w, label=m,
              color=[NAVY, BLUE][i])
    ax.set_xticks(x); ax.set_xticklabels(["FP32", "Pruned+FT", "INT8", "ONNX INT8"])
    ax.set_ylabel("Latency (ms/image)")
    ax.set_title("Latency Across Optimization Configs (CPU, batch=1)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/latency.png", dpi=150); plt.close(fig)


def fig_retention_summary():
    labels, rets, colors = [], [], []
    for r in ROWS:
        if r["config"] in ("fp32",):
            continue
        labels.append(f"{r['model']}\n{r['config']}")
        ret = r.get("accuracy_retention") or 0
        rets.append(ret)
        colors.append(GREEN if ret >= 0.95 else (ORANGE if ret >= 0.7 else RED))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, rets, color=colors)
    ax.axvline(1.0, color=GREY, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Accuracy retention vs FP32 baseline")
    ax.set_title("Accuracy Retention -- Every Optimized Config")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/retention_summary.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_pruning_recovery()
    fig_size_reduction()
    fig_latency_by_config()
    fig_retention_summary()
    print(f"assets -> {OUT}/")
