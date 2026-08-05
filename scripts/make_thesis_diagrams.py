"""
scripts/make_thesis_diagrams.py
---------------------------------
Generates the DFD (Level 0/1/2) and UML-style diagrams (class, activity, use
case) for the thesis document, drawn from the ACTUAL pipeline architecture in
main.py / src/*. Box-and-arrow diagrams via matplotlib -- no external drawing
tool, no fabricated components.

Outputs -> outputs/thesis/*.png
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

OUT = "outputs/thesis"
os.makedirs(OUT, exist_ok=True)

NAVY = "#1b2a4a"; BLUE = "#2e86c1"; LIGHT = "#eaf2f8"; GREY = "#7f8c8d"
GREEN = "#1e8449"; ORANGE = "#e67e22"; YELLOW = "#f4d03f"


def box(ax, xy, w, h, text, color=LIGHT, edge=NAVY, fontsize=9, textcolor="black"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
           fontsize=fontsize, color=textcolor, wrap=True)
    return (x + w / 2, y + h / 2, x, y, w, h)


def arrow(ax, p1, p2, label=None, color="black", style="-|>"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                        linewidth=1.2, color=color)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.15, label, fontsize=7.5, color=color, ha="center")


def new_fig(w=9, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ============================================================ DFD LEVEL 0
def dfd_level0():
    fig, ax = new_fig(8, 6)
    cx, cy = 4, 3
    circ = Circle((cx, cy), 1.0, facecolor=BLUE, edgecolor=NAVY, linewidth=1.5)
    ax.add_patch(circ)
    ax.text(cx, cy, "Surveillance\nPipeline\n(SVS)", ha="center", va="center",
           fontsize=9, color="white", fontweight="bold")

    ents = {"Video Feed": (0.7, 5), "Config /\nWeights": (6.3, 5),
           "Operator": (0.7, 1), "Annotated\nOutput": (6.3, 1)}
    for name, (x, y) in ents.items():
        box(ax, (x - 0.7, y - 0.4), 1.4, 0.8, name, color="#dcdde1", fontsize=8)

    arrow(ax, (1.1, 4.7), (cx - 0.75, cy + 0.6), "frames")
    arrow(ax, (5.9, 4.7), (cx + 0.75, cy + 0.6), "model/thresholds")
    arrow(ax, (cx - 0.6, cy - 0.7), (1.1, 1.3), "risk alerts")
    arrow(ax, (cx + 0.6, cy - 0.7), (5.9, 1.3), "annotated video")
    ax.set_title("Figure. DFD Level 0 (Context Diagram)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/dfd_level0.png", dpi=150); plt.close(fig)


# ============================================================ DFD LEVEL 1
def dfd_level1():
    fig, ax = new_fig(10, 6.5)
    stages = [
        ("1.0\nScene\nClassify", 0.5, 3.5),
        ("2.0\nAdaptive\nEnhance", 2.3, 3.5),
        ("3.0\nDetect\n(Helmet+Wheel)", 4.1, 3.5),
        ("4.0\nTrack + Vote\n+ ReID", 5.9, 3.5),
        ("5.0\nRisk\nIndex", 7.7, 3.5),
    ]
    prev_right_edge = None
    for name, x, y in stages:
        box(ax, (x, y), 1.6, 1.0, name, color=LIGHT, fontsize=7.8)
        if prev_right_edge is not None:
            arrow(ax, (prev_right_edge, y + 0.5), (x, y + 0.5))
        prev_right_edge = x + 1.6
    box(ax, (0.5, 0.8), 1.6, 0.6, "Video Frame", color="#dcdde1", fontsize=7.5)
    box(ax, (7.7, 0.8), 1.6, 0.6, "Annotated Output", color="#dcdde1", fontsize=7.5)
    box(ax, (4.1, 5.3), 1.6, 0.6, "config/settings.yaml", color="#f9e79f", fontsize=7.5)
    arrow(ax, (1.3, 1.4), (1.3, 3.5))
    arrow(ax, (8.5, 3.5), (8.5, 1.4))
    arrow(ax, (4.9, 5.3), (4.9, 4.5))
    ax.set_title("Figure. DFD Level 1 (Pipeline Stages)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/dfd_level1.png", dpi=150); plt.close(fig)


# ============================================================ DFD LEVEL 2 (Detection)
def dfd_level2():
    fig, ax = new_fig(9, 6.5)
    steps = [
        ("2.1 Crop\nRider Region", 0.5, 4.5),
        ("2.2 Helmet\nYOLOv8 Detect", 0.5, 3.0),
        ("2.3 Wheel\nClassify (CNN/\nCSPNeXt/YOLO)", 3.2, 3.0),
        ("2.4 Compliance\nSemantics (by name)", 5.9, 3.0),
        ("2.5 Emit\nDetections", 5.9, 4.5),
    ]
    for name, x, y in steps:
        box(ax, (x, y), 2.2, 1.0, name, color=LIGHT, fontsize=7.6)
    arrow(ax, (1.6, 4.5), (1.6, 4.0))
    arrow(ax, (2.7, 3.5), (3.2, 3.5))
    arrow(ax, (5.4, 3.5), (5.9, 3.5))
    arrow(ax, (7.0, 4.0), (7.0, 4.5))
    box(ax, (0.5, 1.2), 2.2, 0.6, "weights/*.pt", color="#f9e79f", fontsize=7.5)
    arrow(ax, (1.6, 1.8), (1.6, 3.0))
    ax.set_title("Figure. DFD Level 2 - Detection Sub-Process (3.0)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/dfd_level2.png", dpi=150); plt.close(fig)


# ============================================================ UML CLASS DIAGRAM
def class_diagram():
    fig, ax = new_fig(11, 7.7)

    def uml_class(x, y, w, h, title, attrs, methods, color=LIGHT):
        box(ax, (x, y), w, h, "", color=color, fontsize=1)
        ax.plot([x, x + w], [y + h - 0.35, y + h - 0.35], color=NAVY, linewidth=1)
        title_dy = 0.28 if "\n" in title else 0.2
        ax.text(x + w / 2, y + h - title_dy, title, ha="center", fontsize=7.6, fontweight="bold")
        body_h = h - 0.35
        n_attr = len(attrs)
        attr_y0 = y + body_h - 0.15
        for i, a in enumerate(attrs):
            ax.text(x + 0.1, attr_y0 - i * 0.22, f"- {a}", fontsize=6.6, va="top")
        split_y = attr_y0 - n_attr * 0.22 - 0.05
        ax.plot([x, x + w], [split_y, split_y], color=NAVY, linewidth=0.6)
        for i, m in enumerate(methods):
            ax.text(x + 0.1, split_y - 0.18 - i * 0.2, f"+ {m}()", fontsize=6.6, va="top")
        return {"cx": x + w / 2, "top": y + h, "bottom": y, "right": x + w, "left": x}

    p1 = uml_class(0.3, 5.0, 2.6, 2.4, "SurveillancePipeline",
                   ["scene", "enhancer", "helmet", "wheels", "tracker",
                    "reid", "demographics", "risk"],
                   ["process_frame", "run"], color="#d6eaf8")
    p2 = uml_class(3.4, 5.5, 2.3, 1.5, "SceneClassifier /\nLearnedSceneClassifier",
                   ["labels"], ["classify"])
    p3 = uml_class(6.0, 5.5, 2.3, 1.5, "HelmetDetector",
                   ["class_names", "violation_ids"], ["detect"])
    p4 = uml_class(8.6, 5.5, 2.1, 1.5, "WheelClassifier",
                   ["backend"], ["classify_one"])
    p5 = uml_class(3.4, 3.2, 2.3, 1.6, "VehicleTracker",
                   ["_backend", "vote_window"], ["update"])
    p6 = uml_class(6.0, 3.2, 2.3, 1.6, "VehicleReID",
                   ["_backend", "gallery"], ["match", "embed"])
    p7 = uml_class(8.6, 3.2, 2.1, 1.6, "RiskIndexer",
                   ["weights", "thresholds"], ["compute"])
    p8 = uml_class(3.4, 0.8, 2.3, 1.6, "DemographicsEstimator",
                   ["enabled", "queue_dir"], ["estimate", "save_for_offline"])

    def bus_fanout(exit_xy, targets_top, bus_y):
        """Orthogonal tree connector: drop/rise to a horizontal bus clear of
        every box, then a short arrowed stub into each target's edge -- avoids
        diagonal lines cutting through box titles/attributes."""
        ex, ey = exit_xy
        ax.plot([ex, ex], [ey, bus_y], color=NAVY, linewidth=1.1)
        xs = [t[0] for t in targets_top] + [ex]
        ax.plot([min(xs), max(xs)], [bus_y, bus_y], color=NAVY, linewidth=1.1)
        for tx, ty in targets_top:
            arrow(ax, (tx, bus_y), (tx, ty))

    top_targets = [(p2["cx"], p2["top"]), (p3["cx"], p3["top"]), (p4["cx"], p4["top"])]
    bus_fanout((p1["right"], 7.1), top_targets, 7.1)
    bottom_targets = [(p5["cx"], p5["top"]), (p6["cx"], p6["top"]), (p7["cx"], p7["top"])]
    bus_fanout((p1["cx"], p1["bottom"]), bottom_targets, 4.9)
    arrow(ax, (p5["cx"], p5["bottom"]), (p8["cx"], p8["top"]))

    ax.set_title("Figure. UML Class Diagram (Pipeline Core Classes)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/uml_class.png", dpi=150); plt.close(fig)


# ============================================================ ACTIVITY DIAGRAM
def activity_diagram():
    fig, ax = new_fig(7.5, 10)
    y = 9.4
    start = Circle((3.75, y), 0.15, facecolor="black")
    ax.add_patch(start); y -= 0.6

    steps = [
        "Read video frame",
        "Classify scene\n(DAY/NIGHT/FOG/RAIN)",
        "Apply adaptive enhancement\n(only if needed)",
        "Run helmet + wheel detection",
        "Update DeepSORT tracks\n+ majority-vote wheel class",
        "Match vehicle via\ncross-camera ReID",
        "Compute risk index\n(LOW / MEDIUM / HIGH)",
        "Queue face crop for\noffline demographics (if enabled)",
        "Render overlays +\ntelemetry HUD",
        "Write annotated frame",
    ]
    prev_y = y
    for i, s in enumerate(steps):
        cy, *_ = box(ax, (1.75, prev_y - 0.55), 4.0, 0.7, s, color=LIGHT, fontsize=7.8)[:1] or (prev_y,)
        arrow(ax, (3.75, prev_y), (3.75, prev_y - 0.55))
        prev_y -= 0.85

    arrow(ax, (3.75, prev_y + 0.3), (3.75, prev_y - 0.1))
    end_outer = Circle((3.75, prev_y - 0.3), 0.18, facecolor="none",
                       edgecolor="black", linewidth=1.4)
    end_inner = Circle((3.75, prev_y - 0.3), 0.09, facecolor="black")
    ax.add_patch(end_outer); ax.add_patch(end_inner)
    ax.text(3.9, y + 0.3, "start", fontsize=7, color=GREY)

    ax.set_title("Figure. Activity Diagram - Per-Frame Pipeline Flow", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/uml_activity.png", dpi=150); plt.close(fig)


# ============================================================ USE CASE DIAGRAM
def use_case_diagram():
    fig, ax = new_fig(9, 6.5)

    def actor(x, y, label):
        ax.plot([x, x], [y, y - 0.5], color="black", linewidth=1.3)
        ax.plot([x - 0.25, x + 0.25], [y - 0.2, y - 0.2], color="black", linewidth=1.3)
        ax.plot([x, x - 0.2], [y - 0.5, y - 0.85], color="black", linewidth=1.3)
        ax.plot([x, x + 0.2], [y - 0.5, y - 0.85], color="black", linewidth=1.3)
        head = Circle((x, y + 0.12), 0.12, facecolor="none", edgecolor="black", linewidth=1.3)
        ax.add_patch(head)
        ax.text(x, y - 0.95, label, ha="center", va="top", fontsize=8, fontweight="bold")
        return (x, y - 0.4)

    op = actor(0.8, 5.5, "Operator")
    svc = actor(8.2, 5.5, "External\nSystem\n(demographics\nprocess)")

    ellipses = {}

    def ellipse(x, y, w, h, label):
        from matplotlib.patches import Ellipse
        e = Ellipse((x, y), w, h, facecolor=LIGHT, edgecolor=NAVY, linewidth=1.3)
        ax.add_patch(e)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5)
        ellipses[(x, y)] = (w / 2, h / 2)
        return (x, y)

    def edge_point(center, rx, ry, toward):
        import math
        dx, dy = toward[0] - center[0], toward[1] - center[1]
        ang = math.atan2(dy, dx)
        return (center[0] + rx * math.cos(ang), center[1] + ry * math.sin(ang))

    def connect(p_from, p_to, label=None, style="-"):
        rx_to, ry_to = ellipses.get(p_to, (0, 0))
        rx_from, ry_from = ellipses.get(p_from, (0, 0))
        start = edge_point(p_from, rx_from, ry_from, p_to) if (rx_from or ry_from) else p_from
        end = edge_point(p_to, rx_to, ry_to, p_from) if (rx_to or ry_to) else p_to
        arrow(ax, start, end, label, style=style)

    uc1 = ellipse(3.5, 5.0, 2.4, 0.9, "Run Surveillance\non Video Feed")
    uc2 = ellipse(3.5, 3.5, 2.4, 0.9, "View Risk Alerts\n&& Telemetry")
    uc3 = ellipse(3.5, 2.0, 2.4, 0.9, "Retrain / Optimize\nModels (offline)")
    uc4 = ellipse(6.3, 2.7, 2.2, 0.9, "Process Queued\nFace Crops")

    for uc in (uc1, uc2, uc3):
        connect(op, uc, style="-")
    connect(uc4, svc, style="-")
    connect(uc1, uc4, "<<extend>>")

    ax.set_title("Figure. Use Case Diagram", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/uml_usecase.png", dpi=150); plt.close(fig)


# ============================================================ SEQUENCE DIAGRAM (process isolation)
def sequence_diagram():
    """Real architecture: the live pipeline process (which has torch loaded)
    never calls DeepFace directly -- it queues crops to disk and a separate
    process, started independently and never importing torch, drains the
    queue. Drawn as a standard UML sequence diagram (lifelines + messages)."""
    fig, ax = new_fig(9.5, 7)
    lifelines = [("Video Frame", 1.2), ("SurveillancePipeline\n(torch loaded)", 3.6),
                 ("outputs/demographics_queue/\n(disk)", 6.4), ("Offline Demographics Process\n(never imports torch)", 8.6)]
    top_y, bot_y = 6.3, 0.6
    for name, x in lifelines:
        box(ax, (x - 0.9, top_y), 1.8, 0.55, name, color=LIGHT, fontsize=7)
        ax.plot([x, x], [top_y, bot_y], color=GREY, linewidth=1, linestyle=(0, (4, 3)))

    def msg(y, x1, x2, label, dashed=False):
        arrow(ax, (x1, y), (x2, y), style="-|>" if not dashed else "-|>",
             color=NAVY if not dashed else GREY)
        ax.plot([x1, x2], [y, y], color=NAVY if not dashed else GREY,
                linewidth=1.1, linestyle="--" if dashed else "-")
        ax.text((x1 + x2) / 2, y + 0.13, label, fontsize=7, ha="center")

    msg(5.6, 1.2, 3.6, "frame")
    msg(5.0, 3.6, 3.6 + 0.01, "detect + crop face region", dashed=False)
    msg(4.4, 3.6, 6.4, "save_for_offline_analysis(crop)")
    msg(3.6, 3.6, 1.2 + 1.0, "annotated frame (no demographics yet)")
    ax.text(3.6, 3.15, "-- separate OS process, started independently --", fontsize=6.8,
           ha="center", color=GREY, style="italic")
    msg(2.5, 6.4, 8.6, "poll queue dir")
    msg(1.9, 8.6, 8.6 - 0.01, "run DeepFace (age, gender)")
    msg(1.3, 8.6, 6.4, "write results/phase3/demographics.json")
    ax.set_title("Figure. Sequence Diagram - Process-Isolated Demographics", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/uml_sequence.png", dpi=150); plt.close(fig)


# ============================================================ GATE A LEARNING CURVE (real data)
def gate_learning_curve():
    import json
    gates = json.load(open("results/phase0_gates.json"))
    pts = gates["gate_A_helmet_data"]["points"]
    xs = [p["train_images"] for p in pts]
    ys = [p["test_mAP50"] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, marker="o", color=BLUE, linewidth=2, markersize=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=8)
    ax.set_xlabel("Training images"); ax.set_ylabel("Test mAP@50")
    ax.set_title("Gate A: Helmet Detector Learning Curve (Real Data)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/gate_a_learning_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    dfd_level0()
    dfd_level1()
    dfd_level2()
    class_diagram()
    activity_diagram()
    use_case_diagram()
    sequence_diagram()
    gate_learning_curve()
    print(f"diagrams -> {OUT}/")
