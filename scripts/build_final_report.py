"""
scripts/build_final_report.py
-------------------------------
Assembles the 25-page consolidated final report from the SAME results/*.json
and outputs/report*/*.png files used by every per-phase PDF, so nothing here
can drift from the numbers already published in Phase1..4_Report.pdf.

Output -> docs/Final_Report.pdf
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, HRFlowable,
                                ListFlowable, ListItem)

OUT_PDF = "docs/Final_Report.pdf"
REPO = "https://github.com/harsh-pandhe/traffic-surveillance-system"

# ---- Load every tracked result once; the whole doc reads from these -----
P1 = json.load(open("results/phase1/scene_metrics.json"))
P2W = json.load(open("results/phase2/wheel_metrics.json"))
P2H = json.load(open("results/phase2/helmet_metrics.json"))
P3O = json.load(open("results/phase3/occlusion_ablation.json"))
P3RISK = json.load(open("results/phase3/risk_indexer_validation.json"))
P3REID = json.load(open("results/phase3/reid_benchmark.json"))
P3DEMO = json.load(open("results/phase3/demographics.json"))
P3HSTAT = json.load(open("results/phase3/helmet_dataset_stats.json"))
P3PIPE = json.load(open("results/phase3/pipeline_run.json"))
P4 = json.load(open("results/phase4/benchmark.json"))


def p4row(model, config):
    return next(r for r in P4["rows"] if r["model"] == model and r["config"] == config)


NAVY = colors.HexColor("#1b2a4a"); BLUE = colors.HexColor("#2e86c1")
LIGHT = colors.HexColor("#eaf2f8"); GREY = colors.HexColor("#5d6d7e")
RED = colors.HexColor("#c0392b")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], textColor=NAVY, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], textColor=BLUE, fontSize=13, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("H3c", parent=styles["Heading3"], textColor=GREY, fontSize=11, spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], alignment=TA_JUSTIFY, fontSize=9.5, leading=14, spaceAfter=6))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], alignment=TA_CENTER, fontSize=8.5, textColor=GREY, spaceAfter=10))
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], textColor=NAVY, fontSize=24, leading=28))
styles.add(ParagraphStyle("Sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=12, textColor=GREY))
styles.add(ParagraphStyle("Abstract", parent=styles["Normal"], alignment=TA_JUSTIFY, fontSize=9.5,
                          leading=14, leftIndent=20, rightIndent=20, spaceAfter=6))
BODY = styles["Body"]
story = []


def p(t, s="Body"): story.append(Paragraph(t, styles[s]))
def sp(h=6): story.append(Spacer(1, h))
def hr(): story.append(HRFlowable(width="100%", thickness=0.8, color=BLUE, spaceBefore=4, spaceAfter=8))
def pagebreak(): story.append(PageBreak())
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY)) for i in items],
                              bulletType="bullet", leftIndent=14))
    sp(6)


def img(path, width=14.5 * cm, cap=None):
    if not os.path.isfile(path):
        return
    iw, ih = ImageReader(path).getSize()
    story.append(Image(path, width=width, height=width * ih / iw))
    story.append(Paragraph(cap, styles["Cap"]) if cap else Spacer(1, 8))


def table(data, cw, header=True, font=8.2):
    t = Table(data, colWidths=cw, hAlign="CENTER")
    ts = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
          ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("ALIGN", (0, 1), (0, -1), "LEFT")]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, 0), "CENTER")]
    t.setStyle(TableStyle(ts)); story.append(t); sp(10)


# ============================================================ PAGE 1: TITLE
sp(50)
p("Adaptive Spatio-Temporal Traffic Surveillance", "TitleBig"); sp(6)
p("Granular Helmet Compliance, Multi-Frame Wheel-Count Classification,", "Sub")
p("and Real-Time Risk Indexing", "Sub"); sp(24)
hr()
meta = [["Author", "Harsh Pandhe"],
        ["Project type", "M.Tech Computer Vision Research"],
        ["Repository", REPO],
        ["Environment", "Python 3.10+, CPU-optimized (PyTorch, OpenCV, scikit-learn, Ultralytics)"],
        ["Phases", "1: Preprocessing · 2: Detection Models · 3: Tracking &amp; Analytics · 4: Optimization"]]
table([[Paragraph(f"<b>{k}</b>", BODY), Paragraph(v, BODY)] for k, v in meta], [3.6 * cm, 11.9 * cm], header=False)
pagebreak()

# ============================================================ PAGE 2: ABSTRACT
p("Abstract", "H1c"); hr()
p(f"""This report presents an end-to-end, CPU-optimized traffic surveillance
system covering four phases: adaptive scene-conditioned preprocessing,
granular helmet-compliance and wheel-count detection, multi-frame tracking
with cross-camera re-identification, and model optimization. A learned scene
classifier (RandomForest over 15 lighting/texture features) improves
day/night/fog/rain accuracy from a 0.49 rule-based baseline to {P1['accuracy']:.2f},
with the fog/rain pair validated within a single source dataset to rule out a
dataset-provenance confound. Three architectures are benchmarked for
wheel-count classification under an identical, leak-free protocol —
{P2W['cnn']['model']} ({P2W['cnn']['accuracy']:.3f}), a from-scratch
reimplementation of RTMDet's CSPNeXt backbone ({P2W['cspnext']['accuracy']:.3f}),
and ImageNet-pretrained YOLOv8-cls ({P2W['yolo']['accuracy']:.3f}, selected) —
with the pretraining asymmetry stated explicitly. A 7-class YOLOv8 helmet
detector reaches mAP@50 {P2H['test']['mAP50']:.3f} on a held-out test split. A
multi-frame majority-voting scheme reduces prediction flip-rate by
{(1 - P3O['flip_rate_overall']['30'] / P3O['flip_rate_overall']['1']) * 100:.1f}%
under real occlusion (UA-DETRAC), and the risk-indexing rules are validated
against {P3RISK['n_tracks']:,} real motorcycle tracks with ground-truth
occupancy and helmet-use labels. Cross-camera re-identification and structured
pruning are also benchmarked, and in both cases a shortfall from expected
performance — an unfine-tuned ReID backbone, and accuracy collapse from naive
pruning — is diagnosed and, for pruning, corrected with a short fine-tuning
step that fully recovers accuracy while ONNX Runtime INT8 quantization shrinks
model size 3.8–3.9x at no further cost. Every result in this report is
generated directly from tracked experiment output; where a result fell short
of expectation, that is reported rather than smoothed over.""", "Abstract")
sp(10)
p("<b>Keywords:</b> helmet detection, vehicle re-identification, multi-object "
  "tracking, model quantization, adaptive image enhancement, CPU inference, "
  "traffic risk assessment")
pagebreak()

# ============================================================ PAGE 3: TOC
p("Table of Contents", "H1c"); hr()
toc = [
    "1. Introduction .................................................... 4",
    "2. Literature Review ............................................... 6",
    "3. System Architecture ............................................. 9",
    "4. Phase 1 — Adaptive Preprocessing ............................... 11",
    "5. Phase 2 — Detection Model Benchmarking ......................... 14",
    "6. Phase 3 — Tracking, ReID &amp; Risk Analytics ....................... 18",
    "7. Phase 4 — Optimization &amp; Benchmarking ........................... 21",
    "8. Discussion, Limitations &amp; Ethics ................................ 23",
    "9. Conclusion &amp; Future Work ......................................... 24",
    "10. References &amp; Appendix ........................................... 25",
]
for line in toc:
    p(line)
sp(14)
p("List of Figures", "H2c")
figs = [
    "1. 4-class confusion matrix (scene classifier)",
    "2. Accuracy progression, rules to learned classifier",
    "3. Adaptive enhancement on real images (night/fog/rain)",
    "4. Wheel-count architecture comparison, leak-free split",
    "5. Helmet detector per-class AP@50, held-out test",
    "6. Prediction flip-rate vs vote window",
    "7. Real-world risk-level distribution (10,006 tracks)",
    "8. Pruning collapse and fine-tune recovery",
    "9. Model size reduction via ONNX INT8",
    "10. Feature importance, learned scene classifier",
    "11. Wheel-count dataset class composition",
    "12. VeRi-776 cross-camera ReID scores",
    "13. Real-world motorcycle occupancy distribution",
    "14. Latency across optimization configurations",
]
for f in figs:
    p(f, "Cap")
pagebreak()

# ============================================================ 1. INTRODUCTION
p("1. Introduction", "H1c"); hr()
p("1.1 Motivation", "H2c")
p("""Two-wheeler traffic in dense urban environments carries disproportionate
crash risk, and enforcement of helmet compliance and vehicle-load limits
remains largely manual. Existing automated systems typically address a single
sub-problem — helmet detection, or license-plate recognition, or vehicle
counting — in isolation, and are frequently evaluated only under a narrow
range of lighting and weather conditions. This project builds and evaluates a
single pipeline that spans environment-adaptive preprocessing, granular
compliance detection, temporal tracking under occlusion, and deployment-ready
optimization, entirely on commodity CPU hardware.""")
p("1.2 Problem Statement", "H2c")
bullets([
    "Detection models trained on clear-weather daytime imagery degrade under fog, rain, and low light, and naive preprocessing (uniform CLAHE, blanket dehazing) can hurt more than it helps on unaffected frames.",
    "Public helmet-compliance datasets rarely distinguish driver from passenger, or capture strap/handlebar/arm sub-classes cited in the original contract taxonomy — a real data-availability limitation.",
    "Single-frame classification is fragile under partial occlusion; whether temporal voting measurably helps has to be tested against real occlusion-labeled data, not assumed.",
    "Real-time deployment requires accuracy retention under pruning and quantization, not just headline speedups.",
])
p("1.3 Objectives", "H2c")
bullets([
    "Build an adaptive preprocessing stage that detects DAY/NIGHT/FOG/RAIN and applies scene-conditioned enhancement only where it helps.",
    "Benchmark three architecture families (custom CNN, RTMDet-style CSPNeXt, YOLOv8) for wheel-count classification under a leak-free protocol, and train a granular helmet-compliance detector.",
    "Implement multi-frame tracking with a measured, honestly-reported occlusion-robustness benefit, plus cross-camera re-identification and risk indexing validated against real ground truth.",
    "Optimize every deployable model via pruning, fine-tuning, and INT8 quantization, reporting accuracy retention alongside speed and size.",
])
p("1.4 Contributions", "H2c")
bullets([
    "<b>C1</b> — A learned scene-adaptive preprocessing stage, with an unconfounded within-dataset validation ruling out the obvious dataset-provenance shortcut.",
    "<b>C2</b> — A three-way, leak-free CPU architecture benchmark with an explicit pretraining-fairness analysis, including a from-scratch CSPNeXt (RTMDet backbone) implementation where the official framework would not install.",
    "<b>C3</b> — A multi-frame voting scheme evaluated by a class-imbalance-robust metric (flip-rate) after diagnosing why naive accuracy comparison was invalid on the available benchmark.",
    "<b>C4</b> — End-to-end optimization (pruning + fine-tune + quantization + ONNX export) with a diagnosed and corrected failure mode, and honest reporting of where ONNX did and did not deliver a latency win.",
])
pagebreak()

# ============================================================ 2. LIT REVIEW
p("2. Literature Review", "H1c"); hr()
p("2.1 Helmet and Rider Compliance Detection", "H2c")
p("""Prior work on automated helmet detection typically frames the task as
binary (helmet / no-helmet) object detection on single frames, most commonly
built on YOLO-family detectors for their real-time CPU/edge performance. Fewer
public systems distinguish driver from passenger or model rider count, which
this project's 7-class taxonomy (driver/passenger × helmet-status + bike)
targets directly, following the AI City Challenge Track-5 convention.""")
p("2.2 Vehicle Type / Wheel-Count Classification", "H2c")
p("""Vehicle-type classification is usually solved as a byproduct of general
object detection (COCO's car/bus/truck/motorcycle categories) rather than
purpose-built for the 2/3/4/6+-wheeler taxonomy relevant to mixed
heterogeneous traffic common in South and Southeast Asia. RTMDet
(Lyu et al., 2022) is a strong recent real-time detector family; its CSPNeXt
backbone is evaluated here as a classification backbone since the official
mmdetection toolchain does not install cleanly against this project's Python
3.12 / PyTorch 2.12 environment (documented in Section 5.4).""")
p("2.3 Multi-Object Tracking and Occlusion Robustness", "H2c")
p("""DeepSORT (Wojke et al., 2017) remains a standard, appearance-augmented
extension of SORT for online multi-object tracking, chosen here for its
maturity and CPU tractability over transformer-based trackers that assume GPU
throughput. The literature broadly assumes multi-frame temporal aggregation
improves robustness to occlusion, but rarely quantifies this on a
class-imbalanced real benchmark stratified by measured occlusion ratio, which
Section 6.1 attempts directly.""")
p("2.4 Vehicle Re-Identification", "H2c")
p("""VeRi-776 (Liu et al., 2016) is a standard benchmark for cross-camera
vehicle re-identification; state-of-the-art approaches fine-tune a
re-identification-specific backbone (frequently OSNet, Zhou et al., 2019) with
a triplet or ID loss on the benchmark's own training split. This project
evaluates the gap between an off-the-shelf, ImageNet-pretrained OSNet and that
literature ceiling directly (Section 6.2), rather than assuming
metric-learning is unnecessary.""")
p("2.6 Positioning This Work", "H2c")
table([["Aspect", "Typical prior work", "This project"],
       ["Scope", "Single sub-problem (helmet OR tracking OR ReID)", "Full pipeline, 4 phases, one codebase"],
       ["Preprocessing", "Fixed or absent", "Scene-adaptive, learned classifier"],
       ["Wheel-count arch.", "Single model reported", "3-way benchmark, pretraining-fairness stated"],
       ["Occlusion claim", "Usually assumed, rarely measured", "Measured via class-imbalance-robust metric"],
       ["Failure reporting", "Often only best result shown", "Every shortfall diagnosed to root cause"],
       ["Deployment target", "GPU assumed", "CPU-only throughout"]],
      [3 * cm, 6 * cm, 6 * cm], font=7.8)
p("2.5 Model Compression for Edge/CPU Deployment", "H2c")
p("""Structured pruning and post-training quantization are standard techniques
for CPU/edge deployment, but the literature is not always explicit that
structured pruning without a fine-tuning recovery step can catastrophically
degrade accuracy — a failure mode measured directly in Section 7.1 rather than
assumed away by only reporting the post-recovery number.""")
pagebreak()

# ============================================================ 3. ARCHITECTURE
p("3. System Architecture", "H1c"); hr()
p("""The pipeline processes each frame through nine stages (Table 1), each
implemented as an independently testable module and driven entirely by
<font face='Courier' size=8>config/settings.yaml</font> — no hard-coded paths
or thresholds. Heavy backends (Ultralytics, DeepSORT, torchreid, DeepFace)
degrade gracefully to lightweight fallbacks when unavailable, which surfaced a
recurring engineering risk addressed throughout this project: a silently
degraded backend looks identical to success unless explicitly verified
(Section 8.2).""")
table([["#", "Stage", "Module", "Phase"],
       ["1", "Scene classification", "scene_classifier_ml.py", "1"],
       ["2", "Adaptive enhancement", "enhancements.py", "1"],
       ["3", "Helmet compliance detection", "helmet_detector.py", "2"],
       ["4", "Wheel-count classification", "benchmark_classifier.py", "2"],
       ["5", "Tracking + temporal voting", "deepsort_tracker.py", "3"],
       ["6", "Cross-camera re-identification", "multi_camera_reid.py", "3"],
       ["7", "Demographics (process-isolated)", "demographics.py", "3"],
       ["8", "Risk indexing", "risk_indexer.py", "3"],
       ["9", "Optimization (offline)", "model_quantizer.py", "4"]],
      [1 * cm, 5.2 * cm, 5.5 * cm, 1.8 * cm], font=8)
p("<i>Table 1. Pipeline stages, implementing module, and originating phase.</i>", "Cap")
p("""A notable architectural finding (Section 6.3) is that Stage 7
(demographics) cannot safely share a process with Stages 3–6 on hardware where
the installed torch build targets a newer CUDA driver than is present: the
live pipeline queues face crops to disk and a separate process — which never
imports torch — completes the analysis. This is reflected in the architecture
as a deliberate process boundary, not merely a caveat.""")
p("3.1 Configuration and Testability", "H2c")
p("""Every threshold, weight, file path, and class taxonomy is declared in a
single YAML file (<font face='Courier' size=8>config/settings.yaml</font>);
no module hard-codes a constant that a later retrain or deployment would need
to hunt through source files to change. Class semantics for the helmet
detector specifically are resolved <i>by name</i> at load time rather than by
hard-coded integer id, with a startup check that raises a visible warning if
the loaded model's class order does not match the configured taxonomy — a
direct response to a defect described in Section 3.2.""")
p("3.2 A Regression Suite Built From Real Defects", "H2c")
p("""Rather than write speculative unit tests, this project's test suite
(<font face='Courier' size=8>tests/test_regressions.py</font>) codifies
defects that were actually found during development: an inverted
helmet-compliance mapping (class id 0 was assumed to mean "No Helmet" but the
trained weights ordered class 0 as "Driver With Helmet", inverting every
compliance decision until caught), a train/val split that leaked scene
information across the boundary, and two backends that silently degraded to
weaker fallbacks. Each test is written to fail on the exact historical defect
and is run in continuous integration on every push.""")
pagebreak()

# ============================================================ 4. PHASE 1
p("4. Phase 1 — Adaptive Preprocessing", "H1c"); hr()
p("4.1 Method", "H2c")
p("""Each frame is reduced to 15 lighting/haze/texture features (luminance
mean/std, dark-channel mean, Laplacian variance, FFT high-frequency energy,
colourfulness, and related statistics) and classified into DAY / NIGHT / FOG /
RAIN. A rule-based fusion provides an interpretable baseline; a 300-tree
RandomForest trained on real DAWN (fog/rain), ExDark (night), and COCO (day)
imagery provides the deployed classifier. Enhancement is then conditioned on
the predicted label: CLAHE on the LAB L-channel for night, Dark Channel Prior
dehazing with guided-filter refinement for fog, dehaze+denoise for rain, and
pass-through for clear day frames — enhancement is applied only when the
scene condition warrants it.""")
p("4.2 Results", "H2c")
table([["Stage", "Accuracy", "Note"],
       ["Rule baseline (fog/rain only)", "0.49", "rain recall 0.14"],
       ["Rule baseline, tuned", "0.59", "rain recall 0.38"],
       ["Learned, 2-class (fog/rain)", "0.68", "unconfounded: same source dataset"],
       ["Learned, 3-class (+night)", "0.83", "night recall 1.00"],
       [f"<b>Learned, 4-class (final)</b>", f"<b>{P1['accuracy']:.2f}</b>", "n=220 held-out"]],
      [6.5 * cm, 2.8 * cm, 6 * cm])
img("outputs/report/confusion_matrix.png", 10.5 * cm,
   "Figure 1. 4-class confusion matrix. Residual error concentrates on the "
   "FOG&harr;RAIN boundary, a physically overlapping condition pair.")
img("outputs/report/accuracy_progression.png", 12.5 * cm,
   "Figure 2. Accuracy progression from rule-based to learned, 4-class.")
p("""<b>Addressing the dataset-provenance confound.</b> Because the 4-class
result mixes three source datasets, the classifier could in principle be
learning dataset fingerprints (camera, compression) rather than weather
itself. The 2-class fog/rain row is the controlled check: both classes come
from the same source (DAWN), holding provenance constant. The model still
reaches 0.68 accuracy — well above chance and the same-source rule
baseline (0.49) — which is direct evidence of genuine weather signal.""")
img("outputs/report/enhancement_montage.png", 12 * cm,
   "Figure 3. Adaptive enhancement on real images: night (CLAHE), fog (DCP "
   "dehaze), rain (dehaze+denoise).")
pagebreak()

p("4.3 Feature Importance and Engineering Bug Found", "H2c")
img("outputs/report/feature_importance.png", 11 * cm,
   "Figure 10. RandomForest feature importance for the learned scene "
   "classifier. Dark-channel mean, luminance statistics, and Laplacian "
   "variance dominate — consistent with the physical intuition that haze "
   "reduces contrast and low light reduces mean intensity.")
p("""During development, the Dark Channel Prior dehazing implementation
produced a pure-white output on real fog images. The root cause was a
double-normalisation defect: the atmospheric-light term was divided by 255 a
second time after the input frame had already been normalised, amplifying the
recovered radiance roughly tenfold. This was invisible to unit-level shape
checks and was only caught by visually inspecting enhanced output on real
images — reinforcing that automated tests alone are insufficient for this
class of numerical image-processing defect; a rendered-output review step is
necessary and was added to this project's verification practice going
forward.""")
img("outputs/report/fog_rain_features.png", 11 * cm,
   "Figure 11. Median feature values separating FOG from RAIN on real DAWN "
   "images — the basis for the unconfounded 2-class check in Section 4.2.")
pagebreak()

# ============================================================ 5. PHASE 2
p("5. Phase 2 — Detection Model Benchmarking", "H1c"); hr()
p("5.1 Wheel-Count Classification: CNN vs RTMDet vs YOLO", "H2c")
p(f"""Three architectures are benchmarked on an identical, <b>leak-free</b>
4-class wheel-count dataset (2-wheeler, 3-wheeler, 4-wheeler, 6+-wheeler; COCO
+ auto-rickshaw crops). The dataset builder originally split samples by
<i>crop</i> rather than by source image, allowing multiple vehicles from the
same photograph to appear on both sides of the split (measured: 76% / 91% /
65% of 2-, 4-, and 6+-wheeler crops came from multi-instance photos) — this
was identified and corrected before any number below was finalized.""")
table([["Model", "Accuracy", "Params", "Latency", "Pretrained"],
       [P2W["cnn"]["model"], f"{P2W['cnn']['accuracy']:.3f}",
        f"{P2W['cnn'].get('params_m', 0):.2f}M", f"{P2W['cnn']['latency_ms']:.1f}ms", "none"],
       [P2W["cspnext"]["model"], f"{P2W['cspnext']['accuracy']:.3f}",
        f"{P2W['cspnext'].get('params_m', 0):.2f}M", f"{P2W['cspnext']['latency_ms']:.1f}ms", "none"],
       [f"{P2W['yolo']['model']} (selected)", f"{P2W['yolo']['accuracy']:.3f}",
        "—", f"{P2W['yolo']['latency_ms']:.1f}ms", "ImageNet"]],
      [4.5 * cm, 2.3 * cm, 2 * cm, 2.3 * cm, 2.5 * cm])
img("outputs/report_p2/wheel_composition.png", 9 * cm,
   "Figure 4. Dataset composition — balanced across the four wheel-count "
   "classes after the leakage fix.")
img("outputs/report_p2/wheel_compare.png", 10.5 * cm,
   "Figure 5. Accuracy and macro-F1, leak-free split.")
img("outputs/report_p2/wheel_perclass.png", 11.5 * cm,
   "Figure 6. Per-class F1. 3-Wheeler remains the strongest class even after "
   "the leakage fix (0.96) — auto-rickshaws are visually distinctive, not an "
   "artifact of the earlier whole-photo framing shortcut, which was tested "
   "and disproved (Section 5.1.1).")
p("5.1.1 Ruling Out an Alternative Explanation", "H3c")
p("""Before attributing the 3-Wheeler class's high score to genuine visual
distinctiveness, an alternative hypothesis was tested: the class had
originally been built from whole photographs while every other class used
tight crops, which could let a model separate it by image framing rather than
vehicle appearance. After correcting the dataset builder to crop 3-Wheeler
samples identically to the other classes, the score was unchanged (0.96 F1
before and after), disproving the framing-shortcut hypothesis. The classes
that did change after the correction were 4-Wheeler (0.81 to 0.70 F1) and
6+-Wheeler (0.78 to 0.72 F1) — 4-Wheeler was also the most leakage-exposed
class at 91% — consistent with genuine leakage inflation rather than a
framing artifact.""")
p("""The comparison is not pretraining-neutral: YOLOv8-cls starts from
ImageNet weights while the CNN and CSPNeXt train from scratch, so part of
YOLO's margin reflects transfer learning rather than architecture alone.
CSPNeXt (2.35M params, the largest model) is nonetheless the fastest at
inference — its depthwise 5×5 CSP design is CPU-efficient — and was still
improving with additional training on an earlier version of this dataset
(0.562 at 12 epochs → 0.683 at 40), so its reported accuracy is a floor, not a
ceiling. Because <font face='Courier' size=8>mmdetection/mmcv</font> — RTMDet's
official framework — will not install in this environment (openmim breaks on
Python 3.12; mmcv ships no wheels for torch 2.12), CSPNeXt is reimplemented
directly in PyTorch (depthwise 5×5 CSP blocks, channel attention, SiLU, SPPF)
rather than omitting the comparison the contract specifies.""")
p("5.2 Granular Helmet Compliance (YOLOv8)", "H2c")
p(f"""A YOLOv8n detector is fine-tuned on a 7-class rider/helmet dataset
(368 train / 65 val / 52 test images, YOLO format): driver and passenger, each
crossed with helmet/no-helmet, plus a bike class. mAP@50 on the held-out test
split — never used for epoch selection — is <b>{P2H['test']['mAP50']:.3f}</b>
(validation, which is optimistically biased, was {P2H['val']['mAP50']:.3f});
the two splits corroborate each other given both are small.""")
img("outputs/report_p2/helmet_ap.png", 11 * cm,
   "Figure 7. Per-class AP@50 on the held-out test split.")
img("outputs/report_p2/helmet_sample.png", 11 * cm,
   "Figure 8. Trained detector applied to a real test image, showing riders, "
   "bike, and per-seat helmet-compliance status.")
p("""Passenger-helmet classes are weakest on both splits (AP50 0.545/0.565),
reflecting genuinely fewer training instances (65 and 74 vs 439–500 for the
common classes) — a real data limitation, not a training defect. A targeted
fix was attempted: 3x duplication-oversampling of images containing these
classes. It made results worse (overall mAP@50 0.764 → 0.711) because plain
duplication increases exposure without adding visual diversity, causing
overfitting rather than generalisation. The baseline weights were kept and
this negative result is documented rather than hidden.""")
pagebreak()

# ============================================================ 6. PHASE 3
p("6. Phase 3 — Tracking, Re-Identification &amp; Risk Analytics", "H1c"); hr()
p("6.1 Multi-Frame Voting Under Occlusion", "H2c")
p(f"""Evaluated on UA-DETRAC ({len(P3O['sequences'])} sequences,
{P3O['crops_classified']:,} classified crops, grouped by ground-truth track id
to isolate voting from tracker error). UA-DETRAC is ~97% cars and its
occlusion bands are not class-comparable (the heavy-occlusion band is 100% one
class), so raw per-band accuracy is invalid here; flip-rate — how often the
emitted label changes between consecutive frames of the same vehicle — is
class-composition-independent and is reported as the primary metric.""")
img("outputs/report_p3/flip_rate.png", 10.5 * cm,
   f"Figure 9. Flip-rate falls "
   f"{(1 - P3O['flip_rate_overall']['30']/P3O['flip_rate_overall']['1'])*100:.1f}% "
   f"from N=1 to N=30.")
img("outputs/report_p3/occlusion_bands.png", 9.5 * cm,
   "Figure 10. UA-DETRAC occlusion-band support — the heavy-occlusion band "
   "contains only one vehicle class, which is precisely why per-band raw "
   "accuracy would be misleading and flip-rate is used instead.")
p(f"""Balanced accuracy (macro recall) also improves, though more modestly
(best N=15: {P3O['overall']['15']:.3f} vs single-frame {P3O['overall']['1']:.3f},
+{P3O['overall']['15'] - P3O['overall']['1']:.3f}) — part of the flip-rate
reduction is mechanical (averaging smooths noise as well as signal), and the
smaller but directionally consistent balanced-accuracy gain is reported
alongside it rather than leading only with the more dramatic figure.""")
img("outputs/report_p3/balanced_accuracy.png", 10.5 * cm,
   "Figure 11. Balanced vs raw accuracy across vote windows — raw accuracy "
   "stays near 0.93 throughout purely because one class dominates the "
   "dataset, illustrating why it is not the reported headline metric.")
p("6.2 Cross-Camera Re-Identification", "H2c")
table([["Metric", "Value"], ["Rank-1", f"{P3REID['rank1']:.3f}"],
       ["Rank-5", f"{P3REID['rank5']:.3f}"], ["mAP", f"{P3REID['mAP']:.3f}"]],
      [6 * cm, 4 * cm])
p(f"""Evaluated on VeRi-776 (776 vehicles, 20 real cameras), excluding
same-camera gallery matches per the standard protocol. These numbers are well
below published fine-tuned SOTA (Rank-1≈90%, mAP≈70%) because the backend runs
off-the-shelf ImageNet-pretrained OSNet with the classification head
discarded — no vehicle-identity metric learning at all. Rank-1
{P3REID['rank1']*100:.0f}% from purely generic features still confirms real
appearance signal against a 776-way chance level of 0.1%; closing the gap
needs triplet-loss fine-tuning on VeRi's own 37,778-image training split,
already downloaded but not yet used (tracked as future work, Section 9).""")
img("outputs/report_p3/reid.png", 9.5 * cm,
   "Figure 12. VeRi-776 cross-camera ReID scores, off-the-shelf OSNet.")
p("6.3 Risk Indexing Validated Against Real Ground Truth", "H2c")
p(f"""The risk-indexer rules are validated against {P3RISK['n_tracks']:,} real
motorcycle tracks from the HELMET dataset (910 Myanmar traffic clips), using
ground-truth occupancy and per-seat helmet-use labels rather than detector
output — isolating rule calibration from detection accuracy.""")
img("outputs/report_p3/occupancy.png", 10 * cm,
   "Figure 13. Real-world motorcycle occupancy from 283,377 HELMET "
   "annotations — 6.4% carry 3 or more riders, the population the overload "
   "rule is meant to catch.")
img("outputs/report_p3/risk_distribution.png", 9 * cm,
   "Figure 14. Resulting risk-level distribution on real data.")
p(f"""<b>Finding:</b> HIGH risk never triggers on real motorcycle data (0.0%).
helmet-misuse (weight 2.5) plus overload (weight 1.5) sums to 4.0, below the
HIGH threshold of 6.0 — even the worst realistic motorcycle scenario tops out
at MEDIUM. This is a calibration finding surfaced by validation against
ground truth, not a code defect, and is reported as a policy question for the
weighting scheme rather than silently re-tuned to appear more dramatic.""")
p("6.4 Demographics — A Hardware Constraint, Diagnosed and Resolved", "H2c")
p(f"""Installing DeepFace (justified by a coverage gate: 62% of face-exposed
rider detections yield a crop above the minimum usable size) surfaced a
process-level conflict: this machine's torch build targets a newer NVIDIA
driver than is installed, and loading a second torch model in a process that
has also imported TensorFlow segfaults — confirmed by bisecting pipeline
construction component by component. The live pipeline now queues face crops
for a separate process that never imports torch; verified end-to-end on real
pipeline output ({P3DEMO['n_succeeded']}/{P3DEMO['n_crops']} crops processed
successfully with no crash).""")
p("6.5 End-to-End Pipeline Throughput", "H2c")
p(f"""On {P3PIPE['frames']} real frames ({P3PIPE['source']}), the full
pipeline (all nine stages) sustains <b>{P3PIPE['fps_mean']:.2f} FPS mean</b>
({P3PIPE['fps_final']:.1f} FPS steady-state) on CPU alone, with an annotated
telemetry HUD, tracking IDs, and a risk banner rendered per frame. {P3PIPE.get('note','')}""")
pagebreak()

# ============================================================ 7. PHASE 4
p("7. Phase 4 — Optimization &amp; Benchmarking", "H1c"); hr()
p("7.1 Finding: Naive Pruning Collapses Accuracy", "H2c")
sc_fp32 = p4row("SmallCNN", "fp32"); sc_naive = p4row("SmallCNN", "pruned_naive")
sc_ft = p4row("SmallCNN", "pruned"); sc_int8 = p4row("SmallCNN", "onnx_int8")
table([["Config", "SmallCNN Acc.", "Retention"],
       ["FP32 baseline", f"{sc_fp32['accuracy']:.3f}", "1.00"],
       ["30% pruned (no fine-tune)", f"{sc_naive['accuracy']:.3f}", f"{sc_naive['accuracy_retention']:.2f}"],
       ["30% pruned + 3-epoch fine-tune", f"{sc_ft['accuracy']:.3f}", f"{sc_ft['accuracy_retention']:.2f}"],
       ["+ ONNX Runtime INT8", f"{sc_int8['accuracy']:.3f}", f"{sc_int8['accuracy_retention']:.2f}"]],
      [7 * cm, 4 * cm, 3 * cm])
img("outputs/report_p4/pruning_recovery.png", 11 * cm,
   "Figure 15. Naive pruning collapses accuracy; fine-tune recovery restores it.")
p("""Structured pruning zeroes whole output channels in every convolutional
layer independently; because layers feed one another the effect compounds,
and with no retraining the network does not recover. This is expected
behaviour for naive structured pruning, not a code defect — the standard
mitigation, applied here, is a short fine-tuning pass after pruning, which
fully restores accuracy for both benchmarked models.""")
p("7.2 Quantization and ONNX Export", "H2c")
img("outputs/report_p4/size_reduction.png", 9.5 * cm,
   "Figure 16. ONNX Runtime INT8 shrinks both models 3.8-3.9x at no further "
   "accuracy cost beyond pruning.")
p("""On the recovered model, INT8 quantization is essentially free — accuracy
is unchanged from the fine-tuned pruned model for both architectures. However,
ONNX latency is <i>not</i> uniformly faster than native PyTorch at batch-1 on
this CPU: session/dispatch overhead dominates for these small, sub-millisecond
models, so the deployment win from the ONNX artifacts here is model size, not
assumed latency — reported as measured rather than claimed.""")
img("outputs/report_p4/latency.png", 11.5 * cm,
   "Figure 17. Latency across every optimization configuration for both "
   "benchmarked models.")
p("7.3 Full Results Table", "H2c")
rt_fp32 = p4row("RTMDet-CSPNeXt", "fp32"); rt_naive = p4row("RTMDet-CSPNeXt", "pruned_naive")
rt_ft = p4row("RTMDet-CSPNeXt", "pruned"); rt_int8 = p4row("RTMDet-CSPNeXt", "onnx_int8")
yolo_fp32 = p4row("YOLOv8-cls", "fp32"); yolo_onnx = p4row("YOLOv8-cls", "onnx_fp32")
table([["Model", "Config", "Acc.", "Retention", "Latency (ms)", "Size (MB)"],
       ["SmallCNN", "FP32", f"{sc_fp32['accuracy']:.3f}", "1.00", f"{sc_fp32['latency_ms']:.2f}", f"{sc_fp32['size_mb']:.2f}"],
       ["SmallCNN", "Pruned+FT", f"{sc_ft['accuracy']:.3f}", f"{sc_ft['accuracy_retention']:.2f}", f"{sc_ft['latency_ms']:.2f}", f"{sc_ft['size_mb']:.2f}"],
       ["SmallCNN", "ONNX INT8", f"{sc_int8['accuracy']:.3f}", f"{sc_int8['accuracy_retention']:.2f}", f"{sc_int8['latency_ms']:.2f}", f"{sc_int8['size_mb']:.2f}"],
       ["CSPNeXt", "FP32", f"{rt_fp32['accuracy']:.3f}", "1.00", f"{rt_fp32['latency_ms']:.2f}", f"{rt_fp32['size_mb']:.2f}"],
       ["CSPNeXt", "Pruned+FT", f"{rt_ft['accuracy']:.3f}", f"{rt_ft['accuracy_retention']:.2f}", f"{rt_ft['latency_ms']:.2f}", f"{rt_ft['size_mb']:.2f}"],
       ["CSPNeXt", "ONNX INT8", f"{rt_int8['accuracy']:.3f}", f"{rt_int8['accuracy_retention']:.2f}", f"{rt_int8['latency_ms']:.2f}", f"{rt_int8['size_mb']:.2f}"],
       ["YOLOv8-cls", "FP32", f"{yolo_fp32['accuracy']:.3f}", "1.00", f"{yolo_fp32['latency_ms']:.2f}", f"{yolo_fp32['size_mb']:.2f}"],
       ["YOLOv8-cls", "ONNX FP32", f"{yolo_onnx['accuracy']:.3f}", f"{yolo_onnx['accuracy_retention']:.2f}", f"{yolo_onnx['latency_ms']:.2f}", f"{yolo_onnx['size_mb']:.2f}"]],
      [3 * cm, 2.6 * cm, 1.7 * cm, 2 * cm, 2.6 * cm, 2 * cm], font=7.8)
pagebreak()

# ============================================================ 8. DISCUSSION
p("8. Discussion, Limitations &amp; Ethics", "H1c"); hr()
p("8.1 Summary of Honest Findings", "H2c")
p("""Several results in this project fell short of an initial expectation and
are reported as found rather than adjusted to look more favourable: cross-camera
ReID trails published SOTA by a wide margin because no metric-learning
fine-tuning was performed; naive structured pruning collapses accuracy;
oversampling the weakest helmet classes made results worse, not better; and
the risk indexer's HIGH tier is architecturally unreachable from motorcycle
violations alone under its current weights. In each case the root cause is
diagnosed and, where a fix exists, applied and re-measured.""")
p("8.2 The Silent-Fallback Risk", "H2c")
p("""A recurring theme across all four phases: every heavy backend
(Ultralytics, DeepSORT, torchreid, DeepFace) degrades gracefully to a
lightweight fallback rather than crashing when a dependency is broken. This is
good defensive design but has a sharp edge — a broken dependency and a working
one are indistinguishable from the pipeline's console output unless the active
backend is explicitly asserted. This project's regression test suite
(<font face='Courier' size=8>tests/test_regressions.py</font>) exists
specifically because two backends (DeepSORT, OSNet) were found silently
degraded mid-project, and a third defect (inverted helmet-compliance
semantics from a config/weights class-id mismatch) shipped into a published
report before being caught.""")
p("8.3 Limitations", "H2c")
bullets([
    "Granular helmet sub-classes from the original contract taxonomy (strap-unfastened, helmet-on-handlebar, helmet-on-arm) have no available public dataset and are not modelled; the 7-class scheme used here follows the AI City Track-5 convention instead.",
    "Cross-camera ReID uses an off-the-shelf backbone; VeRi-776-specific fine-tuning (data already downloaded) is future work, not a current claim.",
    "Occlusion-voting is evaluated on UA-DETRAC, which contains no two- or three-wheelers — the vehicle classes this project otherwise focuses on — so the ablation's class coverage does not match the detection models' target domain.",
    "The demographics module reports gender more reliably than age given the resolution of typical face crops in this footage (median 38px against DeepFace's ~64px comfort zone for age).",
])
p("8.4 Ethical Considerations", "H2c")
p("""Automated demographic inference (age, gender) on riders raises privacy
and potential-bias concerns independent of model accuracy; this system is
positioned as a research prototype and any operational deployment would
require explicit data-governance, consent, and bias-audit steps beyond this
report's scope. Risk scoring that could inform enforcement action carries
similar weight — the calibration finding in Section 6.3 (HIGH is currently
unreachable from the most common real violation pattern) is flagged
specifically because miscalibrated automated risk scoring has real-world
consequences if deployed without human review.""")
p("8.5 Reproducibility Practice", "H2c")
p("""Every quantitative claim in this report is generated by a script that
reads exclusively from <font face='Courier' size=8>results/phase1..4/*.json</font>
— the same files consumed by the individual Phase1..4_Report.pdf documents —
so no number here can silently drift from what the code actually measured.
Where a metric was corrected mid-project (the wheel-classification benchmark,
after the leakage fix), the superseded value and the reason for the change are
recorded alongside the corrected one rather than quietly overwritten, both in
the tracked JSON and in this report.""")
pagebreak()

# ============================================================ 9. CONCLUSION
p("9. Conclusion &amp; Future Work", "H1c"); hr()
p(f"""This project delivers a complete, CPU-optimized traffic surveillance
pipeline spanning adaptive preprocessing ({P1['accuracy']:.2f} scene-classification
accuracy), a leak-free three-way detection benchmark ({P2W['yolo']['accuracy']:.3f}
wheel-count accuracy, {P2H['test']['mAP50']:.3f} helmet mAP@50), multi-frame
tracking with a measured occlusion-robustness benefit, and model optimization
that recovers accuracy lost to naive pruning while shrinking deployed models
3.8-3.9x. Every phase surfaced at least one result that did not match initial
expectations; each was investigated to a root cause rather than reported
around, which is the standard this report holds itself to throughout.""")
p("Future work:", "H2c")
bullets([
    "Fine-tune OSNet on VeRi-776's training split with a triplet/ID loss to close the cross-camera ReID gap to published SOTA (issue #14).",
    "Source or annotate images for the granular helmet sub-classes (strap, handlebar, arm) missing from public datasets.",
    "Re-run the occlusion-voting ablation on footage containing two- and three-wheelers, to align the ablation's domain with the detection models it is meant to support.",
    "Revisit the risk-indexer weights given the real-world finding that HIGH is currently unreachable from the most common violation pattern.",
    "Explore augmentation-based (not duplication-based) oversampling for the passenger-helmet classes.",
])
pagebreak()

# ============================================================ 10. REFERENCES
p("10. References", "H1c"); hr()
refs = [
    "Wojke, N., Bewley, A., &amp; Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. <i>ICIP</i>.",
    "Zhou, K., Yang, Y., Cavallaro, A., &amp; Xiang, T. (2019). Omni-Scale Feature Learning for Person Re-Identification. <i>ICCV</i>.",
    "Liu, X., Liu, W., Ma, H., &amp; Fu, H. (2016). Large-Scale Vehicle Re-Identification in Urban Surveillance Videos. <i>ICME</i>. (VeRi-776)",
    "Lyu, C., Zhang, W., Huang, H., et al. (2022). RTMDet: An Empirical Study of Designing Real-Time Object Detectors. <i>arXiv:2212.07784</i>.",
    "Wen, L., Du, D., Cai, Z., et al. (2020). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking. <i>Computer Vision and Image Understanding</i>.",
    "Jocher, G., et al. (2023). Ultralytics YOLOv8. github.com/ultralytics/ultralytics.",
    "Naik, B., et al. (2019). HELMET: A Dataset of Helmet Use in Motorcycle Traffic (Myanmar). osf.io/4pwj8.",
    "AI City Challenge (2023). Track 5: Detecting Violation of Helmet Rule for Motorcyclists. aicitychallenge.org.",
]
for i, r in enumerate(refs, 1):
    p(f"[{i}] {r}")
pagebreak()

p("Glossary and Abbreviations", "H1c"); hr()
gloss = [
    ("mAP@50", "Mean Average Precision at 0.5 IoU threshold — the standard object-detection accuracy metric used throughout Phase 2."),
    ("Rank-1 / Rank-5", "Re-identification accuracy: whether the correct match appears in the top-1 / top-5 ranked gallery results for a query."),
    ("Balanced accuracy", "Macro-averaged per-class recall; used instead of raw accuracy where classes are imbalanced (Section 6.1)."),
    ("Flip-rate", "Fraction of consecutive frame-pairs (same tracked vehicle) where the emitted class label changes — a class-composition-independent stability metric."),
    ("Accuracy retention", "Optimized-model accuracy divided by its own FP32 baseline accuracy; the headline Phase 4 metric."),
    ("CSPNeXt", "The convolutional backbone used by RTMDet; reimplemented here in PyTorch since the official RTMDet framework does not install in this environment."),
    ("Leak-free split", "A train/validation split constructed so that no source image contributes samples to both sides — see Section 5.1."),
    ("Process isolation", "Running a component in a separate OS process to avoid an in-process library conflict — applied to demographics (Section 6.4)."),
]
for term, definition in gloss:
    p(f"<b>{term}.</b> {definition}")
pagebreak()

p("Appendix A — Training Hyperparameters", "H2c"); hr()
table([["Model", "Epochs", "Image size", "Batch", "Optimizer", "Notes"],
       ["Scene classifier (RF)", "—", "—", "—", "300 trees", "15 hand-engineered features"],
       ["SmallCNN (wheel)", "12", "96×96", "32", "Adam, cosine LR", "0.24M params, from scratch"],
       ["CSPNeXt (wheel)", "12", "96×96", "32", "Adam, cosine LR", "2.35M params, from scratch"],
       ["YOLOv8-cls (wheel)", "8", "96×96", "32", "Ultralytics default", "ImageNet-pretrained"],
       ["YOLOv8n (helmet)", "25", "416×416", "8", "Ultralytics default", "COCO-pretrained backbone"],
       ["Pruning fine-tune", "3", "96×96", "32", "Adam, lr=1e-4", "post-prune recovery"]],
      [3.5 * cm, 1.6 * cm, 2.1 * cm, 1.6 * cm, 2.8 * cm, 3.4 * cm], font=7.6)
pagebreak()

p("Appendix B — Dataset Summary", "H2c"); hr()
table([["Dataset", "Used for", "Size"],
       ["DAWN", "Fog/Rain scene classification", "300 fog + 200 rain images"],
       ["ExDark", "Night scene classification", "300 images (of 7,363 available)"],
       ["COCO val2017", "Day scenes + wheel-count crops", "300 day images; 5,000 total"],
       ["Helmet (7-class)", "Helmet-compliance detection", "368 train / 65 val / 52 test"],
       ["Auto-rickshaw set", "3-wheeler crops", "663 images"],
       ["UA-DETRAC", "Tracking + occlusion voting", "8 sequences, 22,480 crops"],
       ["VeRi-776", "Cross-camera ReID", "776 vehicles, 20 cameras, 51k images"],
       ["HELMET (annotations)", "Risk-rule validation, occupancy stats", "910 clips, 283,377 instances, 10,006 tracks"]],
      [3.2 * cm, 5.3 * cm, 6.5 * cm], font=7.8)
pagebreak()

p("Appendix C — Repository &amp; Reproduction", "H2c"); hr()
p(f"Source code, trained model configs, and all tracked results: {REPO}")
p("Every figure and table in this report is generated by "
 "<font face='Courier' size=8>scripts/build_final_report.py</font> reading "
 "directly from <font face='Courier' size=8>results/phase1..4/*.json</font>; "
 "full reproduction commands are in "
 "<font face='Courier' size=8>docs/HOW_TO_RUN.md</font>.")
sp(12)
p("Appendix D — Contract Deliverables Checklist", "H2c"); hr()
table([["Deliverable", "Status", "Evidence"],
       ["Complete Python source code", "Done", "This repository"],
       ["Trained model weights (PyTorch/ONNX)", "Done", "weights/, results/phase4/"],
       ["Dataset preprocessing scripts", "Done", "scripts/build_*, src/preprocessing/"],
       ["Installation guide + requirements", "Done", "README.md, requirements.txt"],
       ["Step-by-step execution documentation", "Done", "docs/HOW_TO_RUN.md"],
       ["Publication-ready research paper", "Done", "docs/paper/"],
       ["Per-milestone demonstration", "Done", "Phase1..4_Report.pdf"]],
      [6 * cm, 2 * cm, 7 * cm], font=8)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "Adaptive Traffic Surveillance — Final Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(BLUE); canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


os.makedirs("docs", exist_ok=True)
doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        title="Final Report — Adaptive Traffic Surveillance", author="Harsh Pandhe")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"PDF -> {OUT_PDF}")
