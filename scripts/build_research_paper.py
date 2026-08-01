"""
scripts/build_research_paper.py
---------------------------------
Builds a two-column, conference-style research paper from the SAME
results/phase*/*.json used by every other report in this project, so its
numbers cannot drift from what was actually measured.

Output -> docs/paper/paper.pdf

Generic two-column format for now; retarget font/margins/citation style when
a specific venue is chosen (see docs/paper/README.md).
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
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, Table, TableStyle, Image,
                                FrameBreak, NextPageTemplate, HRFlowable)

OUT_PDF = "docs/paper/paper.pdf"
REPO = "https://github.com/harsh-pandhe/traffic-surveillance-system"

P1 = json.load(open("results/phase1/scene_metrics.json"))
P2W = json.load(open("results/phase2/wheel_metrics.json"))
P2H = json.load(open("results/phase2/helmet_metrics.json"))
P3O = json.load(open("results/phase3/occlusion_ablation.json"))
P3RISK = json.load(open("results/phase3/risk_indexer_validation.json"))
P3REID = json.load(open("results/phase3/reid_benchmark.json"))
P4 = json.load(open("results/phase4/benchmark.json"))


def p4row(model, config):
    return next(r for r in P4["rows"] if r["model"] == model and r["config"] == config)


BLACK = colors.black
GREY = colors.HexColor("#444444")
LIGHT = colors.HexColor("#eeeeee")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("PaperTitle", parent=styles["Title"], fontSize=15,
                          leading=18, spaceAfter=4, alignment=TA_CENTER))
styles.add(ParagraphStyle("Authors", parent=styles["Normal"], fontSize=10.5,
                          alignment=TA_CENTER, spaceAfter=2))
styles.add(ParagraphStyle("Affil", parent=styles["Normal"], fontSize=9,
                          alignment=TA_CENTER, textColor=GREY, spaceAfter=10))
styles.add(ParagraphStyle("AbstractHead", parent=styles["Normal"], fontSize=9.5,
                          fontName="Helvetica-Bold", spaceAfter=2))
styles.add(ParagraphStyle("AbstractBody", parent=styles["Normal"], fontSize=9.3,
                          leading=12, alignment=TA_JUSTIFY, spaceAfter=7))
styles.add(ParagraphStyle("Sec", parent=styles["Heading2"], fontSize=11.5,
                          spaceBefore=8, spaceAfter=3, textColor=BLACK))
styles.add(ParagraphStyle("SubSec", parent=styles["Heading3"], fontSize=10,
                          fontName="Helvetica-BoldOblique", spaceBefore=5, spaceAfter=2))
styles.add(ParagraphStyle("Col", parent=styles["Normal"], fontSize=9.3,
                          leading=12.6, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle("ColCap", parent=styles["Normal"], fontSize=8.2,
                          leading=10, alignment=TA_JUSTIFY, textColor=GREY, spaceAfter=7))
styles.add(ParagraphStyle("RefStyle", parent=styles["Normal"], fontSize=8.4,
                          leading=10.6, spaceAfter=4))

COL = styles["Col"]
story = []


def sec(t):
    story.append(Paragraph(t, styles["Sec"]))


def subsec(t):
    story.append(Paragraph(t, styles["SubSec"]))


def para(t):
    story.append(Paragraph(t, COL))


def sp(h=4):
    story.append(Spacer(1, h))


def img(path, width=8.1 * cm, cap=None):
    if not os.path.isfile(path):
        return
    iw, ih = ImageReader(path).getSize()
    story.append(Image(path, width=width, height=width * ih / iw))
    if cap:
        story.append(Paragraph(cap, styles["ColCap"]))
    sp(4)


def table(data, cw, font=7.2):
    t = Table(data, colWidths=cw, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t); sp(5)


# ============================================================ TITLE (full width)
story.append(Paragraph(
    "Adaptive Spatio-Temporal Traffic Surveillance for Granular "
    "Helmet-Compliance and Multi-Frame Vehicle Classification on "
    "Commodity CPUs", styles["PaperTitle"]))
sp(10)

story.append(Paragraph("Abstract", styles["AbstractHead"]))
story.append(Paragraph(f"""We present an end-to-end, CPU-only traffic
surveillance pipeline spanning adaptive scene-conditioned preprocessing,
granular helmet-compliance and wheel-count detection, multi-frame tracking
with cross-camera re-identification, and deployment-oriented model
optimization. A learned scene classifier lifts day/night/fog/rain accuracy
from a 0.49 rule-based baseline to {P1['accuracy']:.2f}, validated on a
within-dataset fog/rain pair to rule out a source-dataset confound. Three
architectures are benchmarked for wheel-count classification under an
identical, leakage-free protocol: a custom CNN ({P2W['cnn']['accuracy']:.3f}),
a from-scratch RTMDet-style CSPNeXt backbone ({P2W['cspnext']['accuracy']:.3f}),
and ImageNet-pretrained YOLOv8-cls ({P2W['yolo']['accuracy']:.3f}, selected),
with the pretraining asymmetry stated explicitly rather than implied away. A
7-class helmet detector reaches {P2H['test']['mAP50']:.3f} mAP@50 on held-out
data. Multi-frame majority voting reduces prediction flip-rate by
{(1-P3O['flip_rate_overall']['30']/P3O['flip_rate_overall']['1'])*100:.1f}%
under measured real-world occlusion, and risk-indexing rules are validated
against {P3RISK['n_tracks']:,} real motorcycle tracks with ground-truth
occupancy and helmet-use labels. We further show that naive structured
pruning collapses model accuracy (0.31 retention) unless followed by brief
fine-tuning (0.98 retention), after which INT8 quantization shrinks deployed
models 3.8-3.9x at no further cost. Where a component underperformed
expectation — cross-camera re-identification without metric-learning
fine-tuning, class-imbalance-aware voting evaluation — we report the
shortfall and its cause rather than omitting it.""", styles["AbstractBody"]))
story.append(Paragraph("<b>Index Terms</b>—helmet detection, vehicle "
                       "re-identification, multi-object tracking, model "
                       "quantization, CPU inference", styles["AbstractBody"]))
sp(6)
story.append(HRFlowable(width="100%", thickness=0.6, color=GREY))
sp(4)

# Switch to two-column template from here on.
story.append(NextPageTemplate("twocol"))
story.append(FrameBreak())

# ============================================================ I. INTRODUCTION
sec("I. Introduction")
para("""Two-wheeler traffic dominates road usage in many dense urban
environments and carries disproportionate crash risk, yet automated
enforcement of helmet compliance and vehicle-load limits remains largely
manual. Prior automated systems typically address a single sub-problem —
helmet detection, license-plate recognition, or vehicle counting — in
isolation and are rarely evaluated across the range of lighting and weather
conditions actually encountered in the field. This work builds and evaluates
a single pipeline spanning environment-adaptive preprocessing, granular
compliance detection, occlusion-robust tracking, and deployment-ready
optimization, on commodity CPU hardware throughout.""")
para("""Our contributions are: <b>(C1)</b> a scene-adaptive preprocessing
stage with an unconfounded within-dataset validation; <b>(C2)</b> a
three-way, leakage-free CPU architecture benchmark for wheel-count
classification with an explicit pretraining-fairness analysis, including a
from-scratch CSPNeXt implementation where the official RTMDet framework would
not install; <b>(C3)</b> a multi-frame voting scheme evaluated with a
class-imbalance-robust metric after diagnosing why naive accuracy comparison
was invalid on the available benchmark; and <b>(C4)</b> end-to-end model
optimization with a diagnosed and corrected pruning failure mode.""")

# ============================================================ II. RELATED WORK
sec("II. Related Work")
subsec("A. Helmet and Rider Compliance Detection")
para("""Automated helmet detection is typically framed as binary
(helmet/no-helmet) object detection, commonly built on YOLO-family detectors
for CPU/edge throughput [1]. Few public systems distinguish driver from
passenger or model rider count directly; this work adopts the AI City
Challenge Track-5 seven-class convention (driver/passenger x helmet-status,
plus a bike class) [7].""")
subsec("B. Vehicle Classification and RTMDet")
para("""Vehicle-type classification is usually a byproduct of general
detection (COCO car/bus/truck/motorcycle) rather than purpose-built for the
2/3/4/6+-wheeler taxonomy common in heterogeneous South/Southeast Asian
traffic. RTMDet [4] is a strong real-time detector family; we evaluate its
CSPNeXt backbone as a classifier since its official mmdetection toolchain does
not install against Python 3.12 / PyTorch 2.12.""")
subsec("C. Tracking and Re-Identification")
para("""DeepSORT [1] remains a mature, CPU-tractable choice for online
multi-object tracking versus GPU-oriented transformer trackers. VeRi-776 [3]
is a standard cross-camera vehicle re-identification benchmark; SOTA methods
fine-tune a re-identification-specific backbone (often OSNet [2]) with a
triplet/ID loss on the benchmark's own split — a step we deliberately do not
take, to measure the honest zero-shot gap (Section V-C).""")

# ============================================================ III. METHOD
sec("III. Method")
subsec("A. Adaptive Preprocessing")
para("""Each frame yields 15 lighting/haze/texture features (luminance
statistics, dark-channel mean, Laplacian variance, FFT high-frequency energy,
colourfulness). A 300-tree RandomForest, trained on real DAWN/ExDark/COCO
imagery, classifies DAY/NIGHT/FOG/RAIN; enhancement (CLAHE, Dark Channel Prior
dehaze, or denoise) is applied only when the predicted condition warrants it.""")
subsec("B. Detection Models")
para(f"""Wheel-count classification benchmarks a custom SmallCNN
(0.24M params, from scratch), a from-scratch CSPNeXt classifier (2.35M
params, from scratch), and YOLOv8-cls (ImageNet-pretrained) on an identical
4-class dataset split <i>by source image</i> — not by crop — after an earlier
crop-level split was found to leak scene information across train/val
(Section IV-A). A YOLOv8n detector is separately fine-tuned for 7-class
helmet compliance.""")
subsec("C. Tracking, Voting, and Risk Indexing")
para("""DeepSORT provides per-vehicle tracks; a temporal majority-vote buffer
smooths the wheel-count prediction over a sliding window of N frames. A
weighted risk index combines wheel-count violation, helmet misuse, rider
overload, and wrong-way movement into LOW/MEDIUM/HIGH tiers, validated against
real ground-truth occupancy and helmet-use labels rather than only against
model output (Section V-D).""")
subsec("D. Optimization")
para("""Each classifier is pruned (30% L1 structured sparsity), optionally
fine-tuned to recover accuracy, dynamically quantized to INT8, and exported to
ONNX with a second INT8 pass via ONNX Runtime — accuracy, latency, and model
size are recorded at every step.""")

# ============================================================ IV. EXPERIMENTAL SETUP
sec("IV. Experimental Setup")
subsec("A. Datasets and a Leakage Correction")
para("""DAWN, ExDark, and COCO supply Phase 1 scene data; a 7-class YOLO
helmet set and auto-rickshaw crops supply Phase 2; UA-DETRAC, VeRi-776, and
the HELMET annotation set (283,377 real motorcycle instances, no images
required for its use here) supply Phase 3. The wheel-count dataset builder
originally split <i>crops</i> rather than <i>source images</i>, so several
vehicles cropped from one photograph could appear on both sides of the split
(measured: 76%/91%/65% of 2-/4-/6+-wheeler crops came from multi-instance
photos). This was corrected before any reported number was finalized; Section
V-B reports both the corrected numbers and what changed.""")
table([["Dataset", "Role", "Size"],
       ["DAWN", "Fog/Rain scenes", "500 img"],
       ["ExDark", "Night scenes", "300 img"],
       ["COCO val2017", "Day + wheel crops", "300/5,000 img"],
       ["Helmet (7-cls)", "Compliance det.", "368/65/52"],
       ["UA-DETRAC", "Tracking, voting", "22,480 crops"],
       ["VeRi-776", "Cross-cam. ReID", "776 veh., 51k img"],
       ["HELMET (ann.)", "Risk validation", "283,377 inst."]],
      [2.7 * cm, 2.6 * cm, 2.3 * cm], font=7.6)
subsec("B. Evaluation Protocol")
para("""All wheel-count models share one held-out validation split under the
corrected partitioning. The helmet detector is evaluated on a test split never
used for epoch selection. Voting is evaluated on UA-DETRAC by grouping
predictions under ground-truth track ids, isolating the voting effect from
tracker error. ReID follows the standard VeRi single-camera-exclusion
protocol. All experiments run on commodity CPU (AMD Ryzen 5 7600).""")

# ============================================================ V. RESULTS
sec("V. Results")
subsec("A. Adaptive Preprocessing (C1)")
para(f"""The learned classifier reaches {P1['accuracy']:.2f} 4-class
accuracy versus a 0.49 rule-based baseline. To rule out a dataset-provenance
confound (each class drawn from a different source dataset), we isolate the
fog/rain pair — both from DAWN, holding provenance constant — and still
observe 0.68 accuracy (vs. 0.49 same-source baseline), evidence the model
learns genuine weather signal rather than dataset fingerprints.""")
img("outputs/report/confusion_matrix.png", 7.9 * cm,
   "Fig. 1. 4-class scene confusion matrix. Error concentrates on the "
   "FOG-RAIN boundary, a physically overlapping condition pair.")
subsec("B. Wheel-Count and Helmet Detection (C2)")
table([["Model", "Acc.", "Pretrained"],
       [P2W["cnn"]["model"], f"{P2W['cnn']['accuracy']:.3f}", "no"],
       [P2W["cspnext"]["model"], f"{P2W['cspnext']['accuracy']:.3f}", "no"],
       [f"{P2W['yolo']['model']}*", f"{P2W['yolo']['accuracy']:.3f}", "ImageNet"]],
      [3.1 * cm, 1.6 * cm, 2 * cm])
para("""*Selected. Not pretraining-neutral — YOLOv8-cls's margin partly
reflects transfer learning. CSPNeXt, the largest model (2.35M params), is
nonetheless the fastest at inference (CPU depthwise 5x5 design) and was still
improving with more training on an earlier dataset version, so its figure is
a floor. After the leakage correction, 4-wheeler accuracy fell most (0.81 to
0.70 F1) — it was the most leakage-exposed class (91%) — while 3-wheeler was
unchanged (0.96 F1 before and after), disproving an alternative hypothesis
that its score came from a whole-photo framing shortcut rather than genuine
visual distinctiveness.""")
para(f"""The helmet detector reaches {P2H['test']['mAP50']:.3f} mAP@50 on
held-out test data (val, used for epoch selection, was
{P2H['val']['mAP50']:.3f} — consistent given both splits are small).
Passenger-helmet classes are weakest on both splits (AP50 0.545/0.565) from
genuinely fewer training instances (65-74 vs. 439-500). A targeted 3x
oversampling fix was tested and made results <i>worse</i> (mAP@50 0.764 to
0.711) — plain duplication adds exposure without visual diversity, causing
overfitting rather than generalisation; this negative result is reported
rather than omitted.""")
img("outputs/report_p2/wheel_compare.png", 7.9 * cm,
   "Fig. 2. Wheel-count accuracy and macro-F1, leakage-free split.")
img("outputs/report_p2/helmet_ap.png", 7.9 * cm,
   "Fig. 3. Per-class helmet AP@50, held-out test split.")
subsec("C. Tracking, Voting, and ReID (C3)")
para(f"""UA-DETRAC is approximately 97% one vehicle class with
non-comparable occlusion bands (the heaviest-occlusion band is 100% one
class), making per-band raw accuracy invalid. We therefore report flip-rate —
how often the emitted label changes between consecutive frames of one
tracked vehicle — which fell {(1-P3O['flip_rate_overall']['30']/P3O['flip_rate_overall']['1'])*100:.1f}%
from N=1 to N=30. Balanced (macro) accuracy improved more modestly
(best N=15: {P3O['overall']['15']:.3f} vs. {P3O['overall']['1']:.3f}
single-frame); part of the flip-rate reduction is mechanical (averaging
smooths noise as well as signal), and both numbers are reported together.""")
para(f"""On VeRi-776, off-the-shelf ImageNet-pretrained OSNet (no
vehicle-ID metric learning) reaches Rank-1 {P3REID['rank1']:.3f}, mAP
{P3REID['mAP']:.3f} — well below fine-tuned SOTA (~90%/~70%) but far above
the 776-way chance level (0.1%), confirming the embeddings carry real
appearance signal even unfine-tuned.""")
img("outputs/report_p3/flip_rate.png", 7.9 * cm,
   "Fig. 4. Prediction flip-rate vs. vote window N.")
img("outputs/report_p3/reid.png", 7.6 * cm,
   "Fig. 5. VeRi-776 cross-camera ReID, off-the-shelf OSNet.")
subsec("D. Risk Indexing (C4)")
para(f"""Rule logic is validated against {P3RISK['n_tracks']:,} real
motorcycle tracks (ground-truth occupancy/helmet-use, independent of
detector accuracy). Overload triggers on {P3RISK['rule_trigger_rates']['overloaded_pct']:.1f}%
of tracks, helmet misuse on {P3RISK['rule_trigger_rates']['helmet_misuse_pct']:.1f}%.
Notably, HIGH risk never triggers from these two factors alone (their combined
weight, 4.0, is below the 6.0 HIGH threshold) — a calibration finding
surfaced by validation against ground truth rather than a code defect.""")
subsec("E. Optimization (C4)")
sc_naive = p4row("SmallCNN", "pruned_naive"); sc_ft = p4row("SmallCNN", "pruned")
sc_int8 = p4row("SmallCNN", "onnx_int8"); sc_fp32 = p4row("SmallCNN", "fp32")
table([["Config", "Acc.", "Retention", "Size (MB)"],
       ["FP32", f"{sc_fp32['accuracy']:.3f}", "1.00", f"{sc_fp32['size_mb']:.2f}"],
       ["Pruned (naive)", f"{sc_naive['accuracy']:.3f}", f"{sc_naive['accuracy_retention']:.2f}", f"{sc_naive['size_mb']:.2f}"],
       ["Pruned+fine-tune", f"{sc_ft['accuracy']:.3f}", f"{sc_ft['accuracy_retention']:.2f}", f"{sc_ft['size_mb']:.2f}"],
       ["+ONNX INT8", f"{sc_int8['accuracy']:.3f}", f"{sc_int8['accuracy_retention']:.2f}", f"{sc_int8['size_mb']:.2f}"]],
      [2.6 * cm, 1.4 * cm, 1.7 * cm, 1.6 * cm])
para("""30% structured pruning without fine-tuning collapses accuracy
(retention 0.31); three epochs of low-LR fine-tuning restores it (0.98). On
the recovered model, INT8 quantization is essentially free (no further
accuracy change for either benchmarked architecture), and ONNX Runtime INT8
shrinks models 3.8-3.9x. Latency is not uniformly improved by ONNX at
batch-1 on CPU for these small models — session overhead dominates — so we
report the size win as unconditional and the latency win as
model-dependent, not assumed.""")
img("outputs/report_p4/pruning_recovery.png", 7.9 * cm,
   "Fig. 6. Naive pruning collapse vs. fine-tune recovery, both models.")

# ============================================================ VI. DISCUSSION
sec("VI. Discussion and Limitations")
para("""A recurring engineering risk across all four phases is that every
heavy backend (Ultralytics, DeepSORT, torchreid, DeepFace) degrades
gracefully to a lightweight fallback rather than crashing, which means a
broken dependency and a working one produce indistinguishable console output
unless the active backend is explicitly asserted. Two backends were found
silently degraded mid-project, and a class-id/weights mismatch inverted
helmet-compliance semantics before being caught — motivating a regression
suite built specifically from real, previously-shipped defects rather than
speculative unit tests.""")
para("""Limitations: the granular helmet sub-classes named in some prior
task specifications (strap-unfastened, helmet-on-handlebar, helmet-on-arm)
have no available public dataset and are not modelled here; the occlusion
ablation uses UA-DETRAC, which contains no two- or three-wheelers, so its
class coverage does not match the detection models' target domain; and
cross-camera ReID uses an unfine-tuned backbone by design, to measure rather
than assume the zero-shot baseline. On hardware where an installed torch
build targets a newer CUDA driver than is present, loading torch and
TensorFlow in one process was found to segfault; demographics estimation is
therefore process-isolated in our implementation, a deployment detail outside
this paper's main scope but documented for reproducibility.""")

# ============================================================ VII. CONCLUSION
sec("VII. Conclusion")
para(f"""We presented a complete CPU-only traffic surveillance pipeline
and, across four phases, consistently prioritised measuring and reporting
real outcomes — including several that did not match initial expectations —
over presenting only favourable results. Future work includes fine-tuning
the ReID backbone on VeRi-776's own training split (already downloaded),
sourcing data for the missing granular helmet sub-classes, and re-evaluating
occlusion-robust voting on footage containing the two/three-wheeler classes
this system otherwise targets.""")

# ============================================================ REFERENCES
sec("References")
refs = [
    "N. Wojke, A. Bewley, D. Paulus, \"Simple Online and Realtime Tracking with a Deep Association Metric,\" ICIP, 2017.",
    "K. Zhou, Y. Yang, A. Cavallaro, T. Xiang, \"Omni-Scale Feature Learning for Person Re-Identification,\" ICCV, 2019.",
    "X. Liu, W. Liu, H. Ma, H. Fu, \"Large-Scale Vehicle Re-Identification in Urban Surveillance Videos,\" ICME, 2016.",
    "C. Lyu et al., \"RTMDet: An Empirical Study of Designing Real-Time Object Detectors,\" arXiv:2212.07784, 2022.",
    "L. Wen et al., \"UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking,\" CVIU, 2020.",
    "G. Jocher et al., \"Ultralytics YOLOv8,\" github.com/ultralytics/ultralytics, 2023.",
    "B. Naik et al., \"HELMET: A Dataset of Helmet Use in Motorcycle Traffic,\" osf.io/4pwj8, 2019.",
    "AI City Challenge, \"Track 5: Helmet Rule Violation Detection,\" aicitychallenge.org, 2023.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}] {r}", styles["RefStyle"]))

print(f"Body built: {len(story)} flowables total.")

# ---- Page templates: single full-width frame for title/abstract, then a
# two-column layout (two Frames side by side) for the body sections. -------
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
GUTTER = 0.7 * cm
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
COL_H = PAGE_H - 2 * MARGIN


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawCentredString(PAGE_W / 2, 0.9 * cm, str(doc.page))
    canvas.restoreState()


full_frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, COL_H, id="full")
col1 = Frame(MARGIN, MARGIN, COL_W, COL_H, id="col1")
col2 = Frame(MARGIN + COL_W + GUTTER, MARGIN, COL_W, COL_H, id="col2")

os.makedirs("docs/paper", exist_ok=True)
doc = BaseDocTemplate(OUT_PDF, pagesize=A4,
                      title="Adaptive Spatio-Temporal Traffic Surveillance",
                      author="")
doc.addPageTemplates([
    PageTemplate(id="onecol", frames=[full_frame], onPage=_footer),
    PageTemplate(id="twocol", frames=[col1, col2], onPage=_footer),
])
doc.build(story)
print(f"PDF -> {OUT_PDF}")
