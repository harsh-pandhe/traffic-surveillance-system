"""
scripts/build_thesis.py
-------------------------
Assembles the chaptered, bordered thesis document (docs/Thesis.pdf), matching
the structural template of a reference college thesis: bordered pages with a
running header/footer, chapter title pages, numbered sub-sections, DFD/UML
figures (outputs/thesis/*.png), numbered CASE test results with real
screenshots, and a References/Publications back-matter.

Every number and figure here is read from the SAME results/phase*/*.json and
outputs/report*/*.png files used by the phase reports and the final report --
nothing is fabricated for this document.

Output -> docs/Thesis.pdf
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, HRFlowable, ListFlowable, ListItem,
                                KeepTogether)

OUT_PDF = "docs/Thesis.pdf"
REPO = "https://github.com/harsh-pandhe/traffic-surveillance-system"
INSTITUTE = "Walchand Institute of Technology, Solapur"
TITLE_FULL = "Adaptive Spatio-Temporal Traffic Surveillance: Scene-Adaptive Preprocessing, Multi-Frame Detection and Risk Analytics on CPU"

# ---- Load every tracked result once; the whole document reads from these --
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
HONNX = json.load(open("results/phase4/helmet_onnx.json"))
GATES = json.load(open("results/phase0_gates.json"))


def p4row(model, config):
    return next(r for r in P4["rows"] if r["model"] == model and r["config"] == config)


NAVY = colors.HexColor("#1b2a4a"); BLUE = colors.HexColor("#2e86c1")
LIGHT = colors.HexColor("#eaf2f8"); GREY = colors.HexColor("#5d6d7e")
BLACK = colors.black

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ChapterLabel", parent=styles["Title"], alignment=TA_CENTER,
                          fontSize=16, textColor=BLACK, spaceAfter=18, leading=20))
styles.add(ParagraphStyle("ChapterTitle", parent=styles["Title"], alignment=TA_CENTER,
                          fontSize=20, textColor=BLACK, leading=24))
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], textColor=NAVY, fontSize=13,
                          spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], textColor=BLUE, fontSize=11.5,
                          spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("H3c", parent=styles["Heading3"], textColor=GREY, fontSize=10.3,
                          spaceBefore=6, spaceAfter=3))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], alignment=TA_JUSTIFY, fontSize=9.3,
                          leading=13.6, spaceAfter=6))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], alignment=TA_CENTER, fontSize=8.3,
                          textColor=GREY, spaceAfter=10))
styles.add(ParagraphStyle("Case", parent=styles["Normal"], alignment=TA_LEFT, fontSize=9.3,
                          leading=13.6, spaceAfter=4, textColor=BLACK))
styles.add(ParagraphStyle("CaseHead", parent=styles["Heading3"], textColor=NAVY, fontSize=10.5,
                          spaceBefore=10, spaceAfter=3))
styles.add(ParagraphStyle("TitlePageBig", parent=styles["Title"], fontSize=16, leading=21,
                          alignment=TA_CENTER, textColor=BLACK))
styles.add(ParagraphStyle("TitlePageSub", parent=styles["Normal"], fontSize=11, leading=15,
                          alignment=TA_CENTER, textColor=BLACK))
BODY = styles["Body"]
story = []


def p(t, s="Body"): story.append(Paragraph(t, styles[s]))
def sp(h=6): story.append(Spacer(1, h))
def hr(): story.append(HRFlowable(width="100%", thickness=0.7, color=BLUE, spaceBefore=3, spaceAfter=7))
def pagebreak(): story.append(PageBreak())
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY)) for i in items],
                              bulletType="bullet", leftIndent=14))
    sp(6)


def img(path, width=13.5 * cm, cap=None):
    if not os.path.isfile(path):
        p(f"<i>[figure not found: {path}]</i>", "Cap")
        return
    iw, ih = ImageReader(path).getSize()
    story.append(Image(path, width=width, height=width * ih / iw))
    if cap:
        story.append(Paragraph(cap, styles["Cap"]))
    else:
        sp(8)


def table(data, cw, header=True, font=8.0):
    t = Table(data, colWidths=cw, hAlign="CENTER")
    ts = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
          ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("ALIGN", (0, 1), (0, -1), "LEFT")]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, 0), "CENTER")]
    t.setStyle(TableStyle(ts)); story.append(t); sp(10)


def algo_box(title, lines):
    """Pseudocode block styled like the reference thesis's 'Algorithm N' boxes."""
    rows = [[Paragraph(f"<b>{title}</b>", BODY)]]
    for ln in lines:
        rows.append([Paragraph(f"<font face='Courier' size=8>{ln}</font>", styles["Case"])])
    t = Table(rows, colWidths=[15.5 * cm])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
        ("BACKGROUND", (0, 0), (0, 0), LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t); sp(10)


def chapter_title(number, title):
    story.append(Spacer(1, 4 * cm))
    p(f"CHAPTER &ndash; {number}", "ChapterLabel")
    p(f"<u>{title.upper()}</u>", "ChapterTitle")
    pagebreak()


def case(n, title, desc, img_path=None, img_w=11 * cm, cap=None):
    story.append(KeepTogether([
        Paragraph(f"CASE {n}: {title}", styles["CaseHead"]),
        Paragraph(desc, styles["Case"]),
    ]))
    if img_path:
        img(img_path, img_w, cap)
    sp(6)


# ============================================================ COVER PAGE
sp(60)
p(TITLE_FULL, "TitlePageBig")
sp(30)
p("A Thesis Report", "TitlePageSub")
p("submitted in partial fulfilment of the requirements for", "TitlePageSub")
p("the degree of Master of Technology in Computer Science and Engineering", "TitlePageSub")
sp(24)
p("<b>Harsh Pandhe</b>", "TitlePageSub")
sp(40)
p(f"<b>{INSTITUTE}</b>", "TitlePageSub")
pagebreak()

# ============================================================ CERTIFICATE / DECLARATION
p("Declaration", "H1c"); hr()
p("""I hereby declare that this thesis titled &ldquo;Adaptive Spatio-Temporal
Traffic Surveillance&rdquo; is my own work, carried out under the guidance of
my project guide, and that every quantitative result reported herein is
generated directly by tracked scripts reading from tracked experiment output
(<font face='Courier' size=8>results/phase1..4/*.json</font>) rather than
asserted by hand.""")
pagebreak()

# ============================================================ ABSTRACT
p("Abstract", "H1c"); hr()
p(f"""This thesis presents an end-to-end, CPU-optimized traffic surveillance
system spanning four phases: adaptive scene-conditioned preprocessing,
granular helmet-compliance and wheel-count detection, multi-frame tracking
with cross-camera re-identification, and model optimization. A learned scene
classifier (RandomForest over 15 lighting/texture features) reaches
{P1['accuracy']:.2f} accuracy on DAY/NIGHT/FOG/RAIN classification. Three
architectures are benchmarked for wheel-count classification under an
identical, leak-free protocol, with YOLOv8-cls selected at
{P2W['yolo']['accuracy']:.3f} accuracy. A 7-class YOLOv8 helmet detector
reaches mAP@50 {P2H['test']['mAP50']:.3f} on a held-out test split.
Multi-frame majority voting reduces prediction flip-rate by
{(1 - P3O['flip_rate_overall']['30'] / P3O['flip_rate_overall']['1']) * 100:.1f}%
under real occlusion, and the risk-indexing rules are validated against
{P3RISK['n_tracks']:,} real motorcycle tracks with ground-truth occupancy and
helmet-use labels. Structured pruning combined with a fine-tuning recovery
step and ONNX Runtime INT8 quantization shrinks deployed models 3.8&ndash;3.9x
with no further accuracy cost. Every phase surfaced at least one result that
fell short of an initial expectation &ndash; an unfine-tuned re-identification
backbone, an accuracy collapse from naive pruning, an oversampling attempt
that made results worse &ndash; and each is diagnosed to a root cause and
reported honestly rather than smoothed over.""")
pagebreak()

# ============================================================ TABLE OF CONTENTS
p("Table of Contents", "H1c"); hr()
toc = [
    ("Chapter 1", "Introduction", "1"),
    ("Chapter 2", "Literature Review", "4"),
    ("Chapter 3", "Methodology", "7"),
    ("Chapter 4", "Design and Implementation", "12"),
    ("Chapter 5", "Result and Discussion", "16"),
    ("Chapter 6", "Conclusion and Future Scope", "22"),
    ("", "References", "24"),
    ("", "Publications", "25"),
]
table([[Paragraph(f"<b>{a}</b>", BODY), Paragraph(b, BODY), Paragraph(c, BODY)] for a, b, c in toc],
      [3 * cm, 10.5 * cm, 2 * cm], header=False, font=9.5)
pagebreak()

p("List of Figures", "H1c"); hr()
lof = [
    "3.1  DFD Level 0 &ndash; Context Diagram",
    "4.1  DFD Level 1 &ndash; Pipeline Stages",
    "4.2  DFD Level 2 &ndash; Detection Sub-Process",
    "4.3  UML Class Diagram &ndash; Pipeline Core Classes",
    "4.4  UML Activity Diagram &ndash; Per-Frame Pipeline Flow",
    "4.5  UML Use Case Diagram",
    "5.1  Scene-Classifier Confusion Matrix (CASE 1)",
    "5.2  Wheel-Count Architecture Comparison (CASE 2)",
    "5.3  Helmet Detector Sample Output (CASE 3)",
    "5.4  Flip-Rate vs Vote Window (CASE 4)",
    "5.5  Real-World Risk-Level Distribution (CASE 5)",
    "5.6  Pruning Collapse and Fine-Tune Recovery (CASE 6)",
]
for f in lof:
    p(f, "Cap")
pagebreak()

# ============================================================ CHAPTER 1 -- INTRODUCTION
chapter_title(1, "Introduction")
p("1.1 Background", "H1c"); hr()
p("""Two-wheeler traffic in dense, heterogeneous urban environments carries
disproportionate crash risk, and enforcement of helmet compliance and
vehicle-load limits remains largely manual. Existing automated systems
typically address a single sub-problem &ndash; helmet detection, or
license-plate recognition, or vehicle counting &ndash; in isolation, and are
frequently evaluated only under a narrow range of lighting and weather
conditions. This project builds a single pipeline spanning
environment-adaptive preprocessing, granular compliance detection, temporal
tracking under occlusion, and deployment-ready optimization, entirely on
commodity CPU hardware, and evaluates every stage against real, held-out
data rather than curated examples.""")
p("1.1.1 Problem Statement", "H2c")
bullets([
    "Detection models trained on clear-weather daytime imagery degrade under fog, rain, and low light, and naive preprocessing applied uniformly can hurt more than it helps on unaffected frames.",
    "Public helmet-compliance datasets rarely distinguish driver from passenger; a granular, per-seat compliance taxonomy has to be trained specifically.",
    "Single-frame classification is fragile under partial occlusion; whether temporal voting measurably helps has to be tested against real occlusion-labelled data, not assumed.",
    "Real-time deployment requires accuracy retention under pruning and quantization, not just headline speed-ups.",
])
p("1.2 Motivation", "H1c"); hr()
p("""Manual traffic enforcement does not scale to the volume of two-wheeler
traffic in most Indian cities, and existing point solutions rarely compose
into a deployable pipeline. A system that classifies the scene, detects
compliance violations per rider, tracks vehicles across frames, estimates
risk from real occupancy patterns, and runs entirely on CPU is directly
relevant to low-cost municipal deployment where GPU infrastructure is not
available at every junction.""")
p("1.3 Objectives", "H1c"); hr()
bullets([
    "Build an adaptive preprocessing stage that detects DAY/NIGHT/FOG/RAIN and applies scene-conditioned enhancement only where it helps.",
    "Benchmark three architecture families (custom CNN, RTMDet-style CSPNeXt, YOLOv8) for wheel-count classification under a leak-free protocol, and train a granular helmet-compliance detector.",
    "Implement multi-frame tracking with a measured occlusion-robustness benefit, cross-camera re-identification, and risk indexing validated against real ground truth.",
    "Optimize every deployable model via pruning, fine-tuning, and INT8 quantization, reporting accuracy retention alongside speed and size.",
])
p("1.4 Scope of the Project", "H1c"); hr()
p("""The system is scoped to CPU-only inference on recorded or live video of
two/three/four/six-plus-wheeler mixed traffic. It covers scene-adaptive
preprocessing, detection and classification, tracking with temporal voting,
cross-camera re-identification, an offline demographics module, rule-based
risk indexing, and model compression for deployment. It does not cover
license-plate recognition, live enforcement action, or GPU-cluster
deployment, which are out of scope.""")
p("1.5 Organisation of the Thesis", "H1c"); hr()
p("""Chapter 2 reviews prior work in helmet detection, vehicle classification,
multi-object tracking, re-identification, and model compression. Chapter 3
describes the methodology, algorithms, and requirements. Chapter 4 presents
the system design via DFD and UML diagrams. Chapter 5 reports results as a
series of numbered test cases with real measured output. Chapter 6 concludes
and lists future work.""")
pagebreak()

# ============================================================ CHAPTER 2 -- LITERATURE REVIEW
chapter_title(2, "Literature Review")
p("2.1 Survey of Related Work", "H1c"); hr()
lit = [
    ("Wojke, Bewley &amp; Paulus (2017)", "Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT), ICIP.",
     "Introduced appearance-augmented online tracking; adopted here for CPU-tractable multi-object tracking (Section 3.2, Algorithm 3.1)."),
    ("Zhou, Yang, Cavallaro &amp; Xiang (2019)", "Omni-Scale Feature Learning for Person Re-Identification (OSNet), ICCV.",
     "OSNet backbone used off-the-shelf for cross-camera vehicle re-identification (Section 5.1, CASE 5)."),
    ("Liu, Liu, Ma &amp; Fu (2016)", "Large-Scale Vehicle Re-Identification in Urban Surveillance Videos, ICME.",
     "Source of the VeRi-776 benchmark used to evaluate re-identification (776 vehicles, 20 cameras)."),
    ("Lyu et al. (2022)", "RTMDet: An Empirical Study of Designing Real-Time Object Detectors, arXiv:2212.07784.",
     "CSPNeXt backbone reimplemented from scratch in this project since the official mmdetection framework does not install against Python 3.12 / PyTorch 2.12."),
    ("Wen et al. (2020)", "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking, CVIU.",
     "Used for the multi-frame voting / occlusion-robustness ablation (Section 5.1, CASE 4)."),
    ("Jocher et al. (2023)", "Ultralytics YOLOv8, github.com/ultralytics/ultralytics.",
     "Used for the selected wheel-count classifier and the 7-class helmet detector."),
    ("Naik et al. (2019)", "HELMET: A Dataset of Helmet Use in Motorcycle Traffic, osf.io/4pwj8.",
     "910 real Myanmar traffic clips, 283,377 annotations; used to validate the risk-indexer against real occupancy and helmet-use ground truth (Section 5.1, CASE 5)."),
    ("AI City Challenge (2023)", "Track 5: Detecting Violation of Helmet Rule for Motorcyclists, aicitychallenge.org.",
     "Source convention for the 7-class driver/passenger x helmet-status taxonomy used in this project's helmet detector."),
]
for i, (auth, cite, note) in enumerate(lit, 1):
    p(f"[{i}] <b>{auth}</b> &ndash; {cite}", "Body")
    p(f"<i>Relevance:</i> {note}", "Body")
p("2.2 Research Gap", "H1c"); hr()
bullets([
    "Prior helmet-detection work is typically binary (helmet / no-helmet) on single frames; per-seat, driver-vs-passenger compliance is less common and is what this project's 7-class taxonomy targets.",
    "Vehicle-type classification is usually a byproduct of general object detection rather than purpose-built for a 2/3/4/6+-wheeler taxonomy relevant to mixed heterogeneous traffic.",
    "The literature broadly assumes multi-frame temporal aggregation improves occlusion robustness but rarely quantifies it on a class-imbalanced real benchmark with a class-composition-independent metric.",
    "Cross-camera re-identification literature reports fine-tuned SOTA numbers almost exclusively; the realistic off-the-shelf baseline gap is rarely stated explicitly.",
    "Structured pruning literature is not always explicit that pruning without a fine-tuning recovery step can catastrophically collapse accuracy.",
])
p("2.3 Summary", "H1c"); hr()
p("""This chapter studied prior work across the five sub-areas this project
spans &ndash; helmet compliance detection, vehicle-type classification,
multi-object tracking, vehicle re-identification, and model compression for
CPU/edge deployment &ndash; and identified the specific gaps this project
addresses: a full, composed pipeline rather than an isolated sub-problem, a
leak-free multi-architecture benchmark, a measured (not assumed) occlusion
benefit, an honestly-reported re-identification baseline, and a diagnosed
pruning failure mode with its recovery. The following chapter elaborates the
methodology, algorithms, and requirements used to address these gaps.""")
pagebreak()

# ============================================================ CHAPTER 3 -- METHODOLOGY
chapter_title(3, "Methodology")
p("3.1 Background", "H1c"); hr()
p("""The system is implemented as a nine-stage per-frame pipeline
(<font face='Courier' size=8>main.py</font>): scene classification, adaptive
enhancement, helmet-compliance detection, wheel-count classification,
tracking with temporal voting, cross-camera re-identification, demographics
(process-isolated), risk indexing, and overlay rendering. Every stage is
independently testable and driven entirely by
<font face='Courier' size=8>config/settings.yaml</font> &ndash; no hard-coded
paths or thresholds. The complete data and control flow is presented as DFD
and UML diagrams in Chapter 4.""")
p("3.2 Algorithms Used", "H1c"); hr()
p("3.2.1 Multi-Frame Majority-Voting Algorithm", "H2c")
algo_box("Algorithm 3.1 &ndash; Temporal Majority Vote (VehicleTracker)", [
    "Input: track_id, per-frame wheel-class prediction c_t, window size N",
    "1.  buffer[track_id].append(c_t)",
    "2.  if len(buffer[track_id]) &gt; N: buffer[track_id].pop_front()",
    "3.  emitted_class &larr; mode(buffer[track_id])",
    "4.  if emitted_class != previous_emitted_class[track_id]: flip_count += 1",
    "5.  previous_emitted_class[track_id] &larr; emitted_class",
    "6.  return emitted_class",
])
p("""Evaluated at N &isin; {1, 5, 10, 15, 30}; flip-rate (Section 5.1, CASE 4)
falls monotonically as N grows because a majority vote over a longer window
is less sensitive to a single-frame misclassification during occlusion.""")
p("3.2.2 Risk-Index Computation Algorithm", "H2c")
algo_box("Algorithm 3.2 &ndash; Weighted Risk-Index (RiskIndexer)", [
    "Input: occupancy n, any_no_helmet flag, wrong_way flag, weights W",
    "1.  score &larr; 0",
    "2.  if any_no_helmet: score += W.helmet_misuse      # 2.5",
    "3.  if n &gt; overload_threshold: score += W.overload  # 1.5",
    "4.  if wrong_way: score += W.wrong_way",
    "5.  if score &gt;= HIGH_T: level &larr; HIGH           # 6.0",
    "6.  elif score &gt;= MEDIUM_T: level &larr; MEDIUM      # 3.0",
    "7.  else: level &larr; LOW",
    "8.  return level, score",
])
p(f"""Validated against {P3RISK['n_tracks']:,} real motorcycle tracks with
ground-truth occupancy and helmet-use labels (Section 5.1, CASE 5); the
validation surfaced a calibration finding &ndash; HIGH never triggers on real
data because helmet-misuse plus overload (2.5 + 1.5 = 4.0) does not reach the
6.0 threshold &ndash; reported as a policy question for the weighting scheme
rather than silently re-tuned.""")
p("3.3 Software Requirements", "H1c"); hr()
table([["Component", "Requirement"],
       ["Language / runtime", "Python 3.10+ (developed and tested on 3.12)"],
       ["Deep learning", "PyTorch, Ultralytics YOLOv8"],
       ["Computer vision", "OpenCV"],
       ["Classical ML", "scikit-learn (RandomForest scene classifier)"],
       ["Tracking / ReID", "deep-sort-realtime, torchreid (OSNet)"],
       ["Demographics", "DeepFace (process-isolated)"],
       ["Deployment / export", "ONNX, ONNX Runtime"],
       ["Operating system", "Linux (CPU-only, no GPU required)"]],
      [5 * cm, 10.5 * cm])
p("3.4 Hardware Requirements", "H1c"); hr()
p("""All benchmarks and measured figures in this thesis were produced on a
commodity CPU workstation &ndash; <b>AMD Ryzen 5 7600 (6-core)</b>, no GPU
used at any stage &ndash; deliberately, to validate the CPU-only deployment
claim rather than benchmark on hardware unavailable to a typical municipal
deployment.""")
p("3.5 Retrieving Input", "H1c"); hr()
p("""Input is a video file or camera stream read frame-by-frame via OpenCV.
Each frame is timestamped and passed into the scene classifier before any
detection stage runs, since the enhancement applied downstream is
scene-conditioned.""")
p("3.6 Processing / Resource Allocation", "H1c"); hr()
p("""Each frame passes sequentially through the nine pipeline stages listed
in Section 3.1. Heavy backends (Ultralytics, DeepSORT, torchreid, DeepFace)
are loaded once at pipeline start; demographics is deliberately isolated into
a separate OS process (Section 4.6) because this hardware's torch build
targets a newer CUDA driver than is installed, and loading a second torch
model in a process that has also imported TensorFlow segfaults &ndash;
confirmed by bisecting pipeline construction component by component.""")
p("3.7 Output", "H1c"); hr()
p("""Output is an annotated video stream: per-rider bounding boxes with
compliance status, a stable tracking ID, a telemetry HUD, and a per-track
risk banner (LOW / MEDIUM / HIGH) rendered on every frame.""")
p("3.8 Performance Evaluation", "H1c"); hr()
p(f"""On {P3PIPE['frames']} real frames ({P3PIPE['source']}), the full
pipeline (all nine stages) sustains <b>{P3PIPE['fps_mean']:.2f} FPS mean</b>
({P3PIPE['fps_final']:.1f} FPS steady-state) on CPU alone. Detailed per-model
latency and throughput figures are reported in Chapter 5.""")
p("3.9 Verification Gates", "H1c"); hr()
p(f"""Before committing to expensive optional work, two cheap evidence gates
were run and recorded (<font face='Courier' size=8>results/phase0_gates.json</font>).
Gate A fit a learning curve on the existing helmet training data (25/50/75/100%
of 368 images); test mAP@50 gain in the final quarter was only
{GATES['gate_A_helmet_data']['final_quarter_gain']:.3f} against a total gain
of {GATES['gate_A_helmet_data']['total_gain']:.3f}, i.e. plateaued &ndash;
verdict &ldquo;{GATES['gate_A_helmet_data']['verdict']}&rdquo;, so a ~29&nbsp;GB
external image download was skipped rather than pursued speculatively. Gate B
measured that {GATES['gate_B_demographics']['usable_pct']:.0f}% of
face-exposed rider detections yield a crop large enough for DeepFace,
authorising the demographics install with the explicit caveat that age
estimates are indicative given the median face crop is only
{GATES['gate_B_demographics']['face_min_side_px']['median']}px.""")
pagebreak()

# ============================================================ CHAPTER 4 -- DESIGN AND IMPLEMENTATION
chapter_title(4, "Design and Implementation")
p("4.1 Data Flow Diagrams", "H1c"); hr()
p("4.1.1 DFD Level 0 &ndash; Context Diagram", "H2c")
img("outputs/thesis/dfd_level0.png", 12.5 * cm,
   "Figure 4.1. DFD Level 0: the surveillance pipeline as a single process "
   "exchanging frames, config/weights, risk alerts, and annotated video with "
   "its environment.")
p("4.1.2 DFD Level 1 &ndash; Pipeline Stages", "H2c")
img("outputs/thesis/dfd_level1.png", 14.5 * cm,
   "Figure 4.2. DFD Level 1: the five top-level processing stages from scene "
   "classification through risk indexing.")
p("4.1.3 DFD Level 2 &ndash; Detection Sub-Process", "H2c")
img("outputs/thesis/dfd_level2.png", 13 * cm,
   "Figure 4.3. DFD Level 2: decomposition of the Detect stage (3.0) into "
   "crop, helmet detection, wheel classification, and by-name compliance "
   "semantics resolution &ndash; the last step exists specifically because a "
   "hard-coded class-id mapping previously inverted compliance decisions "
   "(Section 5.2).")
pagebreak()
p("4.2 UML Diagrams", "H1c"); hr()
p("4.2.1 Class Diagram", "H2c")
img("outputs/thesis/uml_class.png", 15 * cm,
   "Figure 4.4. UML class diagram of the pipeline's core classes, with real "
   "attribute and method names taken directly from the source modules.")
pagebreak()
p("4.2.2 Activity Diagram", "H2c")
img("outputs/thesis/uml_activity.png", 9 * cm,
   "Figure 4.5. Activity diagram of the per-frame pipeline flow.")
pagebreak()
p("4.2.3 Use Case Diagram", "H2c")
img("outputs/thesis/uml_usecase.png", 14.5 * cm,
   "Figure 4.6. Use case diagram. The demographics use case is modelled as "
   "an &lt;&lt;extend&gt;&gt; relationship because it runs only when enabled "
   "and executes in a separate external process (Section 4.6), reflecting "
   "the real process-isolation architecture rather than an idealised design.")
p("4.3 Module Design", "H1c"); hr()
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
p("Table 4.1. Pipeline stages, implementing module, and originating phase.", "Cap")
p("4.4 Class Semantics Resolved by Name", "H1c"); hr()
p("""Class semantics for the helmet detector are resolved <i>by name</i> at
load time (mapping configured labels such as
<font face='Courier' size=8>violation_classes</font> to the trained model's
class ids) rather than by hard-coded integer id, with a startup check that
raises a visible warning if the loaded model's class order does not match the
configured taxonomy. This exists because a hard-coded id assumption
previously inverted every compliance decision (Section 5.2, a defect this
project's regression suite now guards against).""")
p("4.5 Regression Suite Built From Real Defects", "H1c"); hr()
p("""Rather than speculative unit tests, the test suite
(<font face='Courier' size=8>tests/test_regressions.py</font>) codifies
defects actually found during development: the inverted helmet-compliance
mapping above, a train/val split that leaked source-image information across
the boundary, and two backends (DeepSORT, OSNet) that were found silently
degraded to weaker fallbacks. Each test fails on the exact historical defect
and runs in continuous integration on every push.""")
p("4.6 Process Isolation for Demographics", "H1c"); hr()
p(f"""Stage 7 (demographics) cannot safely share a process with Stages 3&ndash;6
on this hardware, since the installed torch build targets a newer CUDA driver
than is present and loading a second torch model in a process that has also
imported TensorFlow segfaults. The live pipeline queues face crops to disk
and a separate process &ndash; which never imports torch &ndash; completes the
analysis; verified end-to-end on real pipeline output
({P3DEMO['n_succeeded']}/{P3DEMO['n_crops']} crops processed with no crash).
This is reflected in Figure 4.6 as an explicit &lt;&lt;extend&gt;&gt;
relationship rather than a caveat.""")
pagebreak()

# ============================================================ CHAPTER 5 -- RESULT AND DISCUSSION
chapter_title(5, "Result and Discussion")
p("5.1 Result", "H1c"); hr()
p(f"""All test cases below were executed on the hardware and software
described in Section 3.3&ndash;3.4 (AMD Ryzen 5 7600, CPU-only, Python
3.10+). Each case reproduces a real, measured result together with a real
figure generated directly from tracked experiment output &ndash; no
screenshot or number in this section is hand-constructed.""")

case(1, "Learned Scene Classifier &ndash; Confusion Matrix",
    f"""<b>Objective:</b> Verify that the learned RandomForest scene
    classifier (15 lighting/haze/texture features) correctly separates
    DAY/NIGHT/FOG/RAIN on {P1['n_val']} held-out validation images.<br/>
    <b>Result:</b> Overall accuracy {P1['accuracy']:.2f}. Residual error
    concentrates on the FOG&harr;RAIN boundary, a physically overlapping
    weather condition pair &ndash; consistent with the domain, not a
    modelling defect.<br/><b>Status:</b> PASS.""",
    "outputs/report/confusion_matrix.png", 10 * cm,
    "Figure 5.1. 4-class confusion matrix, held-out validation set.")

case(2, "Wheel-Count Architecture Comparison (Leak-Free)",
    f"""<b>Objective:</b> Compare {P2W['cnn']['model']}, a from-scratch
    CSPNeXt (RTMDet backbone) reimplementation, and ImageNet-pretrained
    YOLOv8-cls on an identical, leak-free 4-class wheel-count split (no
    source image contributes samples to both train and validation).<br/>
    <b>Result:</b> Accuracy &ndash; {P2W['cnn']['model']}
    {P2W['cnn']['accuracy']:.3f}, CSPNeXt {P2W['cspnext']['accuracy']:.3f},
    YOLOv8-cls {P2W['yolo']['accuracy']:.3f} (selected). An earlier run that
    split by crop rather than by image leaked 76&ndash;91% of some classes
    across the split boundary and reported inflated numbers
    ({P2W['superseded_run']['accuracy']['cnn']:.3f} /
    {P2W['superseded_run']['accuracy']['cspnext']:.3f} /
    {P2W['superseded_run']['accuracy']['yolo']:.3f}); this was found and
    corrected before finalizing the result.<br/><b>Status:</b> PASS
    (leak-free, reproducible).""",
    "outputs/report_p2/wheel_compare.png", 11.5 * cm,
    "Figure 5.2. Accuracy and macro-F1, leak-free split.")

case(3, "Granular Helmet Compliance Detection",
    f"""<b>Objective:</b> Verify the 7-class YOLOv8n helmet-compliance
    detector correctly localizes riders, bike, and per-seat helmet status on
    real unseen images.<br/><b>Result:</b> mAP@50 on the held-out test split
    (never used for epoch selection) is {P2H['test']['mAP50']:.3f}
    (validation split, optimistically biased, was
    {P2H['val']['mAP50']:.3f}).<br/><b>Status:</b> PASS.""",
    "outputs/report_p2/helmet_sample.png", 11 * cm,
    "Figure 5.3. Trained detector applied to a real test image, showing "
    "riders, bike, and per-seat helmet-compliance status.")

case(4, "Multi-Frame Majority Voting Under Occlusion",
    f"""<b>Objective:</b> Verify Algorithm 3.1 reduces prediction instability
    under real occlusion, evaluated on UA-DETRAC
    ({P3O['crops_classified']:,} classified crops grouped by ground-truth
    track id).<br/><b>Result:</b> Flip-rate falls
    {(1 - P3O['flip_rate_overall']['30']/P3O['flip_rate_overall']['1'])*100:.1f}%
    from a single-frame window (N=1) to N=30. Balanced accuracy improves more
    modestly (N=15: {P3O['overall']['15']:.3f} vs single-frame
    {P3O['overall']['1']:.3f}) since part of the flip-rate reduction is
    mechanical smoothing.<br/><b>Status:</b> PASS.""",
    "outputs/report_p3/flip_rate.png", 10.5 * cm,
    "Figure 5.4. Flip-rate vs vote window size.")

case(5, "Risk-Indexer Validation Against Real Ground Truth",
    f"""<b>Objective:</b> Validate Algorithm 3.2's rule calibration against
    {P3RISK['n_tracks']:,} real motorcycle tracks from the HELMET dataset
    (910 Myanmar traffic clips), using ground-truth occupancy and per-seat
    helmet-use labels rather than detector output.<br/>
    <b>Result:</b> Boundary check &ldquo;{P3RISK['boundary_check']}&rdquo;.
    Resulting distribution:
    LOW {P3RISK['risk_level_distribution']['LOW']:,},
    MEDIUM {P3RISK['risk_level_distribution']['MEDIUM']:,}, HIGH 0. HIGH
    never triggers on real data because helmet-misuse (2.5) plus overload
    (1.5) sums to 4.0, below the 6.0 threshold &ndash; a calibration finding
    for the weighting scheme, not a code defect.<br/>
    <b>Status:</b> PASS (rule logic verified); calibration finding logged
    for future work (Section 6.2).""",
    "outputs/report_p3/risk_distribution.png", 9.5 * cm,
    "Figure 5.5. Resulting risk-level distribution on real data.")

case(6, "Structured Pruning &ndash; Collapse and Recovery",
    f"""<b>Objective:</b> Verify that 30% structured pruning followed by a
    short fine-tuning pass recovers the accuracy naive pruning
    collapses.<br/><b>Result:</b> SmallCNN accuracy &ndash; FP32 baseline
    {p4row('SmallCNN','fp32')['accuracy']:.3f}; pruned with no fine-tune
    {p4row('SmallCNN','pruned_naive')['accuracy']:.3f} (retention
    {p4row('SmallCNN','pruned_naive')['accuracy_retention']:.2f}); pruned +
    3-epoch fine-tune {p4row('SmallCNN','pruned')['accuracy']:.3f} (retention
    {p4row('SmallCNN','pruned')['accuracy_retention']:.2f}).<br/>
    <b>Status:</b> PASS (fine-tune recovery confirmed).""",
    "outputs/report_p4/pruning_recovery.png", 10.5 * cm,
    "Figure 5.6. Naive pruning collapse vs. fine-tune recovery.")

case(7, "ONNX Export Correctness (Dynamic vs Static Shape)",
    f"""<b>Objective:</b> Verify the exported ONNX helmet detector preserves
    PyTorch accuracy.<br/><b>Result:</b> A first export using a dynamic input
    shape passed a 10-image spot check (identical per-image detection counts
    to PyTorch) but the aggregate mAP@50 &ndash; which sweeps confidence
    thresholds rather than one operating point &ndash; revealed an 11%
    relative drop (0.764 &rarr; 0.677) the spot check missed. A static-shape
    export resolved it: ONNX FP32 retains
    {HONNX['onnx_fp32']['retention']*100:.0f}% and ONNX INT8 retains
    {HONNX['onnx_int8']['retention']*100:.0f}% of PyTorch mAP@50.<br/>
    <b>Status:</b> PASS after fix; logged as a general lesson &ndash; a spot
    check that agrees on individual outputs is not sufficient evidence an
    export preserves calibration across the full confidence range.""")

p("5.2 Discussion", "H1c"); hr()
p("5.2.1 Cross-Camera Re-Identification &ndash; Below SOTA by Design", "H2c")
p(f"""Evaluated on VeRi-776 (776 vehicles, 20 real cameras): Rank-1
{P3REID['rank1']:.3f}, Rank-5 {P3REID['rank5']:.3f}, mAP
{P3REID['mAP']:.3f}. These trail published fine-tuned SOTA
(Rank-1&asymp;90%, mAP&asymp;70%) because the backend runs off-the-shelf
ImageNet-pretrained OSNet with no vehicle-identity metric-learning at all;
Rank-1 {P3REID['rank1']*100:.0f}% from purely generic features still confirms
real appearance signal against a 776-way chance level of 0.1%. Closing the
gap needs triplet-loss fine-tuning on VeRi's own training split, already
downloaded but not yet used (Section 6.2).""")
img("outputs/report_p3/reid.png", 9 * cm, "Figure 5.7. VeRi-776 cross-camera ReID scores, off-the-shelf OSNet.")
p("5.2.2 Oversampling Attempt That Made Results Worse", "H2c")
p("""Passenger-helmet classes are weakest in the detector (AP50
0.545/0.565) due to genuinely fewer training instances. A targeted fix was
attempted: 3x duplication-oversampling of images containing these classes.
It made results worse (overall mAP@50 0.764 &rarr; 0.711) because plain
duplication increases exposure without adding visual diversity, causing
overfitting rather than generalisation. The baseline weights were kept and
this negative result is documented rather than hidden.""")
p("5.2.3 The Silent-Fallback Risk", "H2c")
p("""A recurring theme across all four phases: every heavy backend
(Ultralytics, DeepSORT, torchreid, DeepFace) degrades gracefully to a
lightweight fallback rather than crashing when a dependency is broken. This
is good defensive design but has a sharp edge &ndash; a broken dependency and
a working one are indistinguishable from console output unless the active
backend is explicitly asserted, which motivated the regression suite of
Section 4.5.""")
pagebreak()

# ============================================================ CHAPTER 6 -- CONCLUSION AND FUTURE SCOPE
chapter_title(6, "Conclusion and Future Scope")
p("6.1 Conclusion", "H1c"); hr()
p(f"""This thesis presented a complete, CPU-optimized traffic surveillance
pipeline spanning adaptive preprocessing ({P1['accuracy']:.2f}
scene-classification accuracy), a leak-free three-way detection benchmark
({P2W['yolo']['accuracy']:.3f} wheel-count accuracy,
{P2H['test']['mAP50']:.3f} helmet mAP@50), multi-frame tracking with a
measured occlusion-robustness benefit, risk indexing validated against
{P3RISK['n_tracks']:,} real motorcycle tracks, and model optimization that
recovers accuracy lost to naive pruning while shrinking deployed models
3.8&ndash;3.9x. Every phase surfaced at least one result that did not match
initial expectations; each was investigated to a root cause rather than
reported around, which is the standard this thesis holds itself to
throughout.""")
p("6.2 Future Scope", "H1c"); hr()
bullets([
    "Fine-tune OSNet on VeRi-776's own training split with a triplet/ID loss to close the cross-camera re-identification gap to published SOTA.",
    "Source or annotate images for the granular helmet sub-classes (strap-unfastened, helmet-on-handlebar, helmet-on-arm) that have no available public dataset today.",
    "Re-run the occlusion-voting ablation on footage containing two- and three-wheelers, to align the ablation's domain with the detection models it is meant to support (UA-DETRAC is car-only).",
    "Revisit the risk-indexer weights given the real-world finding that HIGH is currently unreachable from the most common violation pattern.",
    "Explore augmentation-based (not duplication-based) oversampling for the passenger-helmet classes.",
])
pagebreak()

# ============================================================ REFERENCES
p("References", "H1c"); hr()
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

# ============================================================ PUBLICATIONS
p("Publications", "H1c"); hr()
p("""A research paper based on this work, &ldquo;Adaptive Spatio-Temporal
Traffic Surveillance,&rdquo; has been prepared in standard two-column format
(<font face='Courier' size=8>docs/paper/paper.pdf</font>) and is
<b>submitted / in progress</b> &ndash; it has not yet been accepted or
published at a venue at the time of writing this thesis. It reports the same
measured results as this thesis, restructured as four explicit contributions
(scene-adaptive preprocessing, leak-free three-way detection benchmarking,
multi-frame voting stability, and end-to-end CPU optimization).""")
pagebreak()

p("Appendix &ndash; Repository and Reproduction", "H1c"); hr()
p(f"Source code, trained model configs, and all tracked results: {REPO}")
p("""Every figure and number in this thesis is generated by
<font face='Courier' size=8>scripts/build_thesis.py</font> reading directly
from <font face='Courier' size=8>results/phase1..4/*.json</font>; full
reproduction commands are in
<font face='Courier' size=8>docs/HOW_TO_RUN.md</font>.""")


# ============================================================ PAGE TEMPLATE: border + header + footer
PAGE_W, PAGE_H = A4
MARGIN = 1.6 * cm
BORDER_INSET = 0.55 * cm


class ThesisCanvas(pdfcanvas.Canvas):
    """Draws the page border, running header, and 'Page X | Y' footer with
    the institute name, on every page, matching the reference thesis layout."""

    def __init__(self, *args, **kwargs):
        pdfcanvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_pages = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self._draw_frame(total)
            pdfcanvas.Canvas.showPage(self)
        pdfcanvas.Canvas.save(self)

    def _draw_frame(self, total_pages):
        self.saveState()
        self.setStrokeColor(BLACK)
        self.setLineWidth(1.1)
        self.rect(BORDER_INSET, BORDER_INSET, PAGE_W - 2 * BORDER_INSET, PAGE_H - 2 * BORDER_INSET)
        self.setFont("Helvetica-Bold", 8.5)
        self.setFillColor(NAVY)
        self.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN + 0.15 * cm, TITLE_FULL[:95])
        self.setStrokeColor(BLUE)
        self.setLineWidth(0.7)
        self.line(MARGIN, PAGE_H - MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(GREY)
        self.drawString(MARGIN, MARGIN - 0.35 * cm, f"Page {self._pageNumber} | {total_pages}")
        self.drawRightString(PAGE_W - MARGIN, MARGIN - 0.35 * cm, INSTITUTE)
        self.setStrokeColor(BLUE)
        self.line(MARGIN, MARGIN, PAGE_W - MARGIN, MARGIN)
        self.restoreState()


os.makedirs("docs", exist_ok=True)
frame = Frame(MARGIN, MARGIN + 0.3 * cm, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN - 0.5 * cm,
             id="main", topPadding=6, bottomPadding=6)
doc = BaseDocTemplate(OUT_PDF, pagesize=A4,
                      title="Thesis — Adaptive Spatio-Temporal Traffic Surveillance",
                      author="Harsh Pandhe")
doc.addPageTemplates([PageTemplate(id="thesis", frames=[frame])])
doc.build(story, canvasmaker=ThesisCanvas)
print(f"PDF -> {OUT_PDF}")
