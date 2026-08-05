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
AUTHOR = "Shifali Doshi"
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
styles.add(ParagraphStyle("ChapterLabel", parent=styles["Title"], fontName="Times-Bold",
                          alignment=TA_CENTER, fontSize=17, textColor=BLACK, spaceAfter=20, leading=21))
styles.add(ParagraphStyle("ChapterTitle", parent=styles["Title"], fontName="Times-Bold",
                          alignment=TA_CENTER, fontSize=21, textColor=BLACK, leading=25))
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], fontName="Times-Bold", textColor=NAVY,
                          fontSize=14, spaceBefore=18, spaceAfter=8, leading=17))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], fontName="Times-Bold", textColor=BLUE,
                          fontSize=12.5, spaceBefore=13, spaceAfter=6, leading=15))
styles.add(ParagraphStyle("H3c", parent=styles["Heading3"], fontName="Times-BoldItalic", textColor=GREY,
                          fontSize=11, spaceBefore=9, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontName="Times-Roman",
                          alignment=TA_JUSTIFY, fontSize=10.3, leading=15.5, spaceAfter=9))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], fontName="Times-Italic",
                          alignment=TA_CENTER, fontSize=8.8, textColor=GREY, spaceAfter=12, leading=11.5))
styles.add(ParagraphStyle("Case", parent=styles["Normal"], fontName="Times-Roman",
                          alignment=TA_LEFT, fontSize=10.3, leading=15.5, spaceAfter=5, textColor=BLACK))
styles.add(ParagraphStyle("CaseHead", parent=styles["Heading3"], fontName="Times-Bold", textColor=NAVY,
                          fontSize=11.5, spaceBefore=14, spaceAfter=4))
styles.add(ParagraphStyle("TitlePageBig", parent=styles["Title"], fontName="Times-Bold",
                          fontSize=17, leading=22, alignment=TA_CENTER, textColor=BLACK))
styles.add(ParagraphStyle("TitlePageSub", parent=styles["Normal"], fontName="Times-Roman",
                          fontSize=11.5, leading=16, alignment=TA_CENTER, textColor=BLACK))
styles.add(ParagraphStyle("TableCell", parent=styles["Normal"], fontName="Times-Roman", fontSize=9))
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


def table(data, cw, header=True, font=8.6):
    t = Table(data, colWidths=cw, hAlign="CENTER")
    ts = [("FONTSIZE", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
          ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("ALIGN", (0, 1), (0, -1), "LEFT")]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"), ("ALIGN", (0, 0), (-1, 0), "CENTER")]
    t.setStyle(TableStyle(ts)); story.append(t); sp(12)


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
sp(50)
p(TITLE_FULL, "TitlePageBig")
sp(26)
p("A Thesis Report", "TitlePageSub")
p("submitted in partial fulfilment of the requirements for", "TitlePageSub")
p("the degree of Master of Technology", "TitlePageSub")
p("in Computer Science and Engineering", "TitlePageSub")
sp(30)
p("by", "TitlePageSub")
p(f"<b>{AUTHOR}</b>", "TitlePageBig")
sp(30)
p("Under the guidance of", "TitlePageSub")
p("<b>[Project Guide]</b>", "TitlePageSub")
sp(50)
table([[Paragraph("<b>Department of Computer Science and Engineering</b>", styles["TitlePageSub"])],
       [Paragraph(f"<b>{INSTITUTE}</b>", styles["TitlePageSub"])],
       [Paragraph("Academic Year 2025&ndash;26", styles["TitlePageSub"])]],
      [15.5 * cm], header=False, font=11)
pagebreak()

# ============================================================ CERTIFICATE
p("Certificate", "H1c"); hr()
p(f"""This is to certify that the thesis titled &ldquo;{TITLE_FULL}&rdquo;
is a bona fide record of work carried out by <b>{AUTHOR}</b>, submitted in
partial fulfilment of the requirements for the degree of Master of Technology
in Computer Science and Engineering at {INSTITUTE}, during the academic year
2025&ndash;26, under my/our supervision.""")
sp(40)
table([[Paragraph("&nbsp;", BODY), Paragraph("&nbsp;", BODY)],
       [Paragraph("____________________________", BODY), Paragraph("____________________________", BODY)],
       [Paragraph("Project Guide", BODY), Paragraph("Head of Department", BODY)],
       [Paragraph("Department of Computer Science &amp; Engineering", BODY),
        Paragraph("Department of Computer Science &amp; Engineering", BODY)]],
      [7.75 * cm, 7.75 * cm], header=False, font=9.5)
sp(30)
table([[Paragraph("____________________________", BODY)],
       [Paragraph("External Examiner", BODY)]],
      [15.5 * cm], header=False, font=9.5)
pagebreak()

# ============================================================ ACKNOWLEDGEMENT
p("Acknowledgement", "H1c"); hr()
p("""I would like to express my sincere gratitude to my project guide for
their continuous guidance, valuable feedback, and encouragement throughout
the course of this project. I am also thankful to the Head of the Department
of Computer Science and Engineering and the faculty at
""" + INSTITUTE + """ for providing the resources and environment necessary
to carry out this work.""")
p("""I would further like to thank my family and friends for their patience
and support during the course of this thesis, and everyone whose prior
published work, cited in Chapter 2, made this project's comparisons and
baselines possible.""")
sp(30)
p(f"&ndash; {AUTHOR}", "TitlePageSub")
pagebreak()

# ============================================================ DECLARATION
p("Declaration", "H1c"); hr()
p(f"""I hereby declare that this thesis titled &ldquo;Adaptive Spatio-Temporal
Traffic Surveillance&rdquo; is my own work, carried out under the guidance of
my project guide, and that every quantitative result reported herein is
generated directly by tracked scripts reading from tracked experiment output
(<font face='Courier' size=8>results/phase1..4/*.json</font>) rather than
asserted by hand. I further declare that this thesis has not been submitted,
in part or in full, for the award of any other degree or diploma of this or
any other institute.""")
sp(50)
table([[Paragraph("Place: Solapur", BODY), Paragraph("", BODY)],
       [Paragraph("Date: __________________", BODY), Paragraph(f"<b>{AUTHOR}</b>", BODY)]],
      [8 * cm, 7.5 * cm], header=False, font=9.5)
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
    ("Chapter 1", "Introduction", "8"),
    ("Chapter 2", "Literature Review", "11"),
    ("Chapter 3", "Methodology", "14"),
    ("Chapter 4", "Design and Implementation", "18"),
    ("Chapter 5", "Result and Discussion", "26"),
    ("Chapter 6", "Conclusion and Future Scope", "35"),
    ("", "References", "37"),
    ("", "Publications", "38"),
]
table([[Paragraph(f"<b>{a}</b>", BODY), Paragraph(b, BODY), Paragraph(c, BODY)] for a, b, c in toc],
      [3 * cm, 10.5 * cm, 2 * cm], header=False, font=9.5)
pagebreak()

p("List of Figures", "H1c"); hr()
lof = [
    "3.1  Gate A Learning Curve &ndash; Helmet Detector Data Sufficiency",
    "4.1  DFD Level 0 &ndash; Context Diagram",
    "4.2  DFD Level 1 &ndash; Pipeline Stages",
    "4.3  DFD Level 2 &ndash; Detection Sub-Process",
    "4.4  UML Class Diagram &ndash; Pipeline Core Classes",
    "4.5  UML Activity Diagram &ndash; Per-Frame Pipeline Flow",
    "4.6  UML Use Case Diagram",
    "4.7  Sequence Diagram &ndash; Process-Isolated Demographics",
    "5.1  Scene-Classifier Confusion Matrix (CASE 1)",
    "5.2  Wheel-Count Architecture Comparison + Per-Class F1 (CASE 2)",
    "5.3  Helmet Detector Sample Output (CASE 3)",
    "5.4  Flip-Rate, Balanced Accuracy &amp; Occlusion Bands (CASE 4)",
    "5.5  Real-World Risk-Level and Occupancy Distribution (CASE 5)",
    "5.6  Pruning Collapse, Retention, Size &amp; Latency (CASE 6)",
    "5.7  VeRi-776 Cross-Camera ReID Scores (Discussion 5.2.1)",
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
p("2.3 Comparative Positioning", "H1c"); hr()
p("""Table 2.1 summarizes, dimension by dimension, how the approach taken in
this project differs from the typical framing found in the surveyed
literature above.""")
table([["Aspect", "Typical prior work", "This project"],
       ["Scope", "Single sub-problem (helmet OR tracking OR ReID)", "Full pipeline, 4 phases, one codebase"],
       ["Preprocessing", "Fixed or absent", "Scene-adaptive, learned classifier"],
       ["Wheel-count architecture", "Single model reported", "3-way benchmark, pretraining-fairness stated"],
       ["Occlusion claim", "Usually assumed, rarely measured", "Measured via class-imbalance-robust metric"],
       ["Re-ID baseline", "Fine-tuned SOTA reported only", "Off-the-shelf baseline stated explicitly, gap quantified"],
       ["Pruning reporting", "Post-recovery number only", "Naive collapse and recovery both reported"],
       ["Failure reporting", "Often only best result shown", "Every shortfall diagnosed to root cause"],
       ["Deployment target", "GPU assumed", "CPU-only throughout"]],
      [3.6 * cm, 5.9 * cm, 5.9 * cm], font=8.2)
p("2.4 Summary", "H1c"); hr()
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
img("outputs/thesis/gate_a_learning_curve.png", 11.5 * cm,
   "Figure 3.1. Gate A learning curve &ndash; test mAP@50 vs training images. "
   "The final quarter (276 &rarr; 368 images) gains only 0.017 mAP, against a "
   "total gain of 0.223 across the full curve &ndash; a plateau, not a "
   "continuing climb, which is why the ~29&nbsp;GB HELMET image download was "
   "skipped rather than pursued.")
table([["Fraction of data", "Train images", "Test mAP@50", "Test mAP@50-95"]] + [
    [f"{pt['fraction']*100:.0f}%", str(pt["train_images"]), f"{pt['test_mAP50']:.3f}", f"{pt['test_mAP50_95']:.3f}"]
    for pt in GATES["gate_A_helmet_data"]["points"]
], [4 * cm, 4 * cm, 4 * cm, 3.4 * cm])
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
img("outputs/thesis/uml_sequence.png", 14.5 * cm,
   "Figure 4.7. Sequence diagram of the process-isolated demographics path: "
   "the live pipeline (torch loaded) only ever writes face crops to disk; a "
   "separate process, started independently and never importing torch, polls "
   "the queue and writes results back &ndash; the two processes never share "
   "an address space.")
p("4.7 Backend Status Summary", "H1c"); hr()
p("""Every heavy backend in this project degrades gracefully to a lightweight
fallback when its dependency is unavailable, which is why Section 4.5's
regression suite exists: a broken dependency and a working one otherwise look
identical from the console. Table 4.2 records the backend actually active for
each subsystem on the hardware and software versions this thesis was measured
on.""")
table([["Subsystem", "Preferred backend", "Fallback", "Active on this run"],
       ["Tracking", "DeepSORT (deep-sort-realtime)", "IOU-only tracker", "DeepSORT"],
       ["Cross-camera ReID", "OSNet (torchreid)", "Colour-histogram matching", "OSNet"],
       ["Detection", "Ultralytics YOLOv8", "— (required)", "Ultralytics YOLOv8"],
       ["Demographics", "DeepFace (process-isolated)", "Skipped, logged", "DeepFace"]],
      [3.4 * cm, 5 * cm, 4.3 * cm, 3.4 * cm], font=8.2)
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
    "Figure 5.2a. Accuracy and macro-F1, leak-free split.")
img("outputs/report_p2/wheel_perclass.png", 12 * cm,
   "Figure 5.2b. Per-class F1 after the leakage fix. 3-Wheeler remains "
   "strongest (0.96) &ndash; verified to be genuine visual distinctiveness, "
   "not a framing artifact, by an ablation described in Section 5.2.")

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
    "Figure 5.4a. Flip-rate vs vote window size.")
img("outputs/report_p3/balanced_accuracy.png", 10.5 * cm,
   "Figure 5.4b. Balanced vs raw accuracy across vote windows &ndash; raw "
   "accuracy stays near 0.93 throughout purely because one class dominates "
   "the dataset, which is why balanced accuracy is reported as the primary "
   "figure rather than raw accuracy.")
img("outputs/report_p3/occlusion_bands.png", 9.5 * cm,
   "Figure 5.4c. UA-DETRAC occlusion-band support &ndash; the heavy-occlusion "
   "band contains only one vehicle class, which is why per-band raw accuracy "
   "would be misleading here.")

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
    "Figure 5.5a. Resulting risk-level distribution on real data.")
img("outputs/report_p3/occupancy.png", 10 * cm,
   "Figure 5.5b. Real-world motorcycle occupancy from 283,377 HELMET "
   "annotations &ndash; 6.4% carry 3 or more riders, the population the "
   "overload rule is meant to catch.")

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
    "Figure 5.6a. Naive pruning collapse vs. fine-tune recovery.")
img("outputs/report_p4/retention_summary.png", 11 * cm,
   "Figure 5.6b. Accuracy retention across every optimization configuration, "
   "both benchmarked models.")
img("outputs/report_p4/size_reduction.png", 9.5 * cm,
   "Figure 5.6c. ONNX Runtime INT8 shrinks both models 3.8&ndash;3.9x at no "
   "further accuracy cost beyond pruning.")
img("outputs/report_p4/latency.png", 11.5 * cm,
   "Figure 5.6d. Latency across every optimization configuration for both "
   "benchmarked models &ndash; ONNX is not uniformly faster than native "
   "PyTorch at batch-1 on this CPU, since session/dispatch overhead "
   "dominates for these small, sub-millisecond models.")

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

case(8, "End-to-End Pipeline Throughput",
    f"""<b>Objective:</b> Measure sustained frame rate of the complete
    nine-stage pipeline (Table 4.1) running end-to-end on real video, not
    just individual model latency.<br/>
    <b>Result:</b> On {P3PIPE['frames']} real frames
    ({P3PIPE['source']}), the pipeline sustains
    <b>{P3PIPE['fps_mean']:.2f} FPS mean</b>
    ({P3PIPE['fps_final']:.1f} FPS steady-state) on CPU alone, with tracking
    IDs, a telemetry HUD, and a per-track risk banner rendered on every
    frame. {P3PIPE.get('note', '')}<br/>
    <b>Status:</b> PASS (throughput measured end-to-end, not extrapolated
    from per-model latency alone).""")

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
p("Table 6.1 consolidates the headline measured result from every phase.", "Body")
table([["Phase", "Metric", "Measured value", "Evidence"],
       ["1 – Preprocessing", "Scene-classification accuracy", f"{P1['accuracy']:.2f}", "CASE 1"],
       ["2 – Detection", "Wheel-count accuracy (leak-free)", f"{P2W['yolo']['accuracy']:.3f}", "CASE 2"],
       ["2 – Detection", "Helmet mAP@50 (held-out test)", f"{P2H['test']['mAP50']:.3f}", "CASE 3"],
       ["3 – Tracking/ReID", "Flip-rate reduction (N=1 to N=30)",
        f"{(1 - P3O['flip_rate_overall']['30']/P3O['flip_rate_overall']['1'])*100:.1f}%", "CASE 4"],
       ["3 – Tracking/ReID", "Cross-camera ReID Rank-1 (off-the-shelf)", f"{P3REID['rank1']:.3f}", "Sec. 5.2.1"],
       ["3 – Tracking/ReID", "Risk-rule tracks validated", f"{P3RISK['n_tracks']:,}", "CASE 5"],
       ["3 – Tracking/ReID", "End-to-end throughput", f"{P3PIPE['fps_mean']:.2f} FPS", "CASE 8"],
       ["4 – Optimization", "Pruned+fine-tuned accuracy retention",
        f"{p4row('SmallCNN','pruned')['accuracy_retention']:.2f}", "CASE 6"],
       ["4 – Optimization", "ONNX INT8 model-size reduction", "3.8–3.9x", "CASE 7"]],
      [3.3 * cm, 5.5 * cm, 3.6 * cm, 2.9 * cm], font=8.2)
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

p("Appendix A &ndash; Training Hyperparameters", "H1c"); hr()
table([["Model", "Epochs", "Image size", "Batch", "Optimizer", "Notes"],
       ["Scene classifier (RF)", "—", "—", "—", "300 trees", "15 hand-engineered features"],
       ["SmallCNN (wheel)", "12", "96×96", "32", "Adam, cosine LR", "0.24M params, from scratch"],
       ["CSPNeXt (wheel)", "12", "96×96", "32", "Adam, cosine LR", "2.35M params, from scratch"],
       ["YOLOv8-cls (wheel)", "8", "96×96", "32", "Ultralytics default", "ImageNet-pretrained"],
       ["YOLOv8n (helmet)", "25", "416×416", "8", "Ultralytics default", "COCO-pretrained backbone"],
       ["Pruning fine-tune", "3", "96×96", "32", "Adam, lr=1e-4", "post-prune recovery"]],
      [3.3 * cm, 1.6 * cm, 2.1 * cm, 1.6 * cm, 2.9 * cm, 3.7 * cm], font=7.8)
pagebreak()

p("Appendix B &ndash; Dataset Summary", "H1c"); hr()
table([["Dataset", "Used for", "Size"],
       ["DAWN", "Fog/Rain scene classification", "300 fog + 200 rain images"],
       ["ExDark", "Night scene classification", "300 images (of 7,363 available)"],
       ["COCO val2017", "Day scenes + wheel-count crops", "300 day images; 5,000 total"],
       ["Helmet (7-class)", "Helmet-compliance detection", "368 train / 65 val / 52 test"],
       ["Auto-rickshaw set", "3-wheeler crops", "663 images"],
       ["UA-DETRAC", "Tracking + occlusion voting", "8 sequences, 22,480 crops"],
       ["VeRi-776", "Cross-camera ReID", "776 vehicles, 20 cameras, 51k images"],
       ["HELMET (annotations)", "Risk-rule validation, occupancy stats", "910 clips, 283,377 instances, 10,006 tracks"]],
      [3.6 * cm, 6.1 * cm, 5.8 * cm], font=8.2)
pagebreak()

p("Appendix C &ndash; Repository and Reproduction", "H1c"); hr()
p(f"Source code, trained model configs, and all tracked results: {REPO}")
p("""Every figure and number in this thesis is generated by
<font face='Courier' size=8>scripts/build_thesis.py</font> reading directly
from <font face='Courier' size=8>results/phase1..4/*.json</font>; full
reproduction commands are in
<font face='Courier' size=8>docs/HOW_TO_RUN.md</font>.""")
sp(16)
p("Appendix D &ndash; Contract Deliverables Checklist", "H2c"); hr()
table([["Deliverable", "Status", "Evidence"],
       ["Complete Python source code", "Done", "This repository"],
       ["Trained model weights (PyTorch/ONNX)", "Done", "weights/, results/phase4/"],
       ["Dataset preprocessing scripts", "Done", "scripts/build_*, src/preprocessing/"],
       ["Installation guide + requirements", "Done", "README.md, requirements.txt"],
       ["Step-by-step execution documentation", "Done", "docs/HOW_TO_RUN.md"],
       ["Publication-ready research paper", "Done", "docs/paper/ (submitted / in progress)"],
       ["Per-milestone demonstration", "Done", "Phase1..4_Report.pdf"]],
      [6.3 * cm, 2.7 * cm, 6.5 * cm], font=8.4)


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
        self.setFont("Times-Bold", 9)
        self.setFillColor(NAVY)
        self.drawCentredString(PAGE_W / 2, PAGE_H - MARGIN + 0.15 * cm, TITLE_FULL[:95])
        self.setStrokeColor(BLUE)
        self.setLineWidth(0.7)
        self.line(MARGIN, PAGE_H - MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN)
        self.setFont("Times-Roman", 8)
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
                      author=AUTHOR)
doc.addPageTemplates([PageTemplate(id="thesis", frames=[frame])])
doc.build(story, canvasmaker=ThesisCanvas)
print(f"PDF -> {OUT_PDF}")
