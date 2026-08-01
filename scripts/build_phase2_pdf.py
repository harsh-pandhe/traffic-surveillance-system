"""
scripts/build_phase2_pdf.py
---------------------------
Build the Phase 2 report PDF from results/phase2/*.json + outputs/report_p2/*.png.
Output -> docs/Phase2_Report.pdf
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
                                TableStyle, Image, PageBreak, HRFlowable)

A = "outputs/report_p2"
OUT_PDF = "docs/Phase2_Report.pdf"
REPO = "https://github.com/harsh-pandhe/traffic-surveillance-system"
W = json.load(open("results/phase2/wheel_metrics.json"))

def _normalise_helmet(h):
    """Accept both the old flat schema and the new {val,test} schema."""
    if "test" in h and isinstance(h["test"], dict):
        out = dict(h)
        out.update(h["test"])          # headline = held-out test numbers
        out["_split"] = "test"
        return out
    h = dict(h); h["_split"] = "val"
    return h

H = json.load(open("results/phase2/helmet_metrics.json"))
H = _normalise_helmet(H)

NAVY = colors.HexColor("#1b2a4a"); BLUE = colors.HexColor("#2e86c1")
LIGHT = colors.HexColor("#eaf2f8"); GREY = colors.HexColor("#5d6d7e")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], textColor=NAVY, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], textColor=BLUE, fontSize=13, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], alignment=TA_JUSTIFY, fontSize=9.5, leading=14, spaceAfter=6))
styles.add(ParagraphStyle("Cap", parent=styles["Normal"], alignment=TA_CENTER, fontSize=8.5, textColor=GREY, spaceAfter=10))
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], textColor=NAVY, fontSize=22, leading=26))
styles.add(ParagraphStyle("Sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, textColor=GREY))
BODY = styles["Body"]
story = []


def p(t, s="Body"): story.append(Paragraph(t, styles[s]))
def sp(h=6): story.append(Spacer(1, h))
def hr(): story.append(HRFlowable(width="100%", thickness=0.8, color=BLUE, spaceBefore=4, spaceAfter=8))


def img(name, width=15*cm, cap=None):
    path = os.path.join(A, name)
    if not os.path.isfile(path): return
    iw, ih = ImageReader(path).getSize()
    story.append(Image(path, width=width, height=width*ih/iw))
    story.append(Paragraph(cap, styles["Cap"]) if cap else Spacer(1, 8))


def table(data, cw, header=True, font=8.5):
    t = Table(data, colWidths=cw, hAlign="CENTER")
    ts = [("FONTSIZE", (0,0),(-1,-1), font), ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c8d0d8")),
          ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("TOPPADDING",(0,0),(-1,-1),3),
          ("BOTTOMPADDING",(0,0),(-1,-1),3), ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
          ("ALIGN",(1,1),(-1,-1),"CENTER"), ("ALIGN",(0,1),(0,-1),"LEFT")]
    if header:
        ts += [("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
               ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ALIGN",(0,0),(-1,0),"CENTER")]
    t.setStyle(TableStyle(ts)); story.append(t); sp(10)


# ---- Title ----
sp(38)
p("Adaptive Spatio-Temporal Traffic Surveillance", "TitleBig"); sp(6)
p("Phase 2 — Model Training &amp; Granular Helmet Compliance", "Sub"); sp(16); hr()
meta = [["Milestone", "Phase 2 of 4 (Days 6–12): Model Training &amp; Helmet Compliance"],
        ["Author", "Harsh Pandhe"],
        ["Repository", REPO],
        ["Wheel model", f"{W['selected']} selected — "
                        f"{W[[k for k in ('cnn','cspnext','yolo') if W.get(k) and W[k]['model']==W['selected']][0]]['accuracy']*100:.1f}% "
                        f"accuracy (4 classes, 3 architectures compared)"],
        ["Helmet model", f"YOLOv8n — mAP@50 {H['mAP50']:.3f} (7 classes)"]]
table([[Paragraph(f"<b>{k}</b>", BODY), Paragraph(v, BODY)] for k, v in meta], [3.2*cm, 12.3*cm], header=False)
sp(8)
p(f"<b>Executive summary.</b> Phase 2 delivers trained models on real data. For "
  f"wheel-count classification, three architectures are benchmarked under an "
  f"identical, <b>leak-free</b> protocol across four classes (2/3/4/6+ wheelers): "
  f"a custom SmallCNN baseline ({W['cnn']['accuracy']:.3f}), RTMDet's CSPNeXt "
  f"backbone ({W['cspnext']['accuracy']:.3f}), and YOLOv8-cls "
  f"({W['yolo']['accuracy']:.3f}, selected). For helmet compliance, a YOLOv8n "
  f"detector reaches mAP@50 {H['mAP50']:.3f} on a <b>held-out test split</b> "
  f"never used for model selection. All metrics are recomputed from the datasets "
  f"and persisted in results/phase2/; the pretraining asymmetry between the arms "
  f"is stated explicitly rather than glossed over.")
p("<b>Correction notice.</b> An earlier version of this report gave 0.751 / "
  "0.683 / 0.867 for the three architectures. That run split the dataset by "
  "<i>crop</i> rather than by source image, so several vehicles from the same "
  "photo appeared on both sides of the split (76% / 91% / 65% of 2-, 4- and "
  "6+-wheeler crops came from such photos), and the 3-wheeler class used whole "
  "photos while every other class used tight crops. Both defects are fixed and "
  "the numbers above are the corrected ones. The model ranking is unchanged.")
story.append(PageBreak())

# ---- Part A ----
p("1. Wheel-Count Classification — CNN vs RTMDet vs YOLO", "H1c"); hr()
p("Four classes: 2-Wheeler (bicycle/motorcycle, COCO), 3-Wheeler (auto-rickshaw "
  "crops), 4-Wheeler (car), 6+ Wheeler (bus/truck). Balanced ~308–350 train "
  "images per class, 96×96 crops, CPU training.")
_arms = [k for k in ("cnn", "cspnext", "yolo") if W.get(k)]
_rows = [["Model","Acc","Macro-P","Macro-R","Macro-F1","Lat (ms)","Params","Pretrained"]]
for k in _arms:
    m = W[k]
    pm = f"{m['params_m']:.2f}M" if m.get("params_m") else "—"
    pre = "ImageNet" if "imagenet" in str(m.get("pretrained","")).lower() else "scratch"
    nm = m["model"] + (" (selected)" if m["model"] == W.get("selected") else "")
    _rows.append([nm, f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
                  f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                  f"{m['latency_ms']:.1f}", pm, pre])
table(_rows, [3.5*cm,1.7*cm,1.9*cm,1.9*cm,2.0*cm,1.7*cm,1.5*cm,1.8*cm], font=7.5)
img("wheel_compare.png", 11.5*cm, "Figure 1. Accuracy and macro-F1 across the three architectures.")
img("wheel_perclass.png", 13*cm, "Figure 2. Per-class F1 on the leak-free split. "
    "3-Wheeler remains strongest (0.96) even after cropping; car vs bus/truck is now the hard pair.")
img("wheel_composition.png", 10.5*cm, "Figure 3. Balanced 4-class wheel dataset.")
p("Methodological note", "H2c")
p(f"This comparison is <b>not pretraining-neutral</b>. YOLOv8-cls starts from "
  f"ImageNet weights; SmallCNN and CSPNeXt train from scratch, so part of YOLO's "
  f"margin is transfer learning rather than architecture. CSPNeXt (2.35M params) "
  f"has the least data per parameter and underfits at this dataset size, yet it "
  f"is the <b>fastest at inference</b> ({W['cspnext']['latency_ms']:.1f} ms — "
  f"essentially tied with the 10x smaller SmallCNN and ~3x faster than "
  f"YOLOv8-cls), because the depthwise 5x5 CSP design is CPU-efficient. On the "
  f"earlier dataset it was still improving with longer training (0.562 at 12 "
  f"epochs to 0.683 at 40), so its figure here is a floor, not a ceiling. For "
  f"this dataset size the pretrained YOLOv8-cls is the correct production choice.")
p("RTMDet implementation", "H2c")
p("mmdetection/mmcv (RTMDet's official home) will not install here: openmim "
  "fails on Python 3.12 (<font face='Courier' size=8>pkgutil.ImpImporter</font> "
  "removed) and mmcv ships no wheels for torch 2.12. RTMDet's architectural "
  "contribution — the CSPNeXt backbone — is therefore reimplemented directly in "
  "PyTorch (<font face='Courier' size=8>src/models/cspnext.py</font>) following "
  "the paper: depthwise 5x5 CSP blocks, channel attention, SiLU, SPPF, "
  "RTMDet-tiny scaling.")
story.append(PageBreak())

# ---- Part B ----
p("2. Granular Helmet Compliance — YOLOv8n", "H1c"); hr()
p("7-class rider/helmet detection (driver/passenger × helmet/no-helmet + bike), "
  "368 train / 65 val / 52 test, YOLO format. Fine-tuned YOLOv8n, 25 epochs, "
  "416×416, CPU. The figures below are on the <b>held-out test split</b>; the "
  "validation split was used for epoch selection and is therefore optimistically "
  "biased. Both splits are small, so both carry wide confidence intervals — they "
  "corroborate each other at roughly 0.73-0.76 rather than one superseding the "
  "other.")
table([["Metric","Value"],
       ["mAP@50", f"{H['mAP50']:.3f}"],["mAP@50-95", f"{H['mAP50_95']:.3f}"],
       ["Precision", f"{H['precision']:.3f}"],["Recall", f"{H['recall']:.3f}"]],
      [7*cm, 4*cm])
img("helmet_ap.png", 13*cm, "Figure 4. Per-class AP@50 (red = below the mAP line).")
img("helmet_sample.png", 12*cm, "Figure 5. Trained detector on a real test image — riders, bikes, and helmet status.")
p("Passenger-helmet classes are weakest (fewest instances, small objects) — "
  "tracked as follow-up issue #8.")
story.append(PageBreak())

# ---- reproduce + gaps ----
p("3. Reproduce", "H1c"); hr()
table([["Command","Purpose"],
       ["python scripts/build_wheel_dataset.py","Build 4-class wheel crops"],
       ["python scripts/train_wheel_classifier.py","Train + benchmark CNN vs YOLO"],
       ["python scripts/train_helmet_detector.py --data ...","Train + eval helmet detector"],
       ["python scripts/make_phase2_assets.py","Regenerate this report's figures"]],
      [8.4*cm, 7.1*cm])

p("4. Status &amp; Follow-ups", "H1c"); hr()
table([["Deliverable","Status"],
       ["Baseline CNN trained","Complete"],
       ["YOLO model trained (wheel + helmet)","Complete"],
       ["P/R/mAP comparison + selected model","Complete"],
       ["3-Wheeler class (auto-rickshaw)","Complete (issue closed)"],
       ["Granular helmet detector (7-class mAP)","Complete"],
       ["Low AP passenger-helmet classes","Open (issue #8)"]],
      [11.5*cm, 4.0*cm])
p("<b>Conclusion.</b> Phase 2 is complete: both models trained and benchmarked "
  "on real data, all four wheel classes active, and the granular helmet detector "
  "reporting per-class AP. Every figure is reproducible from the committed code.")


def _footer(cv, doc):
    cv.saveState(); cv.setFont("Helvetica", 7.5); cv.setFillColor(GREY)
    cv.drawString(2*cm, 1.1*cm, "Adaptive Traffic Surveillance — Phase 2 Report")
    cv.drawRightString(A4[0]-2*cm, 1.1*cm, f"Page {doc.page}")
    cv.setStrokeColor(BLUE); cv.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm); cv.restoreState()


os.makedirs("docs", exist_ok=True)
SimpleDocTemplate(OUT_PDF, pagesize=A4, topMargin=1.6*cm, bottomMargin=1.8*cm,
                  leftMargin=2*cm, rightMargin=2*cm,
                  title="Phase 2 Report", author="Harsh Pandhe").build(
    story, onFirstPage=_footer, onLaterPages=_footer)
print(f"PDF -> {OUT_PDF}")
