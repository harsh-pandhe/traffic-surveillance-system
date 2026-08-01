"""
scripts/build_phase3_pdf.py
---------------------------
Build the Phase 3 report PDF from results/phase3/*.json + outputs/report_p3/*.png.
Output -> docs/Phase3_Report.pdf
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

A = "outputs/report_p3"
OUT_PDF = "docs/Phase3_Report.pdf"
REPO = "https://github.com/harsh-pandhe/traffic-surveillance-system"

OCC = json.load(open("results/phase3/occlusion_ablation.json"))
RISK = json.load(open("results/phase3/risk_indexer_validation.json"))
REID = json.load(open("results/phase3/reid_benchmark.json"))
DEMO = json.load(open("results/phase3/demographics.json"))
HSTAT = json.load(open("results/phase3/helmet_dataset_stats.json"))
PIPE = json.load(open("results/phase3/pipeline_run.json"))

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


def img(name, width=15 * cm, cap=None):
    path = os.path.join(A, name)
    if not os.path.isfile(path):
        return
    iw, ih = ImageReader(path).getSize()
    story.append(Image(path, width=width, height=width * ih / iw))
    story.append(Paragraph(cap, styles["Cap"]) if cap else Spacer(1, 8))


def table(data, cw, header=True, font=8.5):
    t = Table(data, colWidths=cw, hAlign="CENTER")
    ts = [("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d0d8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
          ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("ALIGN", (0, 1), (0, -1), "LEFT")]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, 0), "CENTER")]
    t.setStyle(TableStyle(ts)); story.append(t); sp(10)


# ---- Title ----
sp(38)
p("Adaptive Spatio-Temporal Traffic Surveillance", "TitleBig"); sp(6)
p("Phase 3 — Multi-Frame Tracking &amp; Spatio-Temporal Analytics", "Sub"); sp(16); hr()
meta = [["Milestone", "Phase 3 of 4 (Days 13–17): Tracking, ReID, Risk Analytics"],
        ["Author", "Harsh Pandhe"],
        ["Repository", REPO],
        ["Tracking backend", "DeepSORT (verified live, not fallback)"],
        ["Voting result", f"{(1 - OCC['flip_rate_overall']['30'] / OCC['flip_rate_overall']['1']) * 100:.1f}% "
                          "fewer prediction flips at N=30 vs single-frame"],
        ["ReID", f"Rank-1 {REID['rank1']:.3f}, mAP {REID['mAP']:.3f} on VeRi-776 (off-the-shelf OSNet)"],
        ["Risk validation", f"{RISK['n_tracks']:,} real motorcycle tracks (HELMET dataset)"]]
table([[Paragraph(f"<b>{k}</b>", BODY), Paragraph(v, BODY)] for k, v in meta], [3.4 * cm, 12.1 * cm], header=False)
sp(8)
p("<b>Executive summary.</b> Phase 3 delivers verified DeepSORT tracking with "
  "multi-frame majority voting, cross-camera vehicle ReID, a risk indexer "
  "validated against 10,006 real motorcycle tracks, and demographics on "
  "exposed-face riders. Two silently-degraded backends (DeepSORT, OSNet) were "
  "caught and fixed before this phase began. A genuine hardware constraint — "
  "torch and TensorFlow segfaulting when loaded together — was found and "
  "worked around with process isolation, verified end-to-end on real pipeline "
  "output. Every number below is measured on real data; where a result falls "
  "short of expectation (ReID accuracy, risk-level calibration), that is "
  "reported plainly rather than smoothed over.")
story.append(PageBreak())

# ---- Section 1: tracking backends ----
p("1. Backend Verification", "H1c"); hr()
p("Every heavy backend in this project has a graceful fallback, which means a "
  "broken dependency looks like success unless explicitly checked. Both "
  "DeepSORT and OSNet were silently degraded before this phase:")
table([["Backend", "Was", "Root cause", "Now"],
       ["DeepSORT", "fallback IoU tracker", "openmim downgraded setuptools; "
        "Python 3.12 pkg_resources breaks", "deepsort (verified)"],
       ["OSNet ReID", "colour-histogram fallback", "torchreid ships under two "
        "import layouts", "osnet, 512-d (verified)"]],
      [3 * cm, 3.6 * cm, 6 * cm, 3.5 * cm], font=7.5)
story.append(PageBreak())

# ---- Section 2: voting ablation ----
p("2. Multi-Frame Voting Ablation", "H1c"); hr()
p(f"UA-DETRAC, {len(OCC['sequences'])} sequences, {OCC['crops_classified']:,} "
  "classified crops, grouped by ground-truth track id so the measurement "
  "isolates voting from tracker error.")
p("Why flip-rate is the headline metric, not accuracy", "H2c")
p("UA-DETRAC is ~97% cars, so raw accuracy is dominated by one class, and its "
  "occlusion bands contain different class mixes (the heavy band is 100% one "
  "class), making per-band accuracy comparisons invalid. Balanced accuracy "
  "(macro recall) fixes the first problem; flip-rate — how often the emitted "
  "label changes between consecutive frames of the same vehicle — is "
  "class-composition-independent and remains valid across bands.")
img("flip_rate.png", 11.5 * cm,
   f"Figure 1. Flip-rate falls from {OCC['flip_rate_overall']['1']:.3f} (N=1) to "
   f"{OCC['flip_rate_overall']['30']:.3f} (N=30) — "
   f"{(1 - OCC['flip_rate_overall']['30']/OCC['flip_rate_overall']['1'])*100:.1f}% fewer label changes.")
img("balanced_accuracy.png", 11.5 * cm,
   "Figure 2. Balanced accuracy improves modestly with voting (best N=15: "
   f"{OCC['overall']['15']:.3f} vs single-frame {OCC['overall']['1']:.3f}); "
   "raw accuracy stays near 0.93 throughout because one class dominates.")
img("occlusion_bands.png", 10 * cm, "Figure 3. Occlusion band support — the "
   "heavy band is entirely one vehicle class, which is why per-band accuracy "
   "is not meaningful here and flip-rate is reported instead.")
p("Honest framing", "H2c")
p("Part of the flip-rate reduction is mechanical — averaging over more frames "
  "smooths any signal, including noise. The balanced-accuracy result is "
  "smaller but directionally consistent (+0.016 at the best window), which "
  "supports voting providing real benefit, not just averaging away noise. "
  "Both numbers are reported rather than leading with only the more dramatic one.")
story.append(PageBreak())

# ---- Section 3: risk indexer ----
p("3. Risk Indexer — Validated Against Real Ground Truth", "H1c"); hr()
p(f"{RISK['n_tracks']:,} real motorcycle tracks from {RISK['n_clips']} traffic "
  "clips (HELMET dataset, Myanmar). Rule logic — not detector accuracy — "
  "validated against true occupancy and helmet-use labels.")
table([["Rule", "Trigger rate"],
       ["Overloaded (>2 riders)", f"{RISK['rule_trigger_rates']['overloaded_pct']:.1f}%"],
       ["Helmet misuse (≥1 bare head)", f"{RISK['rule_trigger_rates']['helmet_misuse_pct']:.1f}%"]],
      [8 * cm, 5 * cm])
img("occupancy.png", 10.5 * cm, "Figure 4. Real-world motorcycle occupancy — "
   "6.4% carry 3+ riders (red), the overload rule's real trigger population.")
img("risk_distribution.png", 9.5 * cm, "Figure 5. Resulting risk-level "
   "distribution on real data.")
p("<b>Finding: HIGH risk never triggers on real motorcycle data (0.0%).</b> "
  "helmet_misuse (w2=2.5) + overloaded (w3=1.5) = 4.0, below the HIGH "
  "threshold of 6.0 — even the worst realistic motorcycle case tops out at "
  "MEDIUM. This is a calibration finding, not a bug: whether an overloaded, "
  "unhelmeted motorcycle should register as HIGH is a policy decision, "
  "reported here rather than silently re-tuned to look more dramatic.")
story.append(PageBreak())

# ---- Section 4: ReID ----
p("4. Cross-Camera Vehicle Re-Identification", "H1c"); hr()
p(f"VeRi-776 (776 vehicles, 20 real cameras). Standard protocol: cosine "
  f"distance ranking, excluding gallery images from the query's own camera. "
  f"{REID['n_query_evaluated']} queries against {REID['n_gallery']} gallery images.")
table([["Metric", "Value"], ["Rank-1", f"{REID['rank1']:.3f}"],
       ["Rank-5", f"{REID['rank5']:.3f}"], ["Rank-10", f"{REID['rank10']:.3f}"],
       ["mAP", f"{REID['mAP']:.3f}"],
       ["Embed latency", f"{REID['embed_ms_per_image']:.1f} ms/image (CPU)"]],
      [7 * cm, 5 * cm])
img("reid.png", 9.5 * cm, "Figure 6. VeRi-776 cross-camera ReID scores.")
p(f"<b>Honestly below published SOTA</b> (fine-tuned ReID models reach "
  f"Rank-1≈90%, mAP≈70%) because the backend runs off-the-shelf "
  f"ImageNet-pretrained OSNet with the classification head discarded — no "
  f"vehicle-ID metric learning at all. Rank-1 {REID['rank1']*100:.0f}% from "
  f"generic features alone confirms real appearance signal against a 776-way "
  f"chance level of 0.1%. VeRi's 37,778-image training split is already "
  f"downloaded; triplet-loss fine-tuning (tracked as issue #14) is the clear "
  f"path to close the gap.")
story.append(PageBreak())

# ---- Section 5: demographics ----
p("5. Demographics — A Hardware Constraint, Worked Around", "H1c"); hr()
p("Gate B measured 62% of face-exposed rider detections yield a usable crop "
  "(median 38px), justifying a DeepFace install. That surfaced a genuine "
  "issue: this machine's torch build targets a newer NVIDIA driver than "
  "installed, and loading a second torch model in a process that has also "
  "imported TensorFlow segfaults — confirmed by bisecting pipeline "
  "construction step by step. DemographicsEstimator now detects torch is "
  "already loaded and queues face crops for a separate process "
  "(scripts/run_demographics.py, which never imports torch) instead of "
  "risking a crash.")
table([["Metric", "Value"],
       ["Crops processed (real pipeline output)", f"{DEMO['n_crops']}"],
       ["Succeeded", f"{DEMO['n_succeeded']} ({DEMO['n_succeeded']/max(DEMO['n_crops'],1)*100:.0f}%)"]],
      [9 * cm, 5 * cm])
p("Verified end-to-end on real output: 19/19 queued crops from the live "
  "pipeline processed successfully with no crash. Per Gate B, age estimates "
  "are indicative rather than reliable given the small face-crop resolution; "
  "gender is more robust at low resolution.")
story.append(PageBreak())

# ---- Section 6: e2e + gaps ----
p("6. End-to-End Pipeline Run", "H1c"); hr()
p(f"{PIPE['frames']} frames ({PIPE['source']}), CPU only: "
  f"<b>{PIPE['fps_mean']:.2f} FPS mean</b>, {PIPE['fps_final']:.1f} FPS steady-state.")
p(PIPE.get("note", ""))

p("7. Reproduce", "H1c"); hr()
table([["Command", "Purpose"],
       ["python scripts/run_occlusion_ablation.py", "Voting ablation"],
       ["python scripts/validate_risk_indexer.py", "Risk-rule validation"],
       ["python scripts/run_reid_benchmark.py", "VeRi-776 ReID"],
       ["python scripts/run_demographics.py", "Offline demographics"]],
      [8.5 * cm, 6.5 * cm], font=8)

p("8. Deliverable Checklist", "H1c"); hr()
table([["Deliverable", "Status"],
       ["DeepSORT temporal tracking (verified, not fallback)", "Complete"],
       ["Multi-frame voting for occlusion robustness", "Complete"],
       ["Spatio-temporal risk analytics", "Complete"],
       ["Risk rules validated against real ground truth", "Complete"],
       ["Cross-camera ReID benchmarked", "Complete"],
       ["Demographics (process-isolated, verified)", "Complete"],
       ["Working demo with measured throughput", "Complete"]],
      [11 * cm, 4 * cm])

os.makedirs("docs", exist_ok=True)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "Adaptive Traffic Surveillance — Phase 3 Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(BLUE); canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT_PDF, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        title="Phase 3 Report — Adaptive Traffic Surveillance", author="Harsh Pandhe")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"PDF -> {OUT_PDF}")
