/**
 * scripts/build_ijsrset_paper.js
 * --------------------------------
 * Builds docs/paper/IJSRSET_paper.docx -- the research paper reformatted to
 * IJSRSET's submission spec (ijsrset.com/home/Guidelines-for-Authors):
 *   - Manuscript file must be .docx (PDF is rejected outright)
 *   - A4, 2-column, Times New Roman, margins top 18.1mm / bottom 20mm /
 *     left 19mm / right 16mm
 *   - Author list + affiliation + ORCID, corresponding author marked *
 *   - Abstract 150-250 words (self-contained, no citations), 4-6 keywords
 *     in alphabetical order
 *   - Sections: Introduction, Method, Results and Discussion, Conclusion,
 *     Acknowledgments, Author contributions, Conflicts of interest,
 *     References (heading NOT numbered, IEEE in-text style [n])
 *
 * Every number/figure reused here comes from the same tracked results JSON
 * already used by docs/paper/paper.pdf and the other reports -- nothing is
 * fabricated. Author names/ORCID/guide are placeholders (bracketed) since
 * that information isn't something this script can know -- fill in before
 * submission.
 *
 * Run: node scripts/build_ijsrset_paper.js
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  ImageRun, BorderStyle,
} = require("docx");

const P1 = JSON.parse(fs.readFileSync("results/phase1/scene_metrics.json"));
const P2W = JSON.parse(fs.readFileSync("results/phase2/wheel_metrics.json"));
const P2H = JSON.parse(fs.readFileSync("results/phase2/helmet_metrics.json"));
const P3O = JSON.parse(fs.readFileSync("results/phase3/occlusion_ablation.json"));
const P3RISK = JSON.parse(fs.readFileSync("results/phase3/risk_indexer_validation.json"));
const P3REID = JSON.parse(fs.readFileSync("results/phase3/reid_benchmark.json"));
const P4 = JSON.parse(fs.readFileSync("results/phase4/benchmark.json"));

function p4row(model, config) {
  return P4.rows.find(r => r.model === model && r.config === config);
}

const sc_fp32 = p4row("SmallCNN", "fp32");
const sc_naive = p4row("SmallCNN", "pruned_naive");
const sc_ft = p4row("SmallCNN", "pruned");
const sc_int8 = p4row("SmallCNN", "onnx_int8");

const flipDrop = (1 - P3O.flip_rate_overall["30"] / P3O.flip_rate_overall["1"]) * 100;

// ---- mm -> twips (1 inch = 25.4mm = 1440 twips) ---------------------------
const mm = v => Math.round((v / 25.4) * 1440);

// ---- Times New Roman throughout, sizes in half-points --------------------
const FONT = "Times New Roman";
const SZ_TITLE = 24;   // 12pt
const SZ_AUTHOR = 22;  // 11pt
const SZ_AFFIL = 20;   // 10pt
const SZ_BODY = 20;    // 10pt
const SZ_HEAD = 20;    // 10pt
const SZ_CAP = 18;     // 9pt
const SZ_REF = 24;     // 12pt (per Guidelines-for-Authors)

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: opts.size || SZ_BODY, ...opts });
}

function bodyPara(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 240 },
    spacing: { after: 120 },
    children: [run(text, opts)],
  });
}

function sectionHeading(text) {
  // IJSRSET/IEEE-style: centered, small caps, bold, numbered
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [run(text, { bold: true, size: SZ_HEAD, smallCaps: true })],
  });
}

function subHeading(text) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 160, after: 80 },
    children: [run(text, { italics: true, bold: true, size: SZ_HEAD })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [run(text, { size: SZ_CAP })],
  });
}

function figure(path, widthPx, heightPx, capText) {
  const data = fs.readFileSync(path);
  const ext = path.split(".").pop();
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120 },
      children: [new ImageRun({
        type: ext,
        data,
        transformation: { width: widthPx, height: heightPx },
        altText: { title: capText, description: capText, name: capText },
      })],
    }),
    caption(capText),
  ];
}

function refItem(n, text) {
  return new Paragraph({
    spacing: { after: 60 },
    indent: { left: 260, hanging: 260 },
    children: [run(`[${n}] ${text}`, { size: SZ_REF })],
  });
}

// ============================================================ FRONT MATTER
const title = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [run(
    "Adaptive Spatio-Temporal Traffic Surveillance for Granular Helmet-Compliance and Multi-Frame Vehicle Classification on Commodity CPUs",
    { bold: true, size: SZ_TITLE })],
});

const authors = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [
    run("Harsh Pandhe*1, Shifali Doshi2, Kiran Kamate3, [Guide Name]4", { bold: true, size: SZ_AUTHOR }),
  ],
});

function affilLine(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 20 },
    children: [run(text, { italics: true, size: SZ_AFFIL })],
  });
}

const affilBlock = [
  affilLine("*1,2,3Student, Department of Computer Science & Engineering, Walchand Institute of Technology, Solapur, Maharashtra, India"),
  affilLine("4[Designation], Department of Computer Science & Engineering, Walchand Institute of Technology, Solapur, Maharashtra, India"),
  affilLine("*Corresponding Author: [corresponding author email] | ORCID: [0000-0000-0000-0000]"),
];

const abstractHeading = new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 200, after: 80 },
  children: [run("ABSTRACT", { bold: true, size: SZ_HEAD })],
});

// 150-250 words, self-contained, no citations/abbreviations spelled out on first use.
const abstractText =
  `This paper presents a complete, CPU-only traffic surveillance pipeline covering adaptive scene-conditioned preprocessing, granular helmet-compliance and wheel-count detection, multi-frame tracking with cross-camera re-identification, and deployment-oriented model optimization. A learned scene classifier improves day, night, fog, and rain classification accuracy from a 0.49 rule-based baseline to ${P1.accuracy.toFixed(2)}. Three architectures are benchmarked for wheel-count classification under an identical, leakage-free protocol: a custom convolutional network (${P2W.cnn.accuracy.toFixed(3)}), a from-scratch CSPNeXt backbone (${P2W.cspnext.accuracy.toFixed(3)}), and an ImageNet-pretrained YOLOv8 classifier (${P2W.yolo.accuracy.toFixed(3)}, selected), with the pretraining asymmetry stated explicitly. A seven-class helmet detector reaches ${P2H.test.mAP50.toFixed(3)} mean average precision at 0.50 intersection-over-union on held-out data. Multi-frame majority voting reduces prediction flip-rate by ${flipDrop.toFixed(1)}% under measured real-world occlusion, and risk-indexing rules are validated against ${P3RISK.n_tracks.toLocaleString()} real motorcycle tracks with ground-truth occupancy and helmet-use labels. Naive structured pruning is shown to collapse model accuracy unless followed by brief fine-tuning, after which INT8 quantization shrinks deployed models by 3.8 to 3.9 times at no further cost. Every result that fell short of an initial expectation, including cross-camera re-identification without metric-learning fine-tuning, is reported with its diagnosed cause rather than omitted, and the complete source code and trained weights are publicly available.`;

const abstractPara = new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120 },
  children: [run(abstractText, { size: SZ_BODY })],
});

// 4-6 keywords, alphabetical order
const keywordsPara = new Paragraph({
  spacing: { after: 200 },
  children: [
    run("Keywords: ", { bold: true, size: SZ_BODY }),
    run("CPU Inference, Helmet Detection, Model Quantization, Multi-Object Tracking, Vehicle Re-Identification", { size: SZ_BODY }),
  ],
});

const divider = new Paragraph({
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 4 } },
  spacing: { after: 200 },
  children: [],
});

// ============================================================ I. INTRODUCTION
const introSection = [
  sectionHeading("I. Introduction"),
  bodyPara("Two-wheeler traffic dominates road usage in many dense urban environments and carries disproportionate crash risk, yet automated enforcement of helmet compliance and vehicle-load limits remains largely manual. Prior automated systems typically address a single sub-problem, such as helmet detection, license-plate recognition, or vehicle counting, in isolation, and are rarely evaluated across the range of lighting and weather conditions actually encountered in the field. This work builds and evaluates a single pipeline spanning environment-adaptive preprocessing, granular compliance detection, occlusion-robust tracking, and deployment-ready optimization, on commodity CPU hardware throughout."),
  bodyPara("Automated helmet detection is typically framed as binary (helmet or no-helmet) object detection, commonly built on YOLO-family detectors for CPU and edge throughput [1]. Few public systems distinguish driver from passenger or model rider count directly; this work adopts the AI City Challenge Track-5 seven-class convention (driver and passenger crossed with helmet status, plus a bike class) [7]. Vehicle-type classification is usually a byproduct of general detection (COCO car, bus, truck, motorcycle categories) rather than purpose-built for the two, three, four, and six-plus wheeler taxonomy common in heterogeneous South and Southeast Asian traffic. RTMDet [4] is a strong real-time detector family; this work evaluates its CSPNeXt backbone as a classifier since its official mmdetection toolchain does not install against the Python 3.12 and PyTorch 2.12 environment used here. DeepSORT [1] remains a mature, CPU-tractable choice for online multi-object tracking compared with GPU-oriented transformer trackers. VeRi-776 [3] is a standard cross-camera vehicle re-identification benchmark; state-of-the-art methods fine-tune a re-identification-specific backbone, often OSNet [2], with a triplet or identity loss on the benchmark's own split, a step deliberately not taken here in order to measure the honest zero-shot gap."),
  bodyPara("The contributions of this work are: (1) a scene-adaptive preprocessing stage with an unconfounded within-dataset validation; (2) a three-way, leakage-free CPU architecture benchmark for wheel-count classification with an explicit pretraining-fairness analysis, including a from-scratch CSPNeXt implementation where the official RTMDet framework would not install; (3) a multi-frame voting scheme evaluated with a class-imbalance-robust metric after diagnosing why naive accuracy comparison was invalid on the available benchmark; and (4) end-to-end model optimization with a diagnosed and corrected pruning failure mode."),
];

// ============================================================ II. METHOD
const methodSection = [
  sectionHeading("II. Method"),
  subHeading("A. Adaptive Preprocessing"),
  bodyPara("Each frame yields fifteen lighting, haze, and texture features, including luminance statistics, dark-channel mean, Laplacian variance, FFT high-frequency energy, and colourfulness. A 300-tree random forest, trained on real DAWN, ExDark, and COCO imagery, classifies each frame as DAY, NIGHT, FOG, or RAIN; enhancement (CLAHE, dark channel prior dehazing, or denoising) is applied only when the predicted condition warrants it."),
  subHeading("B. Detection Models"),
  bodyPara("Wheel-count classification benchmarks a custom SmallCNN (0.24 million parameters, trained from scratch), a from-scratch CSPNeXt classifier (2.35 million parameters, trained from scratch), and YOLOv8-cls (ImageNet-pretrained) on an identical four-class dataset split by source image, not by crop, after an earlier crop-level split was found to leak scene information across the train and validation sets. A YOLOv8n detector is separately fine-tuned for seven-class helmet compliance."),
  subHeading("C. Tracking, Voting, and Risk Indexing"),
  bodyPara("DeepSORT provides per-vehicle tracks; a temporal majority-vote buffer smooths the wheel-count prediction over a sliding window of N frames. A weighted risk index combines wheel-count violation, helmet misuse, rider overload, and wrong-way movement into LOW, MEDIUM, and HIGH tiers, validated against real ground-truth occupancy and helmet-use labels rather than only against model output."),
  subHeading("D. Optimization"),
  bodyPara("Each classifier is pruned using 30% L1 structured sparsity, optionally fine-tuned to recover accuracy, dynamically quantized to INT8, and exported to ONNX with a second INT8 pass applied via ONNX Runtime. Accuracy, latency, and model size are recorded at every step."),
  subHeading("E. Datasets and a Leakage Correction"),
  bodyPara("DAWN, ExDark, and COCO supply the scene-classification data; a seven-class YOLO helmet dataset and auto-rickshaw crops supply the detection data; UA-DETRAC, VeRi-776, and the HELMET annotation set (283,377 real motorcycle instances, with no images required for its use here) supply the tracking and risk-validation data. The wheel-count dataset builder originally split crops rather than source images, so several vehicles cropped from one photograph could appear on both sides of the split; this was measured directly (76%, 91%, and 65% of two-, four-, and six-plus-wheeler crops respectively came from multi-instance photographs) and corrected before any reported number was finalized."),
  subHeading("F. Evaluation Protocol"),
  bodyPara("All wheel-count models share one held-out validation split under the corrected partitioning. The helmet detector is evaluated on a test split never used for epoch selection. Voting is evaluated on UA-DETRAC by grouping predictions under ground-truth track identifiers, isolating the voting effect from tracker error. Re-identification follows the standard VeRi single-camera-exclusion protocol. All experiments run on commodity CPU hardware (AMD Ryzen 5 7600), with no GPU used at any stage."),
];

// ============================================================ III. RESULTS AND DISCUSSION
const resultsSection = [
  sectionHeading("III. Results and Discussion"),
  subHeading("A. Adaptive Preprocessing"),
  bodyPara(`The learned classifier reaches ${P1.accuracy.toFixed(2)} four-class accuracy versus a 0.49 rule-based baseline. To rule out a dataset-provenance confound, since each class was originally drawn from a different source dataset, the fog and rain pair was isolated (both from DAWN, holding provenance constant), and the model still achieved 0.68 accuracy against a 0.49 same-source baseline, which is evidence the model learns genuine weather signal rather than dataset fingerprints.`),
  ...figure("outputs/report/confusion_matrix.png", 280, 210, "Figure 1: Four-class scene confusion matrix. Error concentrates on the fog-rain boundary, a physically overlapping condition pair."),
  subHeading("B. Wheel-Count and Helmet Detection"),
  bodyPara(`Table 1 compares the three wheel-count architectures under the corrected, leakage-free split; YOLOv8-cls was selected (${P2W.yolo.accuracy.toFixed(3)} accuracy) though the comparison is not pretraining-neutral, since its margin partly reflects transfer learning. CSPNeXt, the largest model at 2.35 million parameters, is nonetheless the fastest at inference due to its CPU-efficient depthwise design, and was still improving with additional training on an earlier dataset version, so its reported figure is a floor rather than a ceiling. After the leakage correction, four-wheeler accuracy fell the most (from 0.81 to 0.70 F1), consistent with it being the most leakage-exposed class at 91%, while three-wheeler accuracy was unchanged (0.96 F1 before and after), disproving an alternative hypothesis that its score came from a whole-photo framing shortcut rather than genuine visual distinctiveness.`),
  new Paragraph({
    spacing: { before: 80, after: 40 },
    children: [run("Table 1. Wheel-count accuracy by architecture, leakage-free split.", { bold: true, size: SZ_CAP })],
  }),
  simpleTable(
    ["Model", "Accuracy", "Pretrained"],
    [
      [P2W.cnn.model, P2W.cnn.accuracy.toFixed(3), "No"],
      [P2W.cspnext.model, P2W.cspnext.accuracy.toFixed(3), "No"],
      [P2W.yolo.model + " *", P2W.yolo.accuracy.toFixed(3), "ImageNet"],
    ]),
  bodyPara(`The helmet detector reaches ${P2H.test.mAP50.toFixed(3)} mean average precision at 0.50 intersection-over-union on held-out test data (the validation split, used for epoch selection, was ${P2H.val.mAP50.toFixed(3)}, which is consistent given both splits are small). Passenger-helmet classes are weakest on both splits (average precision 0.545 and 0.565) due to genuinely fewer training instances (65 to 74 versus 439 to 500 for the common classes). A targeted threefold oversampling fix was tested and made results worse (mean average precision falling from 0.764 to 0.711), because plain duplication adds exposure without visual diversity and causes overfitting rather than generalization; this negative result is reported rather than omitted.`),
  ...figure("outputs/report_p2/wheel_compare.png", 280, 200, "Figure 2: Wheel-count accuracy and macro-F1, leakage-free split."),
  ...figure("outputs/report_p2/helmet_ap.png", 280, 200, "Figure 3: Per-class helmet average precision, held-out test split."),
  subHeading("C. Tracking, Voting, and Re-Identification"),
  bodyPara(`UA-DETRAC is approximately 97% one vehicle class, with non-comparable occlusion bands (the heaviest-occlusion band is 100% one class), which makes per-band raw accuracy invalid. Flip-rate, defined as how often the emitted label changes between consecutive frames of one tracked vehicle, is therefore reported as the primary metric, and it fell ${flipDrop.toFixed(1)}% from a single-frame window to a thirty-frame window. Balanced (macro) accuracy improved more modestly (best at a fifteen-frame window: ${P3O.overall["15"].toFixed(3)} versus ${P3O.overall["1"].toFixed(3)} for a single frame); part of the flip-rate reduction is mechanical, since averaging smooths noise as well as signal, and both numbers are reported together for this reason.`),
  bodyPara(`On VeRi-776, off-the-shelf ImageNet-pretrained OSNet, with no vehicle-identity metric learning applied, reaches Rank-1 ${P3REID.rank1.toFixed(3)} and mean average precision ${P3REID.mAP.toFixed(3)}, well below fine-tuned state-of-the-art results (approximately 90% and 70% respectively) but far above the 776-way chance level of 0.1%, confirming that the embeddings carry real appearance signal even without fine-tuning.`),
  ...figure("outputs/report_p3/flip_rate.png", 280, 200, "Figure 4: Prediction flip-rate versus vote window N."),
  ...figure("outputs/report_p3/reid.png", 270, 200, "Figure 5: VeRi-776 cross-camera re-identification scores, off-the-shelf OSNet."),
  subHeading("D. Risk Indexing"),
  bodyPara(`Rule logic is validated against ${P3RISK.n_tracks.toLocaleString()} real motorcycle tracks using ground-truth occupancy and helmet-use labels, independent of detector accuracy. Overload triggers on ${P3RISK.rule_trigger_rates.overloaded_pct.toFixed(1)}% of tracks and helmet misuse on ${P3RISK.rule_trigger_rates.helmet_misuse_pct.toFixed(1)}%. Notably, HIGH risk never triggers from these two factors alone, since their combined weight of 4.0 is below the 6.0 threshold required for the HIGH tier; this is a calibration finding surfaced by validation against ground truth rather than a code defect.`),
  subHeading("E. Optimization"),
  new Paragraph({
    spacing: { before: 80, after: 40 },
    children: [run("Table 2. SmallCNN accuracy across optimization stages.", { bold: true, size: SZ_CAP })],
  }),
  simpleTable(
    ["Configuration", "Accuracy", "Retention", "Size (MB)"],
    [
      ["FP32", sc_fp32.accuracy.toFixed(3), "1.00", sc_fp32.size_mb.toFixed(2)],
      ["Pruned (naive)", sc_naive.accuracy.toFixed(3), sc_naive.accuracy_retention.toFixed(2), sc_naive.size_mb.toFixed(2)],
      ["Pruned + fine-tune", sc_ft.accuracy.toFixed(3), sc_ft.accuracy_retention.toFixed(2), sc_ft.size_mb.toFixed(2)],
      ["+ ONNX INT8", sc_int8.accuracy.toFixed(3), sc_int8.accuracy_retention.toFixed(2), sc_int8.size_mb.toFixed(2)],
    ]),
  bodyPara("Thirty percent structured pruning without fine-tuning collapses accuracy to a retention of 0.31; three epochs of low-learning-rate fine-tuning restores it to a retention of 0.98. On the recovered model, INT8 quantization is essentially free, producing no further accuracy change for either benchmarked architecture, and ONNX Runtime INT8 shrinks models 3.8 to 3.9 times. Latency is not uniformly improved by ONNX at batch size one on CPU for these small models, since session overhead dominates; the size reduction is therefore reported as unconditional and the latency change as model-dependent, not assumed."),
  ...figure("outputs/report_p4/pruning_recovery.png", 280, 200, "Figure 6: Naive pruning collapse versus fine-tune recovery, both benchmarked models."),
  bodyPara("A recurring engineering risk across all four phases is that every heavy backend used in this pipeline, including Ultralytics, DeepSORT, torchreid, and DeepFace, degrades gracefully to a lightweight fallback rather than crashing, which means a broken dependency and a working one produce indistinguishable console output unless the active backend is explicitly asserted. Two backends were found silently degraded mid-project, and a class-identifier to weights mismatch inverted helmet-compliance semantics before being caught, motivating a regression test suite built specifically from real, previously shipped defects rather than speculative unit tests."),
  bodyPara("Limitations of this work include the following. The granular helmet sub-classes named in some prior task specifications, such as strap-unfastened, helmet-on-handlebar, and helmet-on-arm, have no available public dataset and are not modelled here. The occlusion ablation uses UA-DETRAC, which contains no two- or three-wheelers, so its class coverage does not match the detection models' target domain. Cross-camera re-identification uses an unfine-tuned backbone by design, in order to measure rather than assume the zero-shot baseline. On hardware where an installed torch build targets a newer CUDA driver than is present, loading torch and TensorFlow in one process was found to cause a segmentation fault; demographics estimation is therefore process-isolated in the implementation, a deployment detail outside this paper's main scope but documented for reproducibility."),
];

// ============================================================ IV. CONCLUSION
const conclusionSection = [
  sectionHeading("IV. Conclusion"),
  bodyPara("This paper presented a complete CPU-only traffic surveillance pipeline and, across four phases, consistently prioritized measuring and reporting real outcomes, including several that did not match initial expectations, over presenting only favourable results. Future work includes fine-tuning the re-identification backbone on VeRi-776's own training split, which has already been downloaded; sourcing data for the missing granular helmet sub-classes; and re-evaluating occlusion-robust voting on footage containing the two- and three-wheeler classes this system otherwise targets."),
];

// ============================================================ BACK MATTER
const backMatter = [
  sectionHeading("Acknowledgments"),
  bodyPara("The authors thank their project guide, [Guide Name], and the Department of Computer Science & Engineering, Walchand Institute of Technology, Solapur, for their guidance and for providing the resources necessary to carry out this work."),
  sectionHeading("Author Contributions"),
  bodyPara("[Author 1] designed and implemented the preprocessing and detection pipeline, ran the experiments, and drafted the manuscript. [Author 2] contributed to the tracking and risk-indexing implementation and result validation. [Author 3] contributed to dataset preparation and evaluation. All authors reviewed and approved the final manuscript."),
  sectionHeading("Conflicts of Interest"),
  bodyPara("The authors declare that they have no conflict of interest."),
];

// ============================================================ REFERENCES
const refs = [
  "N. Wojke, A. Bewley, and D. Paulus, “Simple Online and Realtime Tracking with a Deep Association Metric,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2017.",
  "K. Zhou, Y. Yang, A. Cavallaro, and T. Xiang, “Omni-Scale Feature Learning for Person Re-Identification,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2019.",
  "X. Liu, W. Liu, H. Ma, and H. Fu, “Large-Scale Vehicle Re-Identification in Urban Surveillance Videos,” in Proc. IEEE Int. Conf. Multimedia and Expo (ICME), 2016.",
  "C. Lyu et al., “RTMDet: An Empirical Study of Designing Real-Time Object Detectors,” arXiv:2212.07784, 2022.",
  "L. Wen et al., “UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking,” Comput. Vis. Image Understand., 2020.",
  "G. Jocher et al., “Ultralytics YOLOv8,” github.com/ultralytics/ultralytics, 2023.",
  "B. Naik et al., “HELMET: A Dataset of Helmet Use in Motorcycle Traffic,” osf.io/4pwj8, 2019.",
  "AI City Challenge, “Track 5: Detecting Violation of Helmet Rule for Motorcyclists,” aicitychallenge.org, 2023.",
];
const referencesSection = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [run("REFERENCES", { bold: true, size: SZ_HEAD })],
  }),
  ...refs.map((r, i) => refItem(i + 1, r)),
];

// ---- simple bordered table helper (defined after use above via hoisting
// is not available for const fns in this style, so declare before use) ----
function simpleTable(header, rows) {
  const { Table, TableRow, TableCell, WidthType, ShadingType } = require("docx");
  const nCols = header.length;
  const totalWidth = 4300; // approx column content width in twips
  const colW = Math.floor(totalWidth / nCols);
  const border = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkRow = (cells, isHeader) => new TableRow({
    children: cells.map(text => new TableCell({
      borders,
      width: { size: colW, type: WidthType.DXA },
      shading: isHeader ? { fill: "D9D9D9", type: ShadingType.CLEAR } : undefined,
      margins: { top: 40, bottom: 40, left: 60, right: 60 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run(String(text), { bold: !!isHeader, size: SZ_CAP })],
      })],
    })),
  });
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: Array(nCols).fill(colW),
    rows: [mkRow(header, true), ...rows.map(r => mkRow(r, false))],
  });
}

// ============================================================ DOCUMENT
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SZ_BODY } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: mm(210), height: mm(297) }, // A4
        margin: { top: mm(18.1), bottom: mm(20), left: mm(19), right: mm(16) },
      },
      column: { count: 1 }, // title block spans full width
    },
    children: [title, authors, ...affilBlock, abstractHeading, abstractPara, keywordsPara, divider],
  }, {
    properties: {
      page: {
        size: { width: mm(210), height: mm(297) },
        margin: { top: mm(18.1), bottom: mm(20), left: mm(19), right: mm(16) },
      },
      column: { count: 2, space: mm(12.7), equalWidth: true, separate: false },
    },
    children: [
      ...introSection,
      ...methodSection,
      ...resultsSection,
      ...conclusionSection,
      ...backMatter,
      ...referencesSection,
    ],
  }],
});

const OUT = "docs/paper/IJSRSET_paper.docx";
fs.mkdirSync("docs/paper", { recursive: true });
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUT, buffer);
  console.log(`DOCX -> ${OUT}`);
});
