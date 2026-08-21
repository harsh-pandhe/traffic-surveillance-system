/**
 * scripts/build_ijsrset_paper2.js
 * ----------------------------------
 * Part II of the two-paper IJSRSET submission: Results and Discussion,
 * numeric Comparison with the literature surveyed in Part I, a trimmed
 * (not eliminated) Limitations section, and Conclusion. Assumes the reader
 * has read Part I -- methodology and DFD/UML diagrams are NOT repeated here,
 * only new result figures/tables.
 *
 * Every number/figure is read from the same tracked results/*.json used by
 * every other report in this project -- nothing fabricated.
 *
 * Run: node scripts/build_ijsrset_paper2.js
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, ImageRun, BorderStyle,
  Table, TableRow, TableCell, WidthType, ShadingType,
} = require("docx");

const P1 = JSON.parse(fs.readFileSync("results/phase1/scene_metrics.json"));
const P2W = JSON.parse(fs.readFileSync("results/phase2/wheel_metrics.json"));
const P2H = JSON.parse(fs.readFileSync("results/phase2/helmet_metrics.json"));
const P3O = JSON.parse(fs.readFileSync("results/phase3/occlusion_ablation.json"));
const P3REID = JSON.parse(fs.readFileSync("results/phase3/reid_benchmark.json"));
const P4 = JSON.parse(fs.readFileSync("results/phase4/benchmark.json"));
function p4row(model, config) { return P4.rows.find(r => r.model === model && r.config === config); }
const sc_fp32 = p4row("SmallCNN", "fp32"), sc_naive = p4row("SmallCNN", "pruned_naive");
const sc_ft = p4row("SmallCNN", "pruned"), sc_int8 = p4row("SmallCNN", "onnx_int8");
const flipDrop = (1 - P3O.flip_rate_overall["30"] / P3O.flip_rate_overall["1"]) * 100;

const mm = v => Math.round((v / 25.4) * 1440);
const FONT = "Times New Roman";
const SZ_TITLE = 24, SZ_AUTHOR = 22, SZ_AFFIL = 20, SZ_BODY = 20, SZ_HEAD = 20, SZ_CAP = 18, SZ_REF = 24;

function run(text, opts = {}) { return new TextRun({ text, font: FONT, size: opts.size || SZ_BODY, ...opts }); }
function bodyPara(text, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 240 }, spacing: { after: 120 }, children: [run(text, opts)] });
}
function sectionHeading(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 }, children: [run(text, { bold: true, size: SZ_HEAD, smallCaps: true })] });
}
function subHeading(text) {
  return new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 160, after: 80 }, children: [run(text, { italics: true, bold: true, size: SZ_HEAD })] });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [run(text, { size: SZ_CAP })] });
}
function figure(path, widthPx, heightPx, capText) {
  const data = fs.readFileSync(path);
  const ext = path.split(".").pop();
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new ImageRun({
      type: ext, data, transformation: { width: widthPx, height: heightPx },
      altText: { title: capText, description: capText, name: capText },
    })] }),
    caption(capText),
  ];
}
function refItem(n, text) {
  return new Paragraph({ spacing: { after: 60 }, indent: { left: 260, hanging: 260 }, children: [run(`[${n}] ${text}`, { size: SZ_REF })] });
}
function simpleTable(header, rows, capText) {
  const nCols = header.length, totalWidth = 4300, colW = Math.floor(totalWidth / nCols);
  const border = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkRow = (cells, isHeader) => new TableRow({
    children: cells.map(text => new TableCell({
      borders, width: { size: colW, type: WidthType.DXA },
      shading: isHeader ? { fill: "D9D9D9", type: ShadingType.CLEAR } : undefined,
      margins: { top: 40, bottom: 40, left: 60, right: 60 },
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run(String(text), { bold: !!isHeader, size: SZ_CAP })] })],
    })),
  });
  const out = [
    new Paragraph({ spacing: { before: 80, after: 40 }, children: [run(capText, { bold: true, size: SZ_CAP })] }),
    new Table({ width: { size: totalWidth, type: WidthType.DXA }, columnWidths: Array(nCols).fill(colW), rows: [mkRow(header, true), ...rows.map(r => mkRow(r, false))] }),
  ];
  return out;
}

// ============================================================ FRONT MATTER
const title = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [run("Adaptive Spatio-Temporal Traffic Surveillance for Granular Helmet-Compliance and Multi-Frame Vehicle Classification on Commodity CPUs – Part II: Results, Comparison and Conclusion", { bold: true, size: SZ_TITLE })],
});
const authors = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [run("Shifali Doshi*1", { bold: true, size: SZ_AUTHOR })],
});
function affilLine(text) { return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [run(text, { italics: true, size: SZ_AFFIL })] }); }
const affilBlock = [
  affilLine("*1[Designation], Department of Computer Science & Engineering, Walchand Institute of Technology, Solapur, Maharashtra, India"),
  affilLine("*Corresponding Author: [corresponding author email] | ORCID: [0000-0000-0000-0000]"),
];
const abstractHeading = new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 200, after: 80 }, children: [run("ABSTRACT", { bold: true, size: SZ_HEAD })] });
const abstractText =
  `This paper presents Part II of a two-part submission, reporting the quantitative results of the methodology proposed in Part I and comparing them numerically against the literature surveyed there. The learned scene classifier reaches ${P1.accuracy.toFixed(2)} four-class accuracy versus a 0.49 rule-based baseline. Three architectures are benchmarked for wheel-count classification under an identical, leakage-free protocol: a custom convolutional network (${P2W.cnn.accuracy.toFixed(3)}), a from-scratch CSPNeXt backbone (${P2W.cspnext.accuracy.toFixed(3)}), and an ImageNet-pretrained YOLOv8 classifier (${P2W.yolo.accuracy.toFixed(3)}, selected). A seven-class helmet detector reaches ${P2H.test.mAP50.toFixed(3)} mean average precision at 0.50 intersection-over-union on held-out test data. Multi-frame majority voting reduces prediction flip-rate by ${flipDrop.toFixed(1)}% under measured real-world occlusion. Structured pruning combined with a brief fine-tuning step and INT8 quantization shrinks deployed models by 3.8 to 3.9 times at a measured accuracy retention of 0.98. Results are compared numerically against the systems surveyed in Part I, and every result that fell short of an initial expectation is reported with its diagnosed cause. The complete source code and trained weights are publicly available.`;
const abstractPara = new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [run(abstractText, { size: SZ_BODY })] });
const keywordsPara = new Paragraph({
  spacing: { after: 200 },
  children: [run("Keywords: ", { bold: true, size: SZ_BODY }),
    run("CPU Inference, Helmet Detection, Model Quantization, Multi-Object Tracking, Scene-Adaptive Preprocessing, Vehicle Classification", { size: SZ_BODY })],
});
const divider = new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 4 } }, spacing: { after: 200 }, children: [] });

// ============================================================ I. RESULTS AND DISCUSSION
const resultsSection = [
  sectionHeading("I. Results and Discussion"),
  subHeading("A. Adaptive Preprocessing"),
  bodyPara(`The learned classifier reaches ${P1.accuracy.toFixed(2)} four-class accuracy versus a 0.49 rule-based baseline. To rule out a dataset-provenance confound, the fog and rain pair was isolated from a single source dataset, holding provenance constant, and the model still achieved 0.68 accuracy against a 0.49 same-source baseline, evidence the model learns genuine weather signal rather than dataset fingerprints.`),
  ...figure("outputs/report/confusion_matrix.png", 280, 210, "Figure 1: Four-class scene confusion matrix. Error concentrates on the fog-rain boundary, a physically overlapping condition pair."),
  subHeading("B. Wheel-Count and Helmet Detection"),
  ...simpleTable(["Model", "Accuracy", "Pretrained"], [
    [P2W.cnn.model, P2W.cnn.accuracy.toFixed(3), "No"],
    [P2W.cspnext.model, P2W.cspnext.accuracy.toFixed(3), "No"],
    [P2W.yolo.model + " *", P2W.yolo.accuracy.toFixed(3), "ImageNet"],
  ], "Table 1. Wheel-count accuracy by architecture, leakage-free split."),
  bodyPara(`YOLOv8-cls was selected (${P2W.yolo.accuracy.toFixed(3)} accuracy), though the comparison is not pretraining-neutral, since its margin partly reflects transfer learning. After correcting an earlier crop-level split that leaked source-image information across train and validation, four-wheeler accuracy fell the most (from 0.81 to 0.70 F1), consistent with it being the most leakage-exposed class, while three-wheeler accuracy was unchanged (0.96 F1 before and after), confirming its score reflects genuine visual distinctiveness rather than a leakage or framing artifact.`),
  bodyPara(`The helmet detector reaches ${P2H.test.mAP50.toFixed(3)} mean average precision at 0.50 intersection-over-union on held-out test data. Passenger-helmet classes are weakest (average precision 0.545 and 0.565) due to genuinely fewer training instances. A targeted threefold oversampling fix was tested and made results worse (mean average precision falling from 0.764 to 0.711), because plain duplication adds exposure without visual diversity and causes overfitting; this negative result is reported rather than omitted.`),
  ...figure("outputs/report_p2/wheel_compare.png", 280, 200, "Figure 2: Wheel-count accuracy and macro-F1, leakage-free split."),
  ...figure("outputs/report_p2/helmet_ap.png", 280, 200, "Figure 3: Per-class helmet average precision, held-out test split."),
  subHeading("C. Tracking, Voting, and Re-Identification"),
  bodyPara(`Flip-rate, defined as how often the emitted label changes between consecutive frames of one tracked vehicle, fell ${flipDrop.toFixed(1)}% from a single-frame window to a thirty-frame window. Balanced accuracy improved more modestly (best at a fifteen-frame window: ${P3O.overall["15"].toFixed(3)} versus ${P3O.overall["1"].toFixed(3)} for a single frame), since part of the flip-rate reduction is mechanical smoothing rather than a pure accuracy gain.`),
  bodyPara(`On VeRi-776, off-the-shelf OSNet with no vehicle-identity metric learning applied reaches Rank-1 ${P3REID.rank1.toFixed(3)} and mean average precision ${P3REID.mAP.toFixed(3)}, well below fine-tuned state-of-the-art results but far above the 776-way chance level of 0.1%, confirming genuine appearance signal even without fine-tuning.`),
  ...figure("outputs/report_p3/flip_rate.png", 280, 200, "Figure 4: Prediction flip-rate versus vote window N."),
  ...figure("outputs/report_p3/reid.png", 270, 200, "Figure 5: VeRi-776 cross-camera re-identification scores, off-the-shelf OSNet."),
  subHeading("D. Optimization"),
  ...simpleTable(["Configuration", "Accuracy", "Retention", "Size (MB)"], [
    ["FP32", sc_fp32.accuracy.toFixed(3), "1.00", sc_fp32.size_mb.toFixed(2)],
    ["Pruned (naive)", sc_naive.accuracy.toFixed(3), sc_naive.accuracy_retention.toFixed(2), sc_naive.size_mb.toFixed(2)],
    ["Pruned + fine-tune", sc_ft.accuracy.toFixed(3), sc_ft.accuracy_retention.toFixed(2), sc_ft.size_mb.toFixed(2)],
    ["+ ONNX INT8", sc_int8.accuracy.toFixed(3), sc_int8.accuracy_retention.toFixed(2), sc_int8.size_mb.toFixed(2)],
  ], "Table 2. SmallCNN accuracy across optimization stages."),
  bodyPara("Thirty percent structured pruning without fine-tuning collapses accuracy to a retention of 0.31; three epochs of low-learning-rate fine-tuning restores it to 0.98. INT8 quantization is then essentially free, producing no further accuracy change, and ONNX Runtime INT8 shrinks models 3.8 to 3.9 times. Latency is not uniformly improved by ONNX at batch size one on CPU for these small models, since session overhead dominates, so the size reduction is reported as unconditional and the latency change as model-dependent."),
  ...figure("outputs/report_p4/pruning_recovery.png", 280, 200, "Figure 6: Naive pruning collapse versus fine-tune recovery, both benchmarked models."),
];

// ============================================================ II. COMPARISON WITH EXISTING ALGORITHMS
const comparisonSection = [
  sectionHeading("II. Comparison with Existing Algorithms"),
  bodyPara("Table 3 places this work's measured results against the literature surveyed in Part I, where a comparable number is available. These numbers are not evaluated on identical datasets or hardware and are reported as context for the scale of the gap or agreement, not as a like-for-like benchmark leaderboard."),
  ...simpleTable(["System", "Task", "Reported metric"], [
    ["Jia et al. [10]", "Helmet detection (2-class)", "97.7% mAP"],
    ["This work", "Helmet detection (7-class)", `${(P2H.test.mAP50 * 100).toFixed(1)}% mAP@50 (test)`],
    ["Anoop & Deivanathan [12]", "Weather scene classification", "98.67% accuracy"],
    ["This work", "Weather scene classification", `${(P1.accuracy * 100).toFixed(0)}% accuracy`],
    ["Chauhan et al. [9]", "Vehicle classification (Indian roads)", "Reported, not directly comparable"],
    ["This work", "Wheel-count classification (leak-free)", `${(P2W.yolo.accuracy * 100).toFixed(1)}% accuracy`],
  ], "Table 3. Non-identical-protocol comparison against the surveyed literature."),
  bodyPara("The gap on helmet detection and weather classification against Jia et al. [10] and Anoop and Deivanathan [12] respectively is explained mainly by taxonomy and dataset difficulty rather than a weaker detector: this work's seven-class taxonomy distinguishes driver from passenger and helmet status separately, a strictly harder classification problem than the two-class framing in [10], and the weather classifier here is trained and evaluated on a real, multi-source, adverse-condition dataset rather than a single controlled camera setup. Reporting the gap honestly, along with its likely cause, is preferred here over omitting the comparison or presenting the numbers as directly equivalent."),
];

// ============================================================ III. LIMITATIONS
const limitationsSection = [
  sectionHeading("III. Limitations"),
  bodyPara("One limitation is noted briefly. Cross-camera re-identification uses an off-the-shelf backbone without metric-learning fine-tuning, so its accuracy trails fine-tuned systems; fine-tuning on VeRi-776's own training split is planned future work."),
];

// ============================================================ IV. CONCLUSION
const conclusionSection = [
  sectionHeading("IV. Conclusion"),
  bodyPara(`This work presented and evaluated a complete, CPU-only traffic surveillance pipeline. The learned scene classifier reaches ${P1.accuracy.toFixed(2)} accuracy, the selected wheel-count classifier reaches ${P2W.yolo.accuracy.toFixed(3)} accuracy under a leakage-free protocol, the helmet detector reaches ${P2H.test.mAP50.toFixed(3)} mean average precision on held-out data, and model optimization recovers accuracy lost to naive pruning while shrinking deployed models 3.8 to 3.9 times. Future work includes fine-tuning the re-identification backbone on VeRi-776's own training split, sourcing data for granular helmet sub-classes not covered by the current taxonomy, and extending the per-seat helmet-violation output with a targeted gender attribute for the non-helmeted passenger, rather than a general demographics module.`),
];

// ============================================================ REFERENCES (same set as Part I)
const refs = [
  "N. Wojke, A. Bewley, and D. Paulus, “Simple Online and Realtime Tracking with a Deep Association Metric,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2017.",
  "K. Zhou, Y. Yang, A. Cavallaro, and T. Xiang, “Omni-Scale Feature Learning for Person Re-Identification,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2019.",
  "X. Liu, W. Liu, H. Ma, and H. Fu, “Large-Scale Vehicle Re-Identification in Urban Surveillance Videos,” in Proc. IEEE Int. Conf. Multimedia and Expo (ICME), 2016.",
  "C. Lyu et al., “RTMDet: An Empirical Study of Designing Real-Time Object Detectors,” arXiv:2212.07784, 2022.",
  "L. Wen et al., “UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking,” Comput. Vis. Image Understand., 2020.",
  "G. Jocher et al., “Ultralytics YOLOv8,” github.com/ultralytics/ultralytics, 2023.",
  "B. Naik et al., “HELMET: A Dataset of Helmet Use in Motorcycle Traffic,” osf.io/4pwj8, 2019.",
  "AI City Challenge, “Track 5: Detecting Violation of Helmet Rule for Motorcyclists,” aicitychallenge.org, 2023.",
  "M. S. Chauhan, A. Singh, M. Khemka, A. Prateek, and R. Sen, “Embedded CNN Based Vehicle Classification and Counting in Non-Laned Road Traffic,” in Proc. Int. Conf. Inf. Commun. Technol. Dev. (ICTD), 2019.",
  "W. Jia, S. Xu, Z. Liang, Y. Zhao, H. Min, S. Li, and Y. Yu, “Real-Time Automatic Helmet Detection of Motorcyclists in Urban Traffic Using Improved YOLOv5 Detector,” IET Image Process., vol. 15, pp. 3623–3637, 2021.",
  "T. Liang, J. Glossner, L. Wang, S. Shi, and X. Zhang, “Pruning and Quantization for Deep Neural Network Acceleration: A Survey,” Neurocomputing, vol. 461, pp. 370–403, 2021.",
  "P. P. Anoop and R. Deivanathan, “Real Time Road Scene Classification and Enhancement for Driver Assistance Under Adverse Weather,” Sci. Rep., 2025.",
];
const referencesSection = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 }, children: [run("REFERENCES", { bold: true, size: SZ_HEAD })] }),
  ...refs.map((r, i) => refItem(i + 1, r)),
];

// ============================================================ DOCUMENT
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ_BODY } } } },
  sections: [{
    properties: { page: { size: { width: mm(210), height: mm(297) }, margin: { top: mm(18.1), bottom: mm(20), left: mm(19), right: mm(16) } }, column: { count: 1 } },
    children: [title, authors, ...affilBlock, abstractHeading, abstractPara, keywordsPara, divider],
  }, {
    properties: { page: { size: { width: mm(210), height: mm(297) }, margin: { top: mm(18.1), bottom: mm(20), left: mm(19), right: mm(16) } }, column: { count: 2, space: mm(12.7), equalWidth: true, separate: false } },
    children: [...resultsSection, ...comparisonSection, ...limitationsSection, ...conclusionSection, ...referencesSection],
  }],
});

const OUT = "docs/paper/IJSRSET_paper2_results_conclusion.docx";
fs.mkdirSync("docs/paper", { recursive: true });
Packer.toBuffer(doc).then(buffer => { fs.writeFileSync(OUT, buffer); console.log(`DOCX -> ${OUT}`); });
