/**
 * scripts/build_ijsrset_paper1.js
 * ----------------------------------
 * Part I of the two-paper IJSRSET submission (per guide's requested split):
 * Problem Statement, Literature Review, and Proposed Methodology. Part II
 * (scripts/build_ijsrset_paper2.js) carries Results, Comparison, and
 * Conclusion -- no diagrams/explanations are repeated between the two.
 *
 * Same IJSRSET spec as before: .docx, A4 two-column Times New Roman,
 * abstract 150-250 words, 4-6 alphabetized keywords, unnumbered References
 * heading with IEEE in-text citation style. Author/ORCID/guide fields are
 * bracketed placeholders -- fill in before submission.
 *
 * Run: node scripts/build_ijsrset_paper1.js
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, ImageRun, BorderStyle,
} = require("docx");

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

// ============================================================ FRONT MATTER
const title = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [run("Adaptive Spatio-Temporal Traffic Surveillance for Granular Helmet-Compliance and Multi-Frame Vehicle Classification on Commodity CPUs – Part I: Problem Statement, Literature Review and Proposed Methodology", { bold: true, size: SZ_TITLE })],
});
const authors = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [run("Shifali Doshi*1", { bold: true, size: SZ_AUTHOR })],
});
function affilLine(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [run(text, { italics: true, size: SZ_AFFIL })] });
}
const affilBlock = [
  affilLine("*1[Designation], Department of Computer Science & Engineering, Walchand Institute of Technology, Solapur, Maharashtra, India"),
  affilLine("*Corresponding Author: [corresponding author email] | ORCID: [0000-0000-0000-0000]"),
];
const abstractHeading = new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 200, after: 80 }, children: [run("ABSTRACT", { bold: true, size: SZ_HEAD })] });
const abstractText =
  "Automated enforcement of helmet compliance and vehicle-load limits for two-wheeler traffic remains largely manual despite the disproportionate crash risk this traffic segment carries, and existing automated systems typically address a single sub-problem in isolation rather than a complete pipeline. This paper, presented as Part I of a two-part submission, motivates the problem, surveys the relevant literature across helmet detection, vehicle-type classification, adverse-weather scene enhancement, and neural network compression for edge deployment, and proposes a complete methodology addressing the identified gaps. The proposed system combines a learned scene-adaptive preprocessing stage, a three-way benchmarked wheel-count classifier, a seven-class helmet-compliance detector, multi-frame tracking with temporal majority voting, cross-camera re-identification, and post-training model optimization, designed throughout for CPU-only deployment rather than assuming GPU availability. Unlike prior single-purpose systems, every stage of the proposed pipeline is evaluated together rather than in isolation, and the design explicitly accounts for practical deployment constraints such as commodity hardware and real-time throughput. Part II of this submission reports the quantitative results, a numeric comparison against the literature surveyed here, and the conclusion.";
const abstractPara = new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 }, children: [run(abstractText, { size: SZ_BODY })] });
const keywordsPara = new Paragraph({
  spacing: { after: 200 },
  children: [run("Keywords: ", { bold: true, size: SZ_BODY }),
    run("CPU Inference, Helmet Detection, Model Quantization, Multi-Object Tracking, Scene-Adaptive Preprocessing, Vehicle Classification", { size: SZ_BODY })],
});
const divider = new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000", space: 4 } }, spacing: { after: 200 }, children: [] });

// ============================================================ I. INTRODUCTION
const introSection = [
  sectionHeading("I. Introduction"),
  subHeading("A. Problem Statement"),
  bodyPara("Two-wheeler traffic dominates road usage in many dense urban environments and carries disproportionate crash risk, yet automated enforcement of helmet compliance and vehicle-load limits remains largely manual. Detection models trained on clear-weather daytime imagery degrade under fog, rain, and low light, and uniform preprocessing applied regardless of scene condition can degrade otherwise-clear frames. Public helmet-compliance datasets rarely distinguish driver from passenger, and single-frame classification is fragile under the partial occlusion common in dense traffic, so any claimed benefit of temporal aggregation has to be measured directly rather than assumed. Real-time deployment further requires that any accuracy gained from tracking, re-identification, or larger models be weighed against the latency and memory constraints of commodity CPU hardware, since GPU infrastructure is not available at every deployment site."),
  subHeading("B. Motivation"),
  bodyPara("Prior automated systems typically address a single sub-problem, such as helmet detection, license-plate recognition, or vehicle counting, in isolation, and are rarely evaluated across the full range of lighting and weather conditions actually encountered in the field. A system that classifies the scene, detects per-rider compliance violations, tracks vehicles across frames, and runs entirely on CPU is directly relevant to low-cost municipal deployment where GPU infrastructure is not guaranteed at every junction."),
  subHeading("C. Objectives"),
  bodyPara("The objectives of this work are to build an adaptive preprocessing stage that detects day, night, fog, and rain conditions and applies scene-conditioned enhancement only where it helps; to benchmark multiple architecture families for wheel-count classification under a leakage-free protocol and train a granular helmet-compliance detector; to implement multi-frame tracking with a measured occlusion-robustness benefit and cross-camera re-identification; and to optimize every deployable model via pruning, fine-tuning, and INT8 quantization, reporting accuracy retention alongside speed and size rather than speed alone."),
];

// ============================================================ II. LITERATURE REVIEW
const litSection = [
  sectionHeading("II. Literature Review"),
  bodyPara("Automated helmet detection has been studied extensively using YOLO-family detectors for their real-time, edge-deployable throughput. Jia et al. [10] proposed an improved YOLOv5 detector for real-time motorcyclist helmet detection in urban traffic, reporting a mean average precision of 97.7% and 63 frames per second, though their two-class (helmet or no-helmet) framing does not distinguish driver from passenger or model multiple riders per vehicle, a distinction this work's seven-class taxonomy addresses directly. Earlier YOLO-based systems report accuracies in the 92 to 95% range depending on the dataset used [1], underscoring that reported helmet-detection accuracy varies substantially with dataset difficulty and taxonomy granularity, which motivates reporting results on a held-out test split rather than validation alone, as done in Part II of this submission."),
  bodyPara("Track-then-classify architectures specifically have been explored for helmet compliance. Lin et al. [13] introduced a CNN-based multi-task learning model that first tracks each individual motorcycle across frames and only then registers a per-rider helmet-use label, releasing the HELMET dataset (91,000 annotated frames, 10,006 tracked motorcycles from twelve real observation sites in Myanmar) used in this work's own risk-indexing validation. Duong et al. [14] independently reached a similar track-first ordering, ranking third in the 2023 AI City Challenge Track 5 with a custom tracking framework combined with advanced object detection. Both works motivate, rather than assume, the vehicle-tracked-before-compliance-checked sequencing this work's own methodology follows (Section III-D)."),
  bodyPara("Vehicle-type classification is more commonly addressed as a byproduct of general object detection than as a purpose-built task for the two, three, four, and six-plus wheeler taxonomy relevant to heterogeneous South Asian traffic. Chauhan et al. [9] developed an embedded CNN-based system for vehicle classification and counting on non-laned Indian road traffic, directly relevant to the wheel-count taxonomy adopted here, though their evaluation does not address the train and validation leakage risk that arises when multiple vehicles are cropped from the same source photograph, an issue this work identifies and corrects explicitly. RTMDet [4] is a strong recent real-time detector family; its CSPNeXt backbone is evaluated here as a classification backbone since the official mmdetection toolchain does not install cleanly against this project's Python and PyTorch environment, motivating a from-scratch reimplementation rather than omitting the comparison."),
  bodyPara("Scene-adaptive preprocessing for adverse weather has recently been addressed directly. Anoop and Deivanathan [12] presented a real-time road scene classification and enhancement system for driver assistance under adverse weather, classifying frames into daytime, nighttime, foggy, and rainy categories and applying condition-specific enhancement, reporting 98.67% classification accuracy on an embedded Raspberry Pi platform. Their result is a useful point of reference, though it is not a directly comparable number to this work's own scene classifier, since the two systems are evaluated on different datasets and camera setups; Part II of this submission discusses this distinction explicitly rather than presenting the two accuracies as equivalent."),
  bodyPara("For multi-object tracking, DeepSORT [1] remains a mature, appearance-augmented extension of SORT that is tractable on CPU hardware, in contrast to transformer-based trackers that assume GPU throughput. The literature broadly assumes that multi-frame temporal aggregation improves robustness to occlusion, but rarely quantifies this benefit on a class-imbalanced real benchmark stratified by measured occlusion ratio, a gap this work's methodology addresses directly. For cross-camera vehicle re-identification, VeRi-776 [3] is a standard benchmark; state-of-the-art approaches fine-tune a re-identification-specific backbone, frequently OSNet [2], with a triplet or identity loss on the benchmark's own training split, a step this work evaluates the honest zero-shot gap of rather than assuming unnecessary."),
  bodyPara("For deployment-oriented model compression, Liang et al. [11] surveyed pruning and quantization techniques for deep neural network acceleration, noting that structured pruning removes whole channels or filters rather than individual weights, which is necessary for the compression to translate into real speedup on general-purpose hardware rather than only reducing the stored parameter count. Their survey does not, however, emphasize that structured pruning without a fine-tuning recovery step can catastrophically degrade accuracy, a failure mode this work measures directly rather than only reporting the post-recovery number. Jacob et al. [15] established the integer-arithmetic-only inference scheme underlying this work's INT8 quantization step, though their original evaluation did not combine quantization with a pruning-recovery stage, an interaction this work measures directly in Section III-E and reports quantitatively in Part II."),
  bodyPara("Positioned against this literature, prior work in this space typically covers a single sub-problem with a single architecture reported, evaluates preprocessing that is fixed or absent rather than scene-adaptive, and reports optimization or occlusion-robustness benefits without measuring the failure modes that a naive implementation would produce. This work instead builds one composed pipeline covering all four areas, benchmarks multiple architectures under an identical protocol with an explicit pretraining-fairness analysis, measures occlusion robustness with a class-composition-independent metric, and reports every diagnosed shortfall to its root cause rather than presenting only favourable results, as detailed in the proposed methodology below and evaluated quantitatively in Part II of this submission."),
];

// ============================================================ III. PROPOSED METHODOLOGY
const methodSection = [
  sectionHeading("III. Proposed Methodology"),
  subHeading("A. System Overview"),
  bodyPara("The proposed system is implemented as a sequential per-frame pipeline: scene classification, adaptive enhancement, helmet-compliance detection, wheel-count classification, tracking with temporal majority voting, cross-camera re-identification, and overlay rendering, illustrated in Figure 1. Every stage is independently testable and driven entirely by external configuration, so that a retrain or deployment change does not require hunting through source code for hard-coded constants. A rule-based risk-indexing stage is also implemented in the underlying system but is outside the scope of this submission's results and is not evaluated in Part II."),
  ...figure("outputs/thesis/dfd_level1.png", 290, 210, "Figure 1: Pipeline stages implemented in the underlying system; this submission's results (Part II) cover stages 1.0-4.0 only."),
  subHeading("B. Adaptive Preprocessing"),
  bodyPara("Each frame yields fifteen lighting, haze, and texture features, including luminance statistics, dark-channel mean, Laplacian variance, FFT high-frequency energy, and colourfulness. A random forest classifier, trained on real fog, rain, night, and day imagery, classifies each frame into one of four scene conditions; enhancement, such as contrast-limited adaptive histogram equalization, dark channel prior dehazing, or denoising, is applied only when the predicted condition warrants it, unlike the fixed or absent preprocessing typical of prior systems [12]."),
  subHeading("C. Detection Models"),
  bodyPara("Wheel-count classification benchmarks a custom convolutional network trained from scratch, a from-scratch CSPNeXt classifier trained from scratch, and an ImageNet-pretrained YOLOv8 classifier, on an identical dataset split by source image rather than by crop, to avoid the train and validation leakage risk noted in prior vehicle-classification work [9]. A YOLOv8n detector is separately fine-tuned for seven-class helmet compliance, distinguishing driver from passenger, unlike the binary helmet or no-helmet framing common in the literature [1], [9]."),
  subHeading("D. Tracking and Voting"),
  bodyPara("A DeepSORT-based tracker provides per-vehicle tracks; a temporal majority-vote buffer smooths the wheel-count prediction over a sliding window of N frames, addressing the occlusion-robustness gap identified in the literature review by measuring the benefit directly rather than assuming it."),
  subHeading("E. Optimization"),
  bodyPara("Each classifier is pruned using structured sparsity, optionally fine-tuned to recover accuracy, dynamically quantized to INT8, and exported to ONNX with a second INT8 pass applied via ONNX Runtime, addressing the naive-pruning failure mode noted in the model-compression literature [11] by measuring the collapse and its recovery explicitly rather than only reporting the final, recovered configuration."),
  bodyPara("The quantitative results of applying this methodology, a numeric comparison against the systems surveyed above, and the conclusion are presented in Part II of this submission."),
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
  "M. S. Chauhan, A. Singh, M. Khemka, A. Prateek, and R. Sen, “Embedded CNN Based Vehicle Classification and Counting in Non-Laned Road Traffic,” in Proc. Int. Conf. Inf. Commun. Technol. Dev. (ICTD), 2019.",
  "W. Jia, S. Xu, Z. Liang, Y. Zhao, H. Min, S. Li, and Y. Yu, “Real-Time Automatic Helmet Detection of Motorcyclists in Urban Traffic Using Improved YOLOv5 Detector,” IET Image Process., vol. 15, pp. 3623–3637, 2021.",
  "T. Liang, J. Glossner, L. Wang, S. Shi, and X. Zhang, “Pruning and Quantization for Deep Neural Network Acceleration: A Survey,” Neurocomputing, vol. 461, pp. 370–403, 2021.",
  "P. P. Anoop and R. Deivanathan, “Real Time Road Scene Classification and Enhancement for Driver Assistance Under Adverse Weather,” Sci. Rep., 2025.",
  "H. Lin, J. D. Deng, D. Albers, and F. W. Siebert, “Helmet Use Detection of Tracked Motorcycles Using CNN-Based Multi-Task Learning,” IEEE Access, vol. 8, pp. 162073–162084, 2020.",
  "V. H. Duong, Q. H. Tran, H. S. P. Nguyen, D. Q. Nguyen, and T. C. Nguyen, “Helmet Rule Violation Detection for Motorcyclists Using a Custom Tracking Framework and Advanced Object Detection Techniques,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2023, pp. 5381–5390.",
  "B. Jacob et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 2704–2713.",
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
    children: [...introSection, ...litSection, ...methodSection, ...referencesSection],
  }],
});

const OUT = "docs/paper/IJSRSET_paper1_proposed_system.docx";
fs.mkdirSync("docs/paper", { recursive: true });
Packer.toBuffer(doc).then(buffer => { fs.writeFileSync(OUT, buffer); console.log(`DOCX -> ${OUT}`); });
