"""
scripts/run_occlusion_ablation.py
----------------------------------
Phase 3 - the project's headline experiment.

Question: does multi-frame majority voting actually improve wheel-count
classification under occlusion, and by how much?

Method. For every ground-truth vehicle track in a UA-DETRAC sequence we crop the
annotated box in each frame and classify it. Because UA-DETRAC gives us the true
track id, we can group predictions per vehicle *without* depending on the
tracker being perfect -- this isolates the voting effect from tracking error.
We then compare:

    N = 1   single-frame prediction (baseline)
    N > 1   majority vote over the trailing N frames

Results are stratified by UA-DETRAC's own occlusion annotation
(none / light / medium / heavy), which is the whole point: voting should help
most exactly where single frames are unreliable.

A negative result is a valid result and is reported as such.

Usage:
    python scripts/run_occlusion_ablation.py --root data/raw/UA-DETRAC \
        --sequences 5 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.detrac_loader import DetracSequence, discover_sequences
from src.models.benchmark_classifier import WheelClassifier
from utils.config import load_config

OUT_DIR = "results/phase3"
BANDS = ["none", "light", "medium", "heavy"]


def crop(img, bbox, pad: int = 2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


def majority(votes: List[int]) -> int:
    return int(np.bincount(np.asarray(votes)).argmax())


def balanced_accuracy(per_class: Dict[int, List[int]]) -> float | None:
    """
    Macro-averaged recall (a.k.a. balanced accuracy).

    UA-DETRAC is ~97% cars, so plain accuracy is dominated by one class and an
    always-predict-4-wheeler model would score 0.97. Subsampling to equal class
    counts was tried first and was worse: it collapsed the evaluation to a
    handful of tracks and emptied the heavy-occlusion band entirely. Correcting
    the *metric* instead keeps every sample and every occlusion band populated.
    """
    recalls = [c / n for c, n in per_class.values() if n > 0]
    return sum(recalls) / len(recalls) if recalls else None


def run(root: str, n_sequences: int, max_frames: int, min_box: int,
        vote_windows: List[int], balance: bool = True) -> Dict:
    cfg = load_config()
    clf = WheelClassifier(cfg)
    seqs = discover_sequences(root, limit=n_sequences)
    if not seqs:
        raise SystemExit(f"No UA-DETRAC sequences found under {root}")

    keep_tracks = None   # use every sample; imbalance is handled in the metric

    # per-window, per-band, per-true-class: [correct, total]
    tally: Dict[int, Dict[str, Dict[int, List[int]]]] = {
        n: {b: defaultdict(lambda: [0, 0]) for b in BANDS} for n in vote_windows
    }
    # per-window, per-band: [flips, transitions]
    flips: Dict[int, Dict[str, List[int]]] = {
        n: {b: [0, 0] for b in BANDS} for n in vote_windows
    }
    last_emitted: Dict[int, Dict[tuple, int]] = {n: {} for n in vote_windows}

    # rolling vote buffers per (sequence, target)
    buffers: Dict[tuple, deque] = defaultdict(
        lambda: deque(maxlen=max(vote_windows)))
    n_crops = 0

    for name, images_dir, xml_path in seqs:
        seq = DetracSequence(name, images_dir, xml_path)
        print(f"[{name}] {len(seq)} frames")
        for fi, frame in enumerate(seq):
            if fi >= max_frames:
                break
            if not frame.targets:
                continue
            img = frame.image()
            if img is None:
                continue

            crops, metas = [], []
            for t in frame.targets:
                if keep_tracks is not None and (name, t.target_id) not in keep_tracks:
                    continue
                c = crop(img, t.bbox)
                if c is None or min(c.shape[:2]) < min_box:
                    continue
                crops.append(c)
                metas.append(t)
            if not crops:
                continue

            preds = clf.classify(crops)
            n_crops += len(preds)

            for pred, t in zip(preds, metas):
                key = (name, t.target_id)
                buffers[key].append(pred.cls_id)
                buf = list(buffers[key])
                band = t.occlusion_band
                truth = t.wheel_class
                for n in vote_windows:
                    voted = majority(buf[-n:]) if n > 1 else buf[-1]
                    slot = tally[n][band][truth]
                    slot[1] += 1
                    if voted == truth:
                        slot[0] += 1

                    # Temporal stability: did the emitted label change since the
                    # previous frame of this same vehicle? Flip-rate does not
                    # depend on class balance, so unlike accuracy it stays
                    # comparable across occlusion bands even though UA-DETRAC's
                    # bands contain different class mixes.
                    prev = last_emitted[n].get(key)
                    if prev is not None:
                        flips[n][band][1] += 1
                        if voted != prev:
                            flips[n][band][0] += 1
                    last_emitted[n][key] = voted

    # Assemble report -- headline metric is macro-averaged recall.
    report: Dict = {"sequences": [s[0] for s in seqs], "crops_classified": n_crops,
                    "vote_windows": vote_windows, "metric": "balanced_accuracy",
                    "per_band": {}, "overall": {}, "raw_accuracy": {}}
    for n in vote_windows:
        merged: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        for b in BANDS:
            for cls, (c, t) in tally[n][b].items():
                merged[cls][0] += c
                merged[cls][1] += t
        report["overall"][str(n)] = balanced_accuracy(merged) or 0.0
        tc = sum(v[0] for v in merged.values())
        tn = sum(v[1] for v in merged.values())
        report["raw_accuracy"][str(n)] = tc / tn if tn else 0.0
        report["per_band"][str(n)] = {
            b: balanced_accuracy(tally[n][b]) for b in BANDS
        }
    # Flip-rate: the class-balance-independent view of the same question.
    report["flip_rate"] = {
        str(n): {b: (flips[n][b][0] / flips[n][b][1] if flips[n][b][1] else None)
                 for b in BANDS}
        for n in vote_windows
    }
    report["flip_rate_overall"] = {
        str(n): (sum(flips[n][b][0] for b in BANDS) /
                 sum(flips[n][b][1] for b in BANDS)
                 if sum(flips[n][b][1] for b in BANDS) else None)
        for n in vote_windows
    }
    report["band_support"] = {
        b: sum(v[1] for v in tally[vote_windows[0]][b].values()) for b in BANDS
    }
    report["band_class_mix"] = {
        b: {str(c): v[1] for c, v in tally[vote_windows[0]][b].items()}
        for b in BANDS
    }
    report["class_support"] = {
        str(cls): sum(tally[vote_windows[0]][b][cls][1] for b in BANDS
                      if cls in tally[vote_windows[0]][b])
        for b in BANDS for cls in tally[vote_windows[0]][b]
    }
    return report


def print_report(r: Dict) -> None:
    ns = r["vote_windows"]
    print("\n=== Multi-frame voting ablation (balanced accuracy) ===")
    print(f"crops classified: {r['crops_classified']}  sequences: {len(r['sequences'])}")
    header = "band      " + "".join(f"N={n:<7}" for n in ns) + "support"
    print(header)
    for b in BANDS:
        row = f"{b:<10}"
        for n in ns:
            v = r["per_band"][str(n)][b]
            row += f"{v:.3f}   " if v is not None else "  --    "
        row += str(r["band_support"][b])
        print(row)
    row = f"{'OVERALL':<10}"
    for n in ns:
        row += f"{r['overall'][str(n)]:.3f}   "
    print(row)
    row = f"{'(raw acc)':<10}"
    for n in ns:
        row += f"{r['raw_accuracy'][str(n)]:.3f}   "
    print(row)
    print(f"class support: {r.get('class_support')}")
    print(f"band class mix: {r.get('band_class_mix')}")

    # Flip-rate table -- unlike accuracy this is comparable across bands.
    print("\n=== Prediction flip-rate (lower is more stable) ===")
    print("band      " + "".join(f"N={n:<7}" for n in ns))
    for b in BANDS:
        row = f"{b:<10}"
        for n in ns:
            v = r["flip_rate"][str(n)][b]
            row += f"{v:.3f}   " if v is not None else "  --    "
        print(row)
    row = f"{'OVERALL':<10}"
    for n in ns:
        v = r["flip_rate_overall"][str(n)]
        row += f"{v:.3f}   " if v is not None else "  --    "
    print(row)
    f1 = r["flip_rate_overall"][str(ns[0])]
    fb = min((r["flip_rate_overall"][str(n)] for n in ns
              if r["flip_rate_overall"][str(n)] is not None), default=None)
    if f1 and fb is not None:
        print(f"flip-rate {f1:.3f} -> {fb:.3f}  "
              f"({(f1 - fb) / f1 * 100:.1f}% fewer label changes)")

    base, best_n = r["overall"][str(ns[0])], max(ns, key=lambda n: r["overall"][str(n)])
    delta = r["overall"][str(best_n)] - base
    verdict = ("voting HELPS" if delta > 0.005 else
               "voting has NO material effect" if abs(delta) <= 0.005 else
               "voting HURTS")
    print(f"\nBest window N={best_n}: {r['overall'][str(best_n)]:.3f} "
          f"vs single-frame {base:.3f}  (delta {delta:+.3f}) -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/UA-DETRAC")
    ap.add_argument("--sequences", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--min-box", type=int, default=24)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 5, 10, 15, 30])
    ap.add_argument("--no-balance", action="store_true",
                    help="disable per-class track balancing (not recommended: "
                         "UA-DETRAC is ~97%% cars)")
    args = ap.parse_args()

    r = run(args.root, args.sequences, args.max_frames, args.min_box,
            args.windows, balance=not args.no_balance)
    print_report(r)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/occlusion_ablation.json", "w") as fh:
        json.dump(r, fh, indent=2)
    print(f"\nsaved -> {OUT_DIR}/occlusion_ablation.json")


if __name__ == "__main__":
    main()
