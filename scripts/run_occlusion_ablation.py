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


def run(root: str, n_sequences: int, max_frames: int, min_box: int,
        vote_windows: List[int]) -> Dict:
    cfg = load_config()
    clf = WheelClassifier(cfg)
    seqs = discover_sequences(root, limit=n_sequences)
    if not seqs:
        raise SystemExit(f"No UA-DETRAC sequences found under {root}")

    # per-window, per-band: [correct, total]
    tally: Dict[int, Dict[str, List[int]]] = {
        n: {b: [0, 0] for b in BANDS} for n in vote_windows
    }
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
                    tally[n][band][1] += 1
                    if voted == truth:
                        tally[n][band][0] += 1

    # Assemble report
    report: Dict = {"sequences": [s[0] for s in seqs], "crops_classified": n_crops,
                    "vote_windows": vote_windows, "per_band": {}, "overall": {}}
    for n in vote_windows:
        tot_c = sum(tally[n][b][0] for b in BANDS)
        tot_n = sum(tally[n][b][1] for b in BANDS)
        report["overall"][str(n)] = tot_c / tot_n if tot_n else 0.0
        report["per_band"][str(n)] = {
            b: (tally[n][b][0] / tally[n][b][1] if tally[n][b][1] else None)
            for b in BANDS
        }
    report["band_support"] = {b: tally[vote_windows[0]][b][1] for b in BANDS}
    return report


def print_report(r: Dict) -> None:
    ns = r["vote_windows"]
    print("\n=== Multi-frame voting ablation (accuracy) ===")
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
    args = ap.parse_args()

    r = run(args.root, args.sequences, args.max_frames, args.min_box, args.windows)
    print_report(r)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/occlusion_ablation.json", "w") as fh:
        json.dump(r, fh, indent=2)
    print(f"\nsaved -> {OUT_DIR}/occlusion_ablation.json")


if __name__ == "__main__":
    main()
