"""
scripts/run_reid_benchmark.py
-------------------------------
Phase 3 - cross-camera vehicle ReID benchmark on VeRi-776.

VeRi-776: 776 vehicles, 20 real cameras, 51k images (37,778 train / 11,579
gallery-test / 1,678 query). Filenames encode ground truth:
`<vehicleID>_<camID>_<seq>_<idx>.jpg`.

Standard ReID protocol: for each query image, embed it with the trained OSNet
backend (`src/tracking/multi_camera_reid.py`), rank every gallery image by
cosine distance, and score whether the correct vehicle is retrieved --
excluding gallery images from the SAME camera as the query (the standard VeRi/
Market-1501 single-camera exclusion, since matching a vehicle to itself on the
same camera a moment later is trivial and not what cross-camera ReID measures).

Metrics: Rank-1, Rank-5, Rank-10 accuracy, and mAP.

Usage:
    python scripts/run_reid_benchmark.py --n-query 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.multi_camera_reid import VehicleReID
from utils.config import load_config

ROOT = "data/raw/VeRi/VeRi"
OUT = "results/phase3/reid_benchmark.json"
FNAME_RE = re.compile(r"^(\d+)_c(\d+)_")


def parse_id_cam(path: str) -> tuple[int, int]:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def embed_all(reid: VehicleReID, paths: list[str]) -> np.ndarray:
    embs = []
    for p in paths:
        img = cv2.imread(p)
        embs.append(reid.embed(img) if img is not None else np.zeros(512))
    return np.stack(embs)


def compute_ap(ranked_matches: np.ndarray) -> float:
    """Average precision for one query given a boolean match array in rank order."""
    if not ranked_matches.any():
        return 0.0
    hits = np.cumsum(ranked_matches)
    precisions = hits / (np.arange(len(ranked_matches)) + 1)
    return float(precisions[ranked_matches].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-query", type=int, default=200,
                    help="number of query images to evaluate (VeRi has 1678)")
    ap.add_argument("--gallery-limit", type=int, default=3000,
                    help="cap gallery size for CPU tractability")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    query_paths = sorted(glob.glob(os.path.join(ROOT, "image_query", "*.jpg")))
    gallery_paths = sorted(glob.glob(os.path.join(ROOT, "image_test", "*.jpg")))
    if not query_paths or not gallery_paths:
        print(f"VeRi not found under {ROOT}. Expected image_query/ and image_test/.")
        return

    rng = np.random.default_rng(args.seed)
    if len(query_paths) > args.n_query:
        query_paths = list(rng.choice(query_paths, args.n_query, replace=False))
    if len(gallery_paths) > args.gallery_limit:
        gallery_paths = list(rng.choice(gallery_paths, args.gallery_limit,
                                        replace=False))

    print(f"query={len(query_paths)}  gallery={len(gallery_paths)}")

    cfg = load_config()
    reid = VehicleReID(cfg)
    print(f"ReID backend: {reid._backend}")

    t0 = time.perf_counter()
    q_emb = embed_all(reid, query_paths)
    g_emb = embed_all(reid, gallery_paths)
    embed_time = time.perf_counter() - t0

    q_ids = np.array([parse_id_cam(p)[0] for p in query_paths])
    q_cams = np.array([parse_id_cam(p)[1] for p in query_paths])
    g_ids = np.array([parse_id_cam(p)[0] for p in gallery_paths])
    g_cams = np.array([parse_id_cam(p)[1] for p in gallery_paths])

    ranks_at = {1: 0, 5: 0, 10: 0}
    aps = []
    n_valid = 0

    for i in range(len(q_emb)):
        dists = 1.0 - g_emb @ q_emb[i]           # cosine distance, embeddings are L2-normed
        # Standard single-camera exclusion: don't count the same camera as a
        # "gallery" match for this query -- that's not cross-camera ReID.
        valid = g_cams != q_cams[i]
        if valid.sum() == 0:
            continue
        n_valid += 1
        order = np.argsort(dists[valid])
        matched_ids = g_ids[valid][order]
        matches = matched_ids == q_ids[i]

        for k in ranks_at:
            if matches[:k].any():
                ranks_at[k] += 1
        aps.append(compute_ap(matches))

    result = {
        "backend": reid._backend,
        "n_query_evaluated": n_valid,
        "n_gallery": len(gallery_paths),
        "rank1": ranks_at[1] / n_valid if n_valid else 0.0,
        "rank5": ranks_at[5] / n_valid if n_valid else 0.0,
        "rank10": ranks_at[10] / n_valid if n_valid else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "embed_time_s": embed_time,
        "embed_ms_per_image": embed_time / (len(q_emb) + len(g_emb)) * 1000,
    }

    print("\n=== VeRi-776 cross-camera ReID ===")
    print(f"backend: {result['backend']}")
    print(f"Rank-1:  {result['rank1']:.3f}")
    print(f"Rank-5:  {result['rank5']:.3f}")
    print(f"Rank-10: {result['rank10']:.3f}")
    print(f"mAP:     {result['mAP']:.3f}")
    print(f"embed latency: {result['embed_ms_per_image']:.1f} ms/image")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
