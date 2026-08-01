"""
scripts/run_demographics.py
-----------------------------
Phase 3 - offline demographics processing.

Runs DeepFace on face crops queued by the live pipeline
(`DemographicsEstimator.save_for_offline_analysis`, called automatically from
`estimate()` whenever an in-process DeepFace instance is unavailable).

Why this exists as a separate script, not a pipeline step: on this hardware,
torch's bundled CUDA runtime targets a newer NVIDIA driver than is installed.
Loading a second Ultralytics/torch model in a process that has also imported
TensorFlow (DeepFace's backend) segfaults -- confirmed by bisecting pipeline
construction step by step. DeepFace runs cleanly in a process that never
imports torch, which is exactly what this script is. This module imports
ONLY cv2, numpy, and deepface -- never `torch`, `ultralytics`, or anything from
`src.models` / `src.tracking` -- keep it that way.

Usage:
    python scripts/run_demographics.py
    python scripts/run_demographics.py --queue-dir outputs/demographics_queue
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force CPU before importing TensorFlow (DeepFace's backend). This process
# never imports torch, so unlike the live pipeline this alone is sufficient.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

OUT = "results/phase3/demographics.json"
AGE_BUCKETS = [(0, 12, "Child"), (13, 19, "Teen"),
               (20, 39, "Adult"), (40, 59, "Middle-Aged"), (60, 200, "Senior")]


def age_to_group(age):
    if age is None:
        return "UNKNOWN"
    for lo, hi, name in AGE_BUCKETS:
        if lo <= age <= hi:
            return name
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", default="outputs/demographics_queue")
    ap.add_argument("--detector-backend", default="opencv")
    args = ap.parse_args()

    crops = sorted(glob.glob(os.path.join(args.queue_dir, "*.jpg")))
    if not crops:
        print(f"No queued face crops in {args.queue_dir}/. "
             "Run the pipeline with demographics.enabled=true first.")
        return

    from deepface import DeepFace
    import cv2

    results = []
    for path in crops:
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            res = DeepFace.analyze(
                img_path=img, actions=["age", "gender"],
                detector_backend=args.detector_backend,
                enforce_detection=False, silent=True)
            r = res[0] if isinstance(res, list) else res
            age = r.get("age")
            gender = r.get("dominant_gender", "UNKNOWN")
            gconf = 0.0
            if isinstance(r.get("gender"), dict) and gender in r["gender"]:
                gconf = float(r["gender"][gender]) / 100.0
            results.append({"file": os.path.basename(path),
                            "age": age, "age_group": age_to_group(age),
                            "gender": gender, "gender_confidence": gconf})
        except Exception as exc:
            results.append({"file": os.path.basename(path), "error": str(exc)})

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"processed {len(results)} crops, {n_ok} succeeded")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({
        "n_crops": len(results), "n_succeeded": n_ok,
        "results": results,
        "note": ("Runs offline (separate process from the live pipeline) "
                 "because torch + TensorFlow in one process segfaults on "
                 "this hardware -- see src/analytics/demographics.py "
                 "module docstring. Age estimates should be read as "
                 "indicative: face crops are small (median 38px in Gate B "
                 "measurement), below the ~64px DeepFace/FairFace expects "
                 "for reliable age estimation; gender is more robust at low "
                 "resolution."),
    }, open(OUT, "w"), indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
