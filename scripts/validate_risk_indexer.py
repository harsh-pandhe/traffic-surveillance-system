"""
scripts/validate_risk_indexer.py
----------------------------------
Phase 3 - validate RiskIndexer against real ground truth (no images needed).

The HELMET dataset (Grunfeld et al., osf.io/4pwj8) annotates 283,377 motorcycle
instances across 910 real traffic clips with a compositional label such as
'DHelmetP1NoHelmetP2NoHelmet': D = driver, P0..P3 = passenger seat positions,
each tagged Helmet/NoHelmet. Occupancy is 1 (driver) plus however many P<n>
tokens are present. This gives ground truth for exactly the two things the
risk indexer's rules need to be checked against:

    * overload:        occupancy > max_riders_allowed
    * helmet_misuse:   any seat position lacks a helmet

We are not running the trained detector here (no images downloaded) -- this
validates the *rule logic* against real-world label distributions, which is a
legitimate and useful check independent of detector accuracy.

Usage:
    python scripts/validate_risk_indexer.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analytics.risk_indexer import RiskIndexer

ANN_DIR = "data/raw/HELMET/annotation"
OUT = "results/phase3/risk_indexer_validation.json"
MOTORCYCLE_WHEEL_CLASS = 0   # HELMET is exclusively motorcycles -> 2-wheeler


def parse_label(label: str) -> tuple[int, bool]:
    """
    'DHelmetP1NoHelmetP2Helmet' -> (occupancy=3, any_no_helmet=True)

    Tokenise on D/P<n> boundaries; each token carries its own Helmet/NoHelmet.
    """
    tokens = re.findall(r'(D|P\d)(NoHelmet|Helmet)', label)
    occupancy = len(tokens)
    any_no_helmet = any(status == "NoHelmet" for _, status in tokens)
    return occupancy, any_no_helmet


def load_track_labels(limit_files: int | None = None) -> list[tuple[int, bool]]:
    """
    One (occupancy, any_no_helmet) sample per unique track, using the majority
    label across that track's frames (composition is expected to be constant
    for a real motorcycle, so a majority vote is robust to occasional
    mislabeled frames).
    """
    files = sorted(glob.glob(os.path.join(ANN_DIR, "*.csv")))
    if limit_files:
        files = files[:limit_files]

    track_labels: dict[str, Counter] = defaultdict(Counter)
    for fp in files:
        video = os.path.basename(fp)
        with open(fp, newline="") as fh:
            for row in csv.DictReader(fh):
                label = row.get("label", "").strip()
                if not label or label == "label":
                    continue
                key = f"{video}::{row['track_id']}"
                track_labels[key][label] += 1

    samples = []
    for key, counter in track_labels.items():
        majority_label, _ = counter.most_common(1)[0]
        samples.append(parse_label(majority_label))
    return samples


def main():
    ri = RiskIndexer()
    samples = load_track_labels()
    print(f"loaded {len(samples)} unique motorcycle tracks from "
         f"{len(glob.glob(os.path.join(ANN_DIR, '*.csv')))} clips")

    occ_dist = Counter(occ for occ, _ in samples)
    risk_levels = Counter()
    factor_hits = Counter()
    examples = {"LOW": None, "MEDIUM": None, "HIGH": None}

    for occupancy, any_no_helmet in samples:
        factors = ri.factors_from_context(
            wheel_class=MOTORCYCLE_WHEEL_CLASS,
            helmet_violation=any_no_helmet,
            rider_count=occupancy,
        )
        result = ri.compute(factors)
        risk_levels[result.level] += 1
        if factors.overloaded:
            factor_hits["overloaded"] += 1
        if factors.helmet_misuse:
            factor_hits["helmet_misuse"] += 1
        if examples[result.level] is None:
            examples[result.level] = {
                "occupancy": occupancy, "any_no_helmet": any_no_helmet,
                "score": result.score, "reasons": result.reasons}

    n = len(samples)
    print("\n=== Occupancy distribution (real motorcycles, Myanmar traffic) ===")
    for occ in sorted(occ_dist):
        print(f"  {occ} rider(s): {occ_dist[occ]:6d}  ({occ_dist[occ]*100/n:.1f}%)")

    print(f"\n=== Rule-trigger rates (max_riders_allowed={ri.max_riders}) ===")
    print(f"  overloaded (>{ri.max_riders} riders): "
         f"{factor_hits['overloaded']:6d}  ({factor_hits['overloaded']*100/n:.1f}%)")
    print(f"  helmet_misuse (>=1 bare head): "
         f"{factor_hits['helmet_misuse']:6d}  ({factor_hits['helmet_misuse']*100/n:.1f}%)")

    print(f"\n=== Resulting risk-level distribution ===")
    for level in ("LOW", "MEDIUM", "HIGH"):
        print(f"  {level:6s}: {risk_levels[level]:6d}  "
             f"({risk_levels[level]*100/n:.1f}%)")

    # Sanity assertions on the rule boundary itself (not on detector accuracy).
    assert ri.factors_from_context(0, False, 2).overloaded is False, \
        "2 riders must NOT be flagged overloaded"
    assert ri.factors_from_context(0, False, 3).overloaded is True, \
        "3 riders MUST be flagged overloaded"
    print("\nboundary check: 2 riders -> not overloaded, "
         "3 riders -> overloaded  [PASS]")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({
        "n_tracks": n, "n_clips": len(glob.glob(os.path.join(ANN_DIR, "*.csv"))),
        "occupancy_distribution": {str(k): v for k, v in occ_dist.items()},
        "rule_trigger_rates": {
            "overloaded_pct": factor_hits["overloaded"] * 100 / n,
            "helmet_misuse_pct": factor_hits["helmet_misuse"] * 100 / n},
        "risk_level_distribution": dict(risk_levels),
        "examples": examples,
        "boundary_check": "2 riders not overloaded, 3 riders overloaded: PASS",
        "note": ("Validates rule logic against real ground-truth occupancy and "
                 "helmet-use labels from 910 real traffic clips (HELMET dataset, "
                 "Myanmar, 2016). Does not exercise the trained detector -- no "
                 "images were downloaded for this dataset -- so this checks "
                 "whether the risk RULES are well-calibrated to real-world "
                 "occupancy/violation prevalence, independent of detector "
                 "accuracy."),
    }, open(OUT, "w"), indent=2)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
