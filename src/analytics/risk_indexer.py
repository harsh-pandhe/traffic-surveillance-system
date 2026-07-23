"""
src/analytics/risk_indexer.py
------------------------------
Phase 3 - Real-time Hazard Score & Alert generation.

Computes a weighted Risk Index R per tracked vehicle / scene:

    R = w1 * (Wheel Count Violation)
      + w2 * (Helmet Misuse)
      + w3 * (Rider Count > 2)
      + w4 * (Wrong Way)

Risk levels:
    LOW     : R < low_max        (default 3)
    MEDIUM  : low_max <= R < medium_max   (default 3..6)
    HIGH    : R >= medium_max    (default 6)

Wrong-way detection is derived from the sign of a track's horizontal/vertical
displacement relative to a configured allowed direction (defaults to "any", i.e.
disabled, until the operator sets a lane direction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.config import load_config

LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

# BGR banner colours per level (used by visualization).
LEVEL_COLORS = {LOW: (0, 180, 0), MEDIUM: (0, 170, 255), HIGH: (0, 0, 255)}


@dataclass
class RiskFactors:
    """Boolean/scalar inputs that feed the risk equation for one entity."""
    wheel_violation: bool = False
    helmet_misuse: bool = False
    overloaded: bool = False           # rider count > allowed
    wrong_way: bool = False
    rider_count: int = 0


@dataclass
class RiskResult:
    score: float
    level: str
    factors: RiskFactors
    reasons: List[str] = field(default_factory=list)


class RiskIndexer:
    """Weighted, thresholded hazard scoring."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        r = cfg["risk_indexer"]
        w = r["weights"]
        self.w1 = float(w["w1_wheel_violation"])
        self.w2 = float(w["w2_helmet_misuse"])
        self.w3 = float(w["w3_overload"])
        self.w4 = float(w["w4_wrong_way"])
        self.low_max = float(r["thresholds"]["low_max"])
        self.medium_max = float(r["thresholds"]["medium_max"])
        self.max_riders = int(r["max_riders_allowed"])
        # Allowed vehicle wheel classes (violation if outside this set).
        self.allowed_wheel_classes = set(
            cfg["wheel_classifier"].get("allowed_classes", [0, 1, 2, 3])
        )

    # ------------------------------------------------------------------ #
    def compute(self, factors: RiskFactors) -> RiskResult:
        """Compute R and its level from pre-derived boolean factors."""
        score = (
            self.w1 * int(factors.wheel_violation)
            + self.w2 * int(factors.helmet_misuse)
            + self.w3 * int(factors.overloaded)
            + self.w4 * int(factors.wrong_way)
        )
        reasons: List[str] = []
        if factors.wheel_violation:
            reasons.append("illegal wheel-count / vehicle type")
        if factors.helmet_misuse:
            reasons.append("helmet non-compliance")
        if factors.overloaded:
            reasons.append(f"overloaded ({factors.rider_count} riders)")
        if factors.wrong_way:
            reasons.append("wrong-way movement")
        return RiskResult(score=score, level=self._level(score),
                          factors=factors, reasons=reasons)

    def _level(self, score: float) -> str:
        if score < self.low_max:
            return LOW
        if score < self.medium_max:
            return MEDIUM
        return HIGH

    # ------------------------------------------------------------------ #
    def factors_from_context(
        self,
        wheel_class: Optional[int],
        helmet_violation: bool,
        rider_count: int,
        wrong_way: bool = False,
    ) -> RiskFactors:
        """Build RiskFactors from raw pipeline outputs."""
        wheel_violation = (
            wheel_class is not None and wheel_class not in self.allowed_wheel_classes
        )
        return RiskFactors(
            wheel_violation=wheel_violation,
            helmet_misuse=helmet_violation,
            overloaded=rider_count > self.max_riders,
            wrong_way=wrong_way,
            rider_count=rider_count,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def detect_wrong_way(track_history: List[Tuple[float, float]],
                         allowed_axis: str = "none",
                         allowed_sign: int = 1,
                         min_disp: float = 40.0) -> bool:
        """
        Heuristic wrong-way test from a track's centroid history.

        Args:
            track_history: list of (cx, cy) centroids, oldest -> newest.
            allowed_axis:  "x", "y", or "none" (disabled).
            allowed_sign:  +1 or -1 = legal direction of increasing coordinate.
            min_disp:      minimum displacement (px) before a verdict is trusted.
        """
        if allowed_axis == "none" or len(track_history) < 2:
            return False
        (x0, y0), (x1, y1) = track_history[0], track_history[-1]
        disp = (x1 - x0) if allowed_axis == "x" else (y1 - y0)
        if abs(disp) < min_disp:
            return False
        moving_sign = 1 if disp > 0 else -1
        return moving_sign != allowed_sign


if __name__ == "__main__":
    ri = RiskIndexer()
    f = ri.factors_from_context(wheel_class=3, helmet_violation=True,
                                rider_count=3, wrong_way=True)
    res = ri.compute(f)
    print(f"score={res.score} level={res.level} reasons={res.reasons}")
