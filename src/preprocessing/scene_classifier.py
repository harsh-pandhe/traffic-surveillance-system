"""
src/preprocessing/scene_classifier.py
--------------------------------------
Phase 1 - Adaptive scene understanding.

Classifies a single BGR frame into one of four weather/lighting regimes:
    DAY, NIGHT, FOG, RAIN

The decision is a rule-based fusion of cheap, explainable image statistics so it
runs in real time on CPU and needs no training data:

    * mean luminance          -> darkness (NIGHT)
    * Laplacian variance       -> global sharpness (low => FOG haze)
    * HSV saturation mean      -> colour washout (low => FOG)
    * blue/red channel ratio   -> cool blue cast (RAIN)
    * high-frequency energy    -> rain streaks / texture

Returns a `SceneResult` with the label plus the raw metrics so downstream
logging / debugging can inspect *why* a decision was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import cv2
import numpy as np

from utils.config import load_config

# Canonical scene labels used across the whole pipeline.
DAY = "DAY"
NIGHT = "NIGHT"
FOG = "FOG"
RAIN = "RAIN"
SCENE_LABELS = (DAY, NIGHT, FOG, RAIN)


@dataclass
class SceneResult:
    """Container for the classification decision and the metrics behind it."""
    label: str
    metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        m = ", ".join(f"{k}={v:.2f}" for k, v in self.metrics.items())
        return f"{self.label} ({m})"


class SceneClassifier:
    """Rule-based day/night/fog/rain classifier."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        p = cfg["scene_classifier"]
        self.night_luminance_max = float(p["night_luminance_max"])
        self.fog_laplacian_max = float(p["fog_laplacian_max"])
        self.fog_saturation_max = float(p["fog_saturation_max"])
        self.rain_bluecast_min = float(p["rain_bluecast_min"])
        self.rain_hf_energy_min = float(p["rain_hf_energy_min"])

    # ---- individual metric extractors -----------------------------------

    @staticmethod
    def _mean_luminance(gray: np.ndarray) -> float:
        return float(gray.mean())

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Global sharpness. Haze/fog smears edges => low variance."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _saturation_mean(hsv: np.ndarray) -> float:
        return float(hsv[:, :, 1].mean())

    @staticmethod
    def _blue_red_ratio(bgr: np.ndarray) -> float:
        b = float(bgr[:, :, 0].mean()) + 1e-6
        r = float(bgr[:, :, 2].mean()) + 1e-6
        return b / r

    @staticmethod
    def _high_freq_energy(gray: np.ndarray) -> float:
        """
        Normalised high-frequency energy via FFT magnitude outside a low-pass
        radius. Rain streaks inject fine oriented high-frequency content.
        """
        g = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
        f = np.fft.fftshift(np.fft.fft2(g))
        mag = np.abs(f)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        high = mag[radius > (min(h, w) * 0.15)].sum()
        total = mag.sum() + 1e-6
        return float(high / total)

    # ---- main entry point ------------------------------------------------

    def classify(self, frame_bgr: np.ndarray) -> SceneResult:
        """Classify a single BGR frame. Returns a SceneResult."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("classify() received an empty frame")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        luminance = self._mean_luminance(gray)
        lap_var = self._laplacian_variance(gray)
        sat = self._saturation_mean(hsv)
        br_ratio = self._blue_red_ratio(frame_bgr)
        hf_energy = self._high_freq_energy(gray)

        metrics = {
            "luminance": luminance,
            "laplacian_var": lap_var,
            "saturation": sat,
            "blue_red_ratio": br_ratio,
            "hf_energy": hf_energy,
        }

        label = self._decide(luminance, lap_var, sat, br_ratio, hf_energy)
        return SceneResult(label=label, metrics=metrics)

    def _decide(self, lum: float, lap: float, sat: float,
                br: float, hf: float) -> str:
        """
        Priority-ordered rule fusion:
          1. Very dark  -> NIGHT
          2. Low sharpness + washed-out colour -> FOG
          3. Cool blue cast + high-frequency streaks -> RAIN
          4. Otherwise  -> DAY
        """
        # 1) Night dominates: darkness changes every other statistic.
        if lum < self.night_luminance_max:
            return NIGHT

        # 2) Fog: hazy scenes lose edge energy and colour saturation.
        if lap < self.fog_laplacian_max and sat < self.fog_saturation_max:
            return FOG

        # 3) Rain: blue-shifted, streaky, still reasonably bright.
        if br > self.rain_bluecast_min and hf > self.rain_hf_energy_min:
            return RAIN

        # 4) Default clear daytime.
        return DAY


if __name__ == "__main__":
    # Quick smoke test on a synthetic dark frame.
    clf = SceneClassifier()
    dark = np.full((360, 640, 3), 20, dtype=np.uint8)
    print("synthetic dark frame ->", clf.classify(dark))
    bright = np.full((360, 640, 3), 180, dtype=np.uint8)
    print("synthetic bright frame ->", clf.classify(bright))
