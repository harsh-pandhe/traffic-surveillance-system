"""
src/preprocessing/scene_features.py
------------------------------------
Phase 1 (learned classifier): rich, cheap, explainable feature extractor.

Turns a BGR frame into a fixed-length numeric vector describing its lighting,
haze, colour cast, and texture. The same features power both the rule-based
`SceneClassifier` (a few of them) and the learned `LearnedSceneClassifier`
(all of them). All features are CPU-cheap and normalisation-friendly.

Feature vector (FEATURE_NAMES order):
    luminance_mean, luminance_std, value_mean,
    laplacian_var, log_laplacian, edge_density, contrast,
    saturation_mean, saturation_std, colorfulness,
    blue_red_ratio, blue_green_ratio,
    hf_energy, dark_channel_mean, bright_pixel_ratio
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

FEATURE_NAMES: List[str] = [
    "luminance_mean", "luminance_std", "value_mean",
    "laplacian_var", "log_laplacian", "edge_density", "contrast",
    "saturation_mean", "saturation_std", "colorfulness",
    "blue_red_ratio", "blue_green_ratio",
    "hf_energy", "dark_channel_mean", "bright_pixel_ratio",
]
NUM_FEATURES = len(FEATURE_NAMES)


def _high_freq_energy(gray: np.ndarray) -> float:
    g = cv2.resize(gray, (256, 256)).astype(np.float32) / 255.0
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = mag[radius > (min(h, w) * 0.15)].sum()
    return float(high / (mag.sum() + 1e-6))


def _dark_channel_mean(bgr: np.ndarray, patch: int = 15) -> float:
    """Mean dark channel (min over channels then min-filter). High => hazy."""
    img = bgr.astype(np.float32) / 255.0
    min_c = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(min_c, kernel)
    return float(dark.mean())


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness metric."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


def extract_features(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a NUM_FEATURES float32 vector for a BGR frame."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("extract_features got an empty frame")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lum_mean = float(gray.mean())
    lum_std = float(gray.std())
    value_mean = float(hsv[:, :, 2].mean())

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    log_lap = float(np.log1p(lap_var))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    contrast = lum_std  # std of luminance == RMS contrast proxy

    sat_mean = float(hsv[:, :, 1].mean())
    sat_std = float(hsv[:, :, 1].std())
    colorful = _colorfulness(frame_bgr)

    b = float(frame_bgr[:, :, 0].mean()) + 1e-6
    g = float(frame_bgr[:, :, 1].mean()) + 1e-6
    r = float(frame_bgr[:, :, 2].mean()) + 1e-6
    br_ratio = b / r
    bg_ratio = b / g

    hf = _high_freq_energy(gray)
    dark = _dark_channel_mean(frame_bgr)
    bright_ratio = float((gray > 220).mean())

    return np.array([
        lum_mean, lum_std, value_mean,
        lap_var, log_lap, edge_density, contrast,
        sat_mean, sat_std, colorful,
        br_ratio, bg_ratio,
        hf, dark, bright_ratio,
    ], dtype=np.float32)


if __name__ == "__main__":
    img = (np.random.rand(200, 320, 3) * 255).astype(np.uint8)
    v = extract_features(img)
    print(f"vector len={len(v)} (expected {NUM_FEATURES})")
    for n, val in zip(FEATURE_NAMES, v):
        print(f"  {n:20s} {val:.4f}")
