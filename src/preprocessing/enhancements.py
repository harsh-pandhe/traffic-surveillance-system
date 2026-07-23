"""
src/preprocessing/enhancements.py
----------------------------------
Phase 1 - Adaptive frame enhancement driven by the scene label.

    NIGHT  -> CLAHE on the L-channel (LAB) to recover shadow detail without
              blowing out highlights.
    FOG    -> Dark Channel Prior (DCP) dehazing, refined with a fast guided
              filter for a smooth transmission map.
    RAIN   -> Light DCP dehaze + non-local-means denoise to suppress streaks.
    DAY    -> Pass-through (optionally a mild denoise).

The public entry point `enhance(frame, scene_label)` returns a new BGR frame,
ready to feed straight into the detection stage.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.config import load_config
from src.preprocessing.scene_classifier import DAY, NIGHT, FOG, RAIN


class FrameEnhancer:
    """Scene-conditioned image enhancement (all CPU, all OpenCV/NumPy)."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        e = cfg["enhancements"]
        self.clahe_clip = float(e["clahe_clip_limit"])
        self.clahe_grid = tuple(e["clahe_tile_grid"])
        self.dcp_omega = float(e["dcp_omega"])
        self.dcp_patch = int(e["dcp_patch_size"])
        self.dcp_t_min = float(e["dcp_t_min"])
        self.guided_radius = int(e["guided_radius"])
        self.guided_eps = float(e["guided_eps"])
        self.denoise_h = int(e["denoise_h"])

        # Reuse a single CLAHE object (thread-unsafe but fine per-worker).
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid
        )

    # ------------------------------------------------------------------ #
    # NIGHT: CLAHE on LAB L-channel                                       #
    # ------------------------------------------------------------------ #
    def apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------ #
    # FOG / RAIN: Dark Channel Prior dehazing                            #
    # ------------------------------------------------------------------ #
    def _dark_channel(self, img: np.ndarray, patch: int) -> np.ndarray:
        """Per-pixel min across channels, then a min-filter (erosion)."""
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
        return cv2.erode(min_channel, kernel)

    def _atmospheric_light(self, img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        """Estimate global airlight from the brightest 0.1% dark-channel pixels."""
        h, w = dark.shape
        n_pixels = h * w
        n_top = max(int(n_pixels * 0.001), 1)
        idx = np.argsort(dark.ravel())[-n_top:]
        flat = img.reshape(n_pixels, 3)
        # Pick the pixel with max intensity among the candidates.
        brightest = flat[idx]
        return brightest[np.argmax(brightest.sum(axis=1))].astype(np.float32)

    def _guided_filter(self, guide: np.ndarray, src: np.ndarray,
                       radius: int, eps: float) -> np.ndarray:
        """Fast guided filter to refine the coarse transmission map."""
        guide = guide.astype(np.float32)
        src = src.astype(np.float32)
        mean_i = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        return mean_a * guide + mean_b

    def dehaze(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Full DCP dehaze pipeline with guided-filter transmission refinement."""
        img = frame_bgr.astype(np.float32) / 255.0

        dark = self._dark_channel(img, self.dcp_patch)
        # `img` is already normalised to [0, 1]; _atmospheric_light returns a
        # pixel in the same units, so do NOT divide by 255 again here.
        A = self._atmospheric_light(img, dark)
        A = np.clip(A, 1e-3, 1.0)

        # Coarse transmission estimate.
        norm = img / A.reshape(1, 1, 3)
        transmission = 1.0 - self.dcp_omega * self._dark_channel(norm, self.dcp_patch)

        # Refine with guided filter using the grayscale image as guide.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        transmission = self._guided_filter(
            gray, transmission, self.guided_radius, self.guided_eps
        )
        transmission = np.clip(transmission, self.dcp_t_min, 1.0)

        # Recover the scene radiance J = (I - A) / t + A.
        t = transmission[:, :, np.newaxis]
        recovered = (img - A.reshape(1, 1, 3)) / t + A.reshape(1, 1, 3)
        recovered = np.clip(recovered, 0.0, 1.0)
        return (recovered * 255.0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Denoise (used for RAIN)                                            #
    # ------------------------------------------------------------------ #
    def denoise(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(
            frame_bgr, None, self.denoise_h, self.denoise_h, 7, 21
        )

    # ------------------------------------------------------------------ #
    # Dispatcher                                                          #
    # ------------------------------------------------------------------ #
    def enhance(self, frame_bgr: np.ndarray, scene_label: str) -> np.ndarray:
        """Route the frame to the right enhancement based on the scene label."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("enhance() received an empty frame")

        if scene_label == NIGHT:
            return self.apply_clahe(frame_bgr)
        if scene_label == FOG:
            return self.dehaze(frame_bgr)
        if scene_label == RAIN:
            return self.denoise(self.dehaze(frame_bgr))
        # DAY or unknown: return untouched frame.
        return frame_bgr


if __name__ == "__main__":
    enh = FrameEnhancer()
    dummy = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    for label in (DAY, NIGHT, FOG, RAIN):
        out = enh.enhance(dummy, label)
        print(f"{label:6s} -> output shape {out.shape}, dtype {out.dtype}")
