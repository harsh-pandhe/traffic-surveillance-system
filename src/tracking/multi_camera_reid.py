"""
src/tracking/multi_camera_reid.py
----------------------------------
Phase 3 - Cross-camera vehicle Re-Identification (ReID).

Extracts OSNet appearance embeddings for each vehicle crop and matches them
against a time-limited global gallery, so the same physical vehicle keeps one
global identity as it moves between distinct camera feeds.

Backends (auto-selected, graceful degradation):
    1. torchreid OSNet (preferred, matches config `reid.model_name`)
    2. A colour-histogram embedding fallback (keeps the API working without the
       heavy dependency / weights).

Matching uses cosine distance; entries older than `gallery_ttl` seconds expire.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.config import load_config, resolve_path


@dataclass
class GalleryEntry:
    global_id: int
    embedding: np.ndarray
    last_seen: float
    camera_id: str


class VehicleReID:
    """OSNet-based cross-camera vehicle re-identification."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        r = cfg["reid"]
        self.model_name = r["model_name"]
        self.match_threshold = float(r["match_threshold"])
        self.gallery_ttl = float(r["gallery_ttl"])
        self.device = cfg["runtime"]["device"]
        self._weights = resolve_path(r["weights"])

        self.gallery: Dict[int, GalleryEntry] = {}
        self._next_gid = 1
        self._backend, self._extractor = self._init_backend()

    # ------------------------------------------------------------------ #
    def _init_backend(self):
        try:
            from torchreid.utils import FeatureExtractor
            import os
            model_path = self._weights if os.path.isfile(self._weights) else ""
            extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=model_path,
                device=self.device,
            )
            print(f"[VehicleReID] OSNet backend ready ({self.model_name}).")
            return "osnet", extractor
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[VehicleReID] torchreid unavailable ({exc}); "
                  f"using colour-histogram embedding fallback.")
            return "histogram", None

    # ------------------------------------------------------------------ #
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised appearance embedding for a vehicle crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)

        if self._backend == "osnet":
            feats = self._extractor([crop_bgr])          # (1, D)
            vec = np.asarray(feats[0], dtype=np.float32).ravel()
        else:
            vec = self._histogram_embedding(crop_bgr)

        norm = np.linalg.norm(vec) + 1e-6
        return vec / norm

    @staticmethod
    def _histogram_embedding(crop_bgr: np.ndarray) -> np.ndarray:
        """Cheap HSV colour histogram used when OSNet is unavailable."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten().astype(np.float32)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def _expire(self, now: float) -> None:
        stale = [gid for gid, e in self.gallery.items()
                 if now - e.last_seen > self.gallery_ttl]
        for gid in stale:
            del self.gallery[gid]

    # ------------------------------------------------------------------ #
    def match(self, crop_bgr: np.ndarray, camera_id: str) -> int:
        """
        Match a vehicle crop to a global identity.
        Returns an existing global_id if within threshold, else registers a new one.
        """
        now = time.time()
        self._expire(now)
        emb = self.embed(crop_bgr)

        best_gid, best_dist = -1, float("inf")
        for gid, entry in self.gallery.items():
            d = self._cosine_distance(emb, entry.embedding)
            if d < best_dist:
                best_dist, best_gid = d, gid

        if best_gid != -1 and best_dist <= self.match_threshold:
            # Update running embedding with an exponential moving average.
            entry = self.gallery[best_gid]
            entry.embedding = 0.7 * entry.embedding + 0.3 * emb
            entry.embedding /= (np.linalg.norm(entry.embedding) + 1e-6)
            entry.last_seen = now
            entry.camera_id = camera_id
            return best_gid

        gid = self._next_gid
        self._next_gid += 1
        self.gallery[gid] = GalleryEntry(gid, emb, now, camera_id)
        return gid


if __name__ == "__main__":
    reid = VehicleReID()
    car = (np.random.rand(128, 64, 3) * 255).astype(np.uint8)
    g1 = reid.match(car, "cam_A")
    g2 = reid.match(car.copy(), "cam_B")          # same appearance
    other = (np.random.rand(128, 64, 3) * 255).astype(np.uint8)
    g3 = reid.match(other, "cam_A")
    print(f"g1={g1} g2={g2} (expect equal) g3={g3} (expect new)")
