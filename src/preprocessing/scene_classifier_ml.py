"""
src/preprocessing/scene_classifier_ml.py
------------------------------------------
Phase 1 (learned classifier): a trained DAY/NIGHT/FOG/RAIN classifier.

A RandomForest over the `scene_features` vector. Drop-in compatible with the
rule-based `SceneClassifier` — exposes `classify(frame) -> SceneResult` with the
same label vocabulary — so `main.py` can swap one for the other transparently.

If a trained model file is missing, `classify()` transparently falls back to the
rule-based classifier, so nothing breaks before training is run.

Train with `train_scene_classifier.py`, which persists the model to
`weights/scene_classifier.joblib`.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from utils.config import load_config, resolve_path
from src.preprocessing.scene_features import extract_features, FEATURE_NAMES
from src.preprocessing.scene_classifier import (
    SceneClassifier, SceneResult, SCENE_LABELS,
)


class LearnedSceneClassifier:
    """RandomForest scene classifier with rule-based fallback."""

    def __init__(self, config: dict | None = None, model_path: str | None = None):
        self.cfg = config or load_config()
        mp = model_path or self.cfg.get("scene_classifier_ml", {}).get(
            "model_path", "weights/scene_classifier.joblib"
        )
        self.model_path = resolve_path(mp)
        self.labels: List[str] = list(SCENE_LABELS)
        self._model = None
        self._rule_fallback = SceneClassifier(self.cfg)
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not os.path.isfile(self.model_path):
            print(f"[LearnedSceneClassifier] no model at {self.model_path}; "
                  f"using rule-based fallback until trained.")
            return
        import joblib
        # Safe: this .joblib is produced by our own train_scene_classifier.py and
        # lives inside the project's weights/ dir — not an untrusted source.
        bundle = joblib.load(self.model_path)
        self._model = bundle["model"]
        self.labels = bundle["labels"]
        print(f"[LearnedSceneClassifier] loaded model ({len(self.labels)} classes) "
              f"from {self.model_path}")

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    def classify(self, frame_bgr: np.ndarray) -> SceneResult:
        """Predict a scene label. Falls back to rules if not trained."""
        if self._model is None:
            return self._rule_fallback.classify(frame_bgr)

        feats = extract_features(frame_bgr).reshape(1, -1)
        pred_idx = int(self._model.predict(feats)[0])
        label = self.labels[pred_idx]

        metrics = {}
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(feats)[0]
            metrics = {self.labels[i]: float(p) for i, p in enumerate(proba)}
        return SceneResult(label=label, metrics=metrics)

    # ------------------------------------------------------------------ #
    @staticmethod
    def feature_importances(model_path: str) -> Optional[dict]:
        """Return {feature_name: importance} for a trained model (introspection)."""
        if not os.path.isfile(model_path):
            return None
        import joblib
        bundle = joblib.load(model_path)
        model = bundle["model"]
        if not hasattr(model, "feature_importances_"):
            return None
        return dict(sorted(
            zip(FEATURE_NAMES, model.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))


if __name__ == "__main__":
    clf = LearnedSceneClassifier()
    dummy = (np.random.rand(200, 320, 3) * 255).astype(np.uint8)
    print("trained:", clf.is_trained, "->", clf.classify(dummy).label)
