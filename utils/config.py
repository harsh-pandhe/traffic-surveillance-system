"""
utils/config.py
---------------
Tiny helper that loads the central YAML configuration once and exposes it as a
plain nested dict. Keeps every module free of hard-coded paths / thresholds.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

# Absolute path to config/settings.yaml regardless of the current working dir.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "settings.yaml")


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load and cache the YAML config. Repeated calls return the same dict."""
    cfg_path = path or _DEFAULT_CONFIG_PATH
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_project_root"] = _PROJECT_ROOT
    return cfg


def resolve_path(rel_or_abs: str) -> str:
    """Resolve a config path relative to the project root if it is not absolute."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(_PROJECT_ROOT, rel_or_abs)


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2, default=str))
