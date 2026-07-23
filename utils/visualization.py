"""
utils/visualization.py
-----------------------
Drawing helpers: bounding boxes, telemetry overlays, and risk banners.

All functions take and return BGR frames (OpenCV convention) and never mutate
the input in place unless explicitly noted.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.analytics.risk_indexer import LEVEL_COLORS, LOW, MEDIUM, HIGH

# Per-helmet-class BGR colours (index == class id).
HELMET_COLORS = {
    0: (0, 0, 255),      # No Helmet - red
    1: (0, 200, 0),      # Full-Face - green
    2: (0, 200, 200),    # Half-Face - yellow
    3: (0, 128, 255),    # Strap Unfastened - orange
    4: (200, 0, 200),    # Helmet on Handlebar - magenta
    5: (200, 100, 0),    # Helmet on Arm - blue
    6: (255, 200, 0),    # Rider - cyan
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(frame: np.ndarray, text: str, org: Tuple[int, int],
               color: Tuple[int, int, int], scale: float = 0.5) -> None:
    """Draw text with a filled background box for readability."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 2, y), color, -1)
    cv2.putText(frame, text, (x + 1, y - baseline), _FONT, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_box(frame: np.ndarray, bbox: List[float], label: str,
             color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    _put_label(frame, label, (x1, max(y1, 14)), color)
    return frame


def draw_helmet_detections(frame: np.ndarray, detections) -> np.ndarray:
    """Draw all HelmetDetection boxes with class-coded colours."""
    out = frame.copy()
    for d in detections:
        color = HELMET_COLORS.get(d.cls_id, (255, 255, 255))
        draw_box(out, d.bbox, f"{d.cls_name} {d.conf:.2f}", color)
    return out


def draw_track(frame: np.ndarray, bbox: List[float], track_id: int,
               wheel_name: str, global_id: Optional[int] = None,
               risk_level: Optional[str] = None) -> np.ndarray:
    """Draw a tracked vehicle box with ID + wheel class + optional risk tint."""
    color = LEVEL_COLORS.get(risk_level, (200, 200, 200))
    gid = f" G{global_id}" if global_id is not None else ""
    draw_box(frame, bbox, f"ID{track_id}{gid} {wheel_name}", color)
    return frame


def draw_risk_banner(frame: np.ndarray, level: str, score: float,
                     reasons: Optional[List[str]] = None) -> np.ndarray:
    """Overlay a top risk banner coloured by level."""
    out = frame.copy()
    h, w = out.shape[:2]
    color = LEVEL_COLORS.get(level, (128, 128, 128))
    cv2.rectangle(out, (0, 0), (w, 40), color, -1)
    text = f"RISK: {level}  (R={score:.1f})"
    cv2.putText(out, text, (10, 28), _FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if reasons:
        sub = " | ".join(reasons)[:90]
        cv2.putText(out, sub, (10, 58), _FONT, 0.5, color, 1, cv2.LINE_AA)
    return out


def draw_telemetry(frame: np.ndarray, scene: str, fps: float,
                   ram_mb: float, cpu: float,
                   frame_idx: int) -> np.ndarray:
    """Bottom-left HUD with scene label + runtime telemetry."""
    out = frame.copy()
    h = out.shape[0]
    lines = [
        f"Scene: {scene}",
        f"FPS: {fps:.1f}",
        f"RAM: {ram_mb:.0f} MB   CPU: {cpu:.0f}%",
        f"Frame: {frame_idx}",
    ]
    y = h - 10 - 18 * (len(lines) - 1)
    for line in lines:
        cv2.putText(out, line, (10, y), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # shadow
        y += 18
    return out


def draw_demographics(frame: np.ndarray, bbox: List[float], demo) -> np.ndarray:
    """Annotate an exposed-face rider with age group + gender."""
    if demo is None or demo.age_group == "UNKNOWN":
        return frame
    x1, y1, _x2, _y2 = [int(v) for v in bbox]
    txt = f"{demo.gender[:1]} / {demo.age_group}"
    _put_label(frame, txt, (x1, max(y1 - 4, 14)), (120, 0, 120), scale=0.45)
    return frame


if __name__ == "__main__":
    canvas = np.full((360, 640, 3), 60, dtype=np.uint8)
    canvas = draw_box(canvas, [50, 80, 200, 300], "No Helmet 0.91",
                      HELMET_COLORS[0])
    canvas = draw_risk_banner(canvas, HIGH, 7.5, ["helmet non-compliance"])
    canvas = draw_telemetry(canvas, "NIGHT", 14.2, 512, 63, 120)
    cv2.imwrite("outputs/_viz_demo.png", canvas) if False else None
    print("visualization smoke test OK", canvas.shape)
