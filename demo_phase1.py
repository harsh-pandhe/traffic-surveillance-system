"""Phase 1 acceptance demo: classify + enhance synthetic scenes, dump images."""
import os
import cv2
import numpy as np

from src.preprocessing.scene_classifier import SceneClassifier, DAY, NIGHT, FOG, RAIN
from src.preprocessing.enhancements import FrameEnhancer

OUT = "outputs/phase1"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def base_scene():
    """A textured 'road' scene: gradient + lane lines + random blobs."""
    img = np.zeros((360, 640, 3), np.uint8)
    img[:] = (120, 120, 120)
    cv2.rectangle(img, (0, 250), (640, 360), (90, 90, 90), -1)   # road
    for x in range(40, 640, 80):
        cv2.line(img, (x, 300), (x + 30, 360), (230, 230, 230), 4)  # lane dashes
    for _ in range(60):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = tuple(int(v) for v in rng.integers(0, 360, 2))
        cv2.circle(img, (p[1] % 640, p[0]), int(rng.integers(3, 12)), c, -1)
    return img


def make_night(img):
    return (img.astype(np.float32) * 0.18).astype(np.uint8)          # very dark


def make_fog(img):
    # Physically-plausible haze: I = J*t + A*(1-t), t decays with row (depth).
    h, w = img.shape[:2]
    A = 210.0
    rows = np.linspace(0.9, 0.35, h).reshape(h, 1, 1)               # t: near->far
    t = np.repeat(np.repeat(rows, w, axis=1), 3, axis=2)
    hazed = img.astype(np.float32) * t + A * (1.0 - t)
    hazed = cv2.GaussianBlur(hazed.astype(np.uint8), (0, 0), 2)     # mild blur
    return hazed


def make_rain(img):
    out = img.copy()
    out[:, :, 0] = np.clip(out[:, :, 0].astype(int) + 45, 0, 255)    # blue cast
    for _ in range(400):                                            # streaks
        x, y = int(rng.integers(0, 640)), int(rng.integers(0, 330))
        cv2.line(out, (x, y), (x + 2, y + 18), (200, 200, 200), 1)
    return out


clf = SceneClassifier()
enh = FrameEnhancer()

scenes = {
    "DAY":   base_scene(),
    "NIGHT": make_night(base_scene()),
    "FOG":   make_fog(base_scene()),
    "RAIN":  make_rain(base_scene()),
}

print(f"{'expected':8s} | {'predicted':8s} | {'match':5s} | metrics")
print("-" * 88)
ok = 0
for expected, frame in scenes.items():
    res = clf.classify(frame)
    match = "YES" if res.label == expected else "no"
    ok += res.label == expected
    m = res.metrics
    print(f"{expected:8s} | {res.label:8s} | {match:5s} | "
          f"lum={m['luminance']:6.1f} lap={m['laplacian_var']:7.1f} "
          f"sat={m['saturation']:5.1f} b/r={m['blue_red_ratio']:.2f} "
          f"hf={m['hf_energy']:.3f}")

    enhanced = enh.enhance(frame, res.label)
    cv2.imwrite(f"{OUT}/{expected}_before.png", frame)
    cv2.imwrite(f"{OUT}/{expected}_after.png", enhanced)

print("-" * 88)
print(f"scene accuracy: {ok}/4   images written to {OUT}/")
