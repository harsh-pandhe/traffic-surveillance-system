"""
src/tracking/helmet_video_loader.py
------------------------------------
Phase 3 - HELMET dataset loader (motorcycle traffic video).

The HELMET dataset (Myanmar, 910 clips x 100 frames @ 10 fps, 1920x1080) is the
piece UA-DETRAC could not provide: it is *motorcycle* traffic, tracked, with
per-rider helmet annotations. That gives us three things at once:

  1. the 2-wheeler class, which UA-DETRAC contains none of;
  2. rider occupancy per motorcycle, which validates the "more than two riders"
     overload rule in the risk indexer;
  3. per-position helmet compliance, which matches the Phase 2 detector's
     driver/passenger taxonomy.

Annotation format is one CSV per clip:

    track_id,frame_id,x,y,w,h,label

where `label` is a compositional string encoding every occupant and whether
they wear a helmet, e.g.

    DHelmet                          driver helmeted, riding alone
    DNoHelmet                        driver bare-headed, alone
    DHelmetP1NoHelmet                driver helmeted, one bare-headed pillion
    DNoHelmetP1NoHelmetP2NoHelmet    three riders, none helmeted (triple riding)

Positions seen in the data are D (driver) and P0/P1/P2 (passengers).
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2

# Matches one occupant token: a position followed by its helmet state.
# Order matters -- "NoHelmet" must be tried before "Helmet".
_OCCUPANT_RE = re.compile(r"(D|P\d)(NoHelmet|Helmet)")

WHEEL_CLASS_MOTORCYCLE = 0   # project taxonomy: 0 == 2-wheeler


@dataclass
class Occupant:
    position: str        # "D", "P0", "P1", "P2"
    helmeted: bool

    @property
    def is_driver(self) -> bool:
        return self.position == "D"


@dataclass
class HelmetAnnotation:
    """One motorcycle in one frame."""
    track_id: str
    frame_id: int
    bbox: List[float]              # [x1, y1, x2, y2]
    label: str
    occupants: List[Occupant] = field(default_factory=list)

    # --- derived properties used by the risk indexer -------------------
    @property
    def rider_count(self) -> int:
        return len(self.occupants)

    @property
    def helmet_violations(self) -> int:
        return sum(1 for o in self.occupants if not o.helmeted)

    @property
    def any_violation(self) -> bool:
        return self.helmet_violations > 0

    @property
    def driver_helmeted(self) -> Optional[bool]:
        for o in self.occupants:
            if o.is_driver:
                return o.helmeted
        return None

    @property
    def overloaded(self) -> bool:
        """More than two people on a two-wheeler (triple riding)."""
        return self.rider_count > 2

    @property
    def wheel_class(self) -> int:
        return WHEEL_CLASS_MOTORCYCLE


def parse_label(label: str) -> List[Occupant]:
    """Turn a compositional HELMET label into a list of occupants."""
    return [Occupant(position=p, helmeted=(h == "Helmet"))
            for p, h in _OCCUPANT_RE.findall(label or "")]


@dataclass
class HelmetClip:
    """One 100-frame clip: annotations grouped by frame."""
    name: str
    images_dir: Optional[str]
    by_frame: Dict[int, List[HelmetAnnotation]] = field(default_factory=dict)

    def frame_image(self, frame_id: int):
        """Load the image for a frame, if the image part is present on disk."""
        if not self.images_dir:
            return None
        for pattern in (f"{frame_id}.jpg", f"{frame_id:03d}.jpg",
                        f"{frame_id:05d}.jpg", f"img{frame_id:05d}.jpg"):
            p = os.path.join(self.images_dir, pattern)
            if os.path.isfile(p):
                return cv2.imread(p)
        return None

    def frames(self) -> Iterator[Tuple[int, List[HelmetAnnotation]]]:
        for fid in sorted(self.by_frame):
            yield fid, self.by_frame[fid]

    def stats(self) -> Dict:
        occ, viol, trip, n = defaultdict(int), 0, 0, 0
        tracks = set()
        for _fid, anns in self.frames():
            for a in anns:
                n += 1
                tracks.add(a.track_id)
                occ[a.rider_count] += 1
                viol += a.any_violation
                trip += a.overloaded
        return {"annotations": n, "tracks": len(tracks),
                "rider_count_hist": dict(occ),
                "with_violation": viol, "triple_riding": trip}


def load_clip(csv_path: str, images_root: Optional[str] = None) -> HelmetClip:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    images_dir = None
    if images_root:
        cand = os.path.join(images_root, name)
        images_dir = cand if os.path.isdir(cand) else None

    clip = HelmetClip(name=name, images_dir=images_dir)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                fid = int(row["frame_id"])
                x, y = float(row["x"]), float(row["y"])
                w, h = float(row["w"]), float(row["h"])
            except (ValueError, KeyError, TypeError):
                continue
            label = row.get("label", "") or ""
            ann = HelmetAnnotation(
                track_id=row["track_id"], frame_id=fid,
                bbox=[x, y, x + w, y + h], label=label,
                occupants=parse_label(label),
            )
            clip.by_frame.setdefault(fid, []).append(ann)
    return clip


def discover_clips(root: str, images_root: Optional[str] = None,
                   limit: Optional[int] = None,
                   require_images: bool = False) -> List[HelmetClip]:
    """Load clips from <root>/annotation/*.csv."""
    ann_dir = os.path.join(root, "annotation")
    if not os.path.isdir(ann_dir):
        ann_dir = root
    out: List[HelmetClip] = []
    for f in sorted(os.listdir(ann_dir)):
        if not f.endswith(".csv"):
            continue
        clip = load_clip(os.path.join(ann_dir, f), images_root)
        if require_images and not clip.images_dir:
            continue
        out.append(clip)
        if limit and len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/HELMET"
    imgs = sys.argv[2] if len(sys.argv) > 2 else None
    clips = discover_clips(root, imgs, limit=3)
    for c in clips:
        print(f"{c.name}: {c.stats()}")
    print("\nlabel parsing examples:")
    for lbl in ["DHelmet", "DNoHelmet", "DHelmetP1NoHelmet",
                "DNoHelmetP1NoHelmetP2NoHelmet"]:
        occ = parse_label(lbl)
        print(f"  {lbl:32s} riders={len(occ)} "
              f"violations={sum(1 for o in occ if not o.helmeted)}")
