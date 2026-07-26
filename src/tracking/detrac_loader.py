"""
src/tracking/detrac_loader.py
------------------------------
Phase 3 - UA-DETRAC dataset loader.

UA-DETRAC ships each sequence as a folder of JPEG frames plus one XML file of
ground-truth annotations. Crucially for this project, every box carries an
**occlusion** entry, which is what makes the multi-frame-voting ablation
possible: we can measure classification accuracy as a function of how occluded
a vehicle actually is, rather than guessing.

XML structure (abridged):

    <sequence name="MVI_20011">
      <frame density="7" num="1">
        <target_list>
          <target id="1">
            <box left="592.75" top="378.8" width="160.05" height="162.2"/>
            <attribute orientation="18.488" speed="6.859" trajectory_length="5"
                       truncation_ratio="0" vehicle_type="car"/>
            <occlusion>
              <region_overlap left="..." top="..." width="..." height="..."
                              occlusion_status="1" occlusion_id="2"/>
            </occlusion>
          </target>
        </target_list>
      </frame>
    </sequence>

`vehicle_type` is one of car / bus / van / others, which maps onto the project's
wheel-count taxonomy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2

# These XML files come from a third-party dataset mirror, so parse them with
# defusedxml (guards against XXE / billion-laughs) when it is available.
try:
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover - fallback if defusedxml is absent
    import xml.etree.ElementTree as ET
    print("[detrac_loader] defusedxml not installed; falling back to the stdlib "
          "XML parser. `pip install defusedxml` is recommended.")

# UA-DETRAC vehicle_type -> project wheel-count class id
#   0 = 2-wheeler, 1 = 3-wheeler, 2 = 4-wheeler, 3 = 6+ wheeler
DETRAC_TO_WHEEL = {
    "car": 2,
    "van": 2,
    "bus": 3,
    "truck": 3,
    "others": 2,
}


@dataclass
class DetracTarget:
    """One annotated vehicle in one frame."""
    target_id: int
    bbox: List[float]                 # [x1, y1, x2, y2]
    vehicle_type: str
    truncation_ratio: float = 0.0
    occluded: bool = False
    occlusion_area_frac: float = 0.0  # occluded area / box area

    @property
    def wheel_class(self) -> int:
        return DETRAC_TO_WHEEL.get(self.vehicle_type.lower(), 2)

    @property
    def occlusion_band(self) -> str:
        """Coarse band used to stratify the voting ablation."""
        f = self.occlusion_area_frac
        if f <= 0.01:
            return "none"
        if f < 0.25:
            return "light"
        if f < 0.5:
            return "medium"
        return "heavy"


@dataclass
class DetracFrame:
    num: int
    image_path: str
    targets: List[DetracTarget] = field(default_factory=list)

    def image(self):
        return cv2.imread(self.image_path)


class DetracSequence:
    """One UA-DETRAC sequence: frames + ground-truth targets."""

    def __init__(self, name: str, images_dir: str, xml_path: str):
        self.name = name
        self.images_dir = images_dir
        self.xml_path = xml_path
        self.frames: List[DetracFrame] = []
        self._parse()

    # ------------------------------------------------------------------ #
    def _parse(self) -> None:
        root = ET.parse(self.xml_path).getroot()
        for frame_el in root.findall("frame"):
            num = int(frame_el.get("num"))
            img = os.path.join(self.images_dir, f"img{num:05d}.jpg")
            frame = DetracFrame(num=num, image_path=img)

            for tgt in frame_el.findall(".//target"):
                box = tgt.find("box")
                if box is None:
                    continue
                left = float(box.get("left")); top = float(box.get("top"))
                w = float(box.get("width")); h = float(box.get("height"))
                bbox = [left, top, left + w, top + h]
                area = max(w * h, 1e-6)

                attr = tgt.find("attribute")
                vtype = attr.get("vehicle_type", "car") if attr is not None else "car"
                trunc = float(attr.get("truncation_ratio", 0)) if attr is not None else 0.0

                # Sum overlapping occlusion regions for this target.
                occ_area = 0.0
                occ_el = tgt.find("occlusion")
                if occ_el is not None:
                    for reg in occ_el.findall("region_overlap"):
                        ow = float(reg.get("width", 0))
                        oh = float(reg.get("height", 0))
                        occ_area += max(ow, 0) * max(oh, 0)

                frame.targets.append(DetracTarget(
                    target_id=int(tgt.get("id")),
                    bbox=bbox,
                    vehicle_type=vtype,
                    truncation_ratio=trunc,
                    occluded=occ_area > 0,
                    occlusion_area_frac=min(occ_area / area, 1.0),
                ))
            self.frames.append(frame)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[DetracFrame]:
        return iter(self.frames)

    def stats(self) -> Dict[str, int]:
        bands: Dict[str, int] = {}
        types: Dict[str, int] = {}
        n_targets = 0
        for f in self.frames:
            for t in f.targets:
                n_targets += 1
                bands[t.occlusion_band] = bands.get(t.occlusion_band, 0) + 1
                types[t.vehicle_type] = types.get(t.vehicle_type, 0) + 1
        return {"frames": len(self.frames), "targets": n_targets,
                "occlusion_bands": bands, "vehicle_types": types}


def discover_sequences(root: str, limit: Optional[int] = None
                       ) -> List[Tuple[str, str, str]]:
    """
    Find (name, images_dir, xml_path) triples under a UA-DETRAC root.

    Handles the common mirror layout:
        <root>/DETRAC-Images/<split>/<SEQ>/img00001.jpg
        <root>/DETRAC-Train-Annotations-XML/<SEQ>.xml
    """
    xml_dirs, img_dirs = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if "annotation" in base and any(f.endswith(".xml") for f in filenames):
            xml_dirs.append(dirpath)
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in filenames):
            img_dirs.append(dirpath)

    # Map sequence name -> image dir (sequence name is the folder holding frames)
    by_name = {os.path.basename(d): d for d in img_dirs}

    out: List[Tuple[str, str, str]] = []
    for xd in xml_dirs:
        for f in sorted(os.listdir(xd)):
            if not f.endswith(".xml"):
                continue
            name = os.path.splitext(f)[0]
            if name in by_name:
                out.append((name, by_name[name], os.path.join(xd, f)))
                if limit and len(out) >= limit:
                    return out
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/UA-DETRAC"
    seqs = discover_sequences(root, limit=3)
    if not seqs:
        print(f"No UA-DETRAC sequences found under {root}")
    for name, imgs, xml in seqs:
        s = DetracSequence(name, imgs, xml)
        print(f"{name}: {s.stats()}")
