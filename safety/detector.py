"""Block 1: person detection, tracking and pose keypoints (YOLOv8n-pose)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO keypoint indices used by YOLOv8-pose
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_ANKLE, R_ANKLE = 15, 16
FACE_KEYPOINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Person:
    track_id: int | None       # temporary tracker number, never stored or linked to a name
    box: Box
    conf: float
    keypoints: np.ndarray      # (17, 2) pixel coordinates
    keypoint_conf: np.ndarray  # (17,)
    foot: tuple[float, float]  # point where the person stands, in pixels
    foot_from_ankles: bool     # False = ankles hidden, used bottom-center of the box


def foot_point(keypoints: np.ndarray, keypoint_conf: np.ndarray, box: Box,
               min_conf: float) -> tuple[tuple[float, float], bool]:
    """Midpoint of the visible ankles, or bottom-center of the box when both are hidden."""
    visible = [i for i in (L_ANKLE, R_ANKLE) if keypoint_conf[i] >= min_conf]
    if visible:
        x, y = keypoints[visible].mean(axis=0)
        return (float(x), float(y)), True
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2), False


def is_full_body(person: Person, min_conf: float) -> bool:
    """Head and at least one ankle visible (used by the gate check)."""
    head = person.keypoint_conf[NOSE] >= min_conf
    return bool(head and person.foot_from_ankles)


def openvino_dir(pt_path: Path, imgsz: int) -> Path:
    # Name must end in "_openvino_model" so ultralytics recognises the format
    return pt_path.with_name(f"{pt_path.stem}_{imgsz}_openvino_model")


def ensure_openvino(pt_path: Path, imgsz: int) -> Path:
    """Export the .pt model to OpenVINO once and reuse it afterwards."""
    target = openvino_dir(pt_path, imgsz)
    if not target.exists():
        from ultralytics import YOLO

        exported = Path(YOLO(str(pt_path)).export(format="openvino", imgsz=imgsz, half=False))
        exported.rename(target)
    return target


class PoseTracker:
    def __init__(self, weights: str, imgsz: int, device: str = "cpu", backend: str = "openvino",
                 tracker: str = "bytetrack.yaml", min_keypoint_conf: float = 0.5):
        from ultralytics import YOLO  # heavy import, kept out of module load

        path = Path(weights)
        if backend == "openvino":
            path = ensure_openvino(path, imgsz)
        self.model = YOLO(str(path), task="pose")
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker
        self.min_keypoint_conf = min_keypoint_conf

    def update(self, frame: np.ndarray) -> list[Person]:
        result = self.model.track(frame, persist=True, imgsz=self.imgsz, device=self.device,
                                  tracker=self.tracker, classes=[0], verbose=False)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = (result.boxes.id.int().cpu().tolist() if result.boxes.id is not None
               else [None] * len(boxes))
        kps = result.keypoints.xy.cpu().numpy()
        kconf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
                 else np.ones(kps.shape[:2]))

        people = []
        for box, conf, tid, kp, kc in zip(boxes, confs, ids, kps, kconf, strict=True):
            box = tuple(float(v) for v in box)
            foot, from_ankles = foot_point(kp, kc, box, self.min_keypoint_conf)
            people.append(Person(tid, box, float(conf), kp, kc, foot, from_ankles))
        return people
