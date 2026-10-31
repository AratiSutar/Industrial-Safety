"""Face blurring. Applied to every frame before it is shown or stored.

Two layers, so a face is blurred even when one of them misses it:
  1. the head of every tracked person (from the pose keypoints)
  2. every face found by a small dedicated face detector (YuNet), e.g. a forklift driver whose
     body is hidden, or a person the tracker hasn't confirmed yet
"""

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from safety.detector import FACE_KEYPOINTS, Box, Person

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def face_box(keypoints: np.ndarray, keypoint_conf: np.ndarray, box: Box, min_conf: float) -> Box:
    """Area to blur: eyebrows down to the chin, so a hard hat on top of the head stays visible.

    When the face keypoints are uncertain, blur the whole head zone instead (privacy wins).
    """
    x1, y1, x2, y2 = box
    visible = [i for i in FACE_KEYPOINTS if keypoint_conf[i] >= min_conf]
    if len(visible) >= 2:
        pts = keypoints[visible]
        cx = float(pts[:, 0].mean())
        spread = float(np.ptp(pts, axis=0).max())
        # Generous, but sized from body HEIGHT: a crouching person's box is very wide, and sizing
        # from width blurred the whole helmet away (seen on a real image of a kneeling grinder)
        half = max(spread, (y2 - y1) * 0.1)
        top = float(pts[:, 1].min()) - half * 0.3   # just above the eyebrows
        bottom = float(pts[:, 1].max()) + half      # below the chin
        return (cx - half, top, cx + half, bottom)
    return (x1, y1, x2, y1 + (y2 - y1) * 0.3)


def _pixelate(out: np.ndarray, box: Box) -> None:
    h, w = out.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    small = cv2.resize(out[y1:y2, x1:x2], (6, 6), interpolation=cv2.INTER_AREA)
    out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def blur_faces(frame: np.ndarray, people: list[Person], min_conf: float, faces: list[Box] = ()) -> np.ndarray:
    """Return a copy of the frame with every face pixelated. The original is not modified."""
    out = frame.copy()
    for p in people:
        _pixelate(out, face_box(p.keypoints, p.keypoint_conf, p.box, min_conf))
    for box in faces:
        _pixelate(out, box)
    return out


def ensure_face_model(path: Path) -> Path:
    """Download the YuNet face model once and refuse it if the checksum doesn't match."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".download")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != YUNET_SHA256:
        raise RuntimeError(f"Face model {path} has an unexpected checksum; delete it to download again")
    return path


class FaceDetector:
    """YuNet on a downscaled copy of the frame. Boxes are padded, because blurring too much is safer."""

    def __init__(self, model_path: Path, width: int = 640, score: float = 0.5, pad: float = 0.25):
        self.width = width
        self.pad = pad
        self._det = cv2.FaceDetectorYN.create(str(ensure_face_model(model_path)), "", (width, width),
                                              score_threshold=score)
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray) -> list[Box]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.width / w)
        size = (round(w * scale), round(h * scale))
        if size != self._size:
            self._det.setInputSize(size)
            self._size = size
        small = frame if scale == 1.0 else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        _, faces = self._det.detect(small)
        boxes = []
        for f in faces if faces is not None else []:
            x, y, fw, fh = (float(v) / scale for v in f[:4])
            px, py = fw * self.pad, fh * self.pad
            boxes.append((x - px, y - py, x + fw + px, y + fh + py))
        return boxes
