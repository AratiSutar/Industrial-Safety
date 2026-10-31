"""Face blurring. Applied to every frame before it is shown or stored."""

import cv2
import numpy as np

from safety.detector import FACE_KEYPOINTS, Box, Person


def face_box(keypoints: np.ndarray, keypoint_conf: np.ndarray, box: Box, min_conf: float) -> Box:
    """Area to blur. When the face keypoints are uncertain, blur the whole head zone instead."""
    x1, y1, x2, y2 = box
    visible = [i for i in FACE_KEYPOINTS if keypoint_conf[i] >= min_conf]
    if len(visible) >= 2:
        pts = keypoints[visible]
        cx, cy = pts.mean(axis=0)
        spread = float(np.ptp(pts, axis=0).max())
        half = max(spread, (x2 - x1) * 0.25)  # generous: never smaller than a quarter body width
        return (cx - half, cy - half * 1.3, cx + half, cy + half)  # extra room above for hair/hat
    return (x1, y1, x2, y1 + (y2 - y1) * 0.3)


def blur_faces(frame: np.ndarray, people: list[Person], min_conf: float) -> np.ndarray:
    """Return a copy of the frame with every face pixelated. The original is not modified."""
    out = frame.copy()
    h, w = out.shape[:2]
    for p in people:
        fx1, fy1, fx2, fy2 = face_box(p.keypoints, p.keypoint_conf, p.box, min_conf)
        x1, y1 = max(0, int(fx1)), max(0, int(fy1))
        x2, y2 = min(w, int(fx2)), min(h, int(fy2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        small = cv2.resize(roi, (6, 6), interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return out
