"""Cut one image per PPE item out of a person, sized for Jev.

Privacy: the hard-hat crop stops above the eyes, and vest/boot crops are taken from the
face-blurred frame, so no recognisable face is ever sent to the API.
"""

import cv2
import numpy as np

from safety.detector import (
    L_ANKLE,
    L_EYE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_EYE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    Box,
    Person,
)

# Jev rejects (does not resize) images outside 280-1568 px per side
MIN_SIDE, MAX_SIDE = 280, 1568
MIN_CROP_PX = 16  # smaller than this is too little detail to judge


def _visible(person: Person, idx: tuple[int, ...], min_conf: float) -> np.ndarray:
    return person.keypoints[[i for i in idx if person.keypoint_conf[i] >= min_conf]]


def _band(box: Box, top: float, bottom: float) -> Box:
    """Horizontal slice of the person box, as fractions of its height."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1, y1 + h * top, x2, y1 + h * bottom)


def head_region(person: Person, min_conf: float) -> Box:
    """Top of the head, cut off above the eyes."""
    x1, y1, x2, y2 = person.box
    eyes = _visible(person, (L_EYE, R_EYE), min_conf)
    if len(eyes):
        eye_y = float(eyes[:, 1].min())
        cx = float(eyes[:, 0].mean())
        half = (x2 - x1) * 0.35
        top = max(y1 - (eye_y - y1) * 0.5, y1 - (y2 - y1) * 0.05)  # room for the helmet
        return (cx - half, top, cx + half, eye_y - (eye_y - y1) * 0.1)
    return _band(person.box, -0.03, 0.12)


def torso_region(person: Person, min_conf: float) -> Box:
    pts = _visible(person, (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP), min_conf)
    if len(pts) >= 3:
        (px1, py1), (px2, py2) = pts.min(axis=0), pts.max(axis=0)
        mx, my = (px2 - px1) * 0.25, (py2 - py1) * 0.1
        return (px1 - mx, py1 - my, px2 + mx, py2 + my)
    return _band(person.box, 0.18, 0.6)


def feet_region(person: Person, min_conf: float) -> Box:
    pts = _visible(person, (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE), min_conf)
    x1, _, x2, y2 = person.box
    if len(pts) >= 2:
        (px1, py1), (px2, _) = pts.min(axis=0), pts.max(axis=0)
        mx = max((px2 - px1) * 0.3, (x2 - x1) * 0.15)
        return (px1 - mx, py1, px2 + mx, y2 + (y2 - py1) * 0.1)
    return _band(person.box, 0.75, 1.03)


def cut(frame: np.ndarray, region: Box) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(region[0])), max(0, int(region[1]))
    x2, y2 = min(w, int(region[2])), min(h, int(region[3]))
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def fit_for_jev(img: np.ndarray) -> np.ndarray:
    """Scale into 280-1568 px per side, padding very thin crops with grey."""
    h, w = img.shape[:2]
    scale = min(max(MIN_SIDE / min(h, w), 1.0), MAX_SIDE / max(h, w))
    if scale != 1.0:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)
    h, w = img.shape[:2]
    pad_h, pad_w = max(0, MIN_SIDE - h), max(0, MIN_SIDE - w)
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2,
                                 cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img


def to_jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", fit_for_jev(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def ppe_crops(frame: np.ndarray, blurred: np.ndarray, person: Person, min_conf: float) -> dict[str, bytes | None]:
    """JPEG bytes per PPE item, or None when that part of the body isn't visible enough."""
    regions = {
        "hard_hat": (frame, head_region(person, min_conf)),     # stops above the eyes
        "vest": (blurred, torso_region(person, min_conf)),
        "boots": (blurred, feet_region(person, min_conf)),
    }
    out = {}
    for item, (src, region) in regions.items():
        crop = cut(src, region)
        out[item] = to_jpeg(crop) if crop is not None else None
    return out
