"""Block 2: convert image pixels to floor coordinates in meters (one calibration per camera)."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_GOOD_ERROR_M = 0.10  # calibration points should map back within 10 cm


@dataclass
class Calibration:
    camera: str
    image_points: list[tuple[float, float]]  # pixels
    floor_points: list[tuple[float, float]]  # meters
    H: np.ndarray                            # 3x3, pixels -> meters
    error_m: float                           # mean error of the calibration points

    def to_floor(self, points) -> np.ndarray:
        """(N, 2) pixel points -> (N, 2) floor points in meters."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2))
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, floor_points) -> np.ndarray:
        """(N, 2) floor points in meters -> (N, 2) pixels. Used to draw zones on the video.

        Floor points beyond the camera's horizon have no image position and come back as NaN.
        """
        pts = np.asarray(floor_points, dtype=np.float64).reshape(-1, 2)
        h = np.hstack([pts, np.ones((len(pts), 1))]) @ np.linalg.inv(self.H).T
        w = h[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = h[:, :2] / w
        out[w[:, 0] <= 0] = np.nan
        return out


def compute(camera: str, image_points, floor_points) -> Calibration:
    img = np.asarray(image_points, dtype=np.float64)
    floor = np.asarray(floor_points, dtype=np.float64)
    if img.shape != floor.shape or img.ndim != 2 or img.shape[1] != 2:
        raise ValueError("image_points and floor_points must be matching lists of (x, y)")
    if len(img) < 4:
        raise ValueError("Need at least 4 point pairs")

    H, _ = cv2.findHomography(img, floor, method=0)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Points are degenerate (e.g. 3 or more on one line); pick spread-out floor points")

    cal = Calibration(camera, [tuple(p) for p in img.tolist()], [tuple(p) for p in floor.tolist()], H, 0.0)
    cal.error_m = float(np.linalg.norm(cal.to_floor(img) - floor, axis=1).mean())
    return cal


def save(cal: Calibration, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{cal.camera}.json"
    path.write_text(json.dumps({
        "camera": cal.camera,
        "image_points": cal.image_points,
        "floor_points": cal.floor_points,
        "H": cal.H.tolist(),
        "error_m": cal.error_m,
    }, indent=2), encoding="utf-8")
    return path


def load(camera: str, directory: Path) -> Calibration | None:
    path = directory / f"{camera}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Calibration(
        camera=raw["camera"],
        image_points=[tuple(p) for p in raw["image_points"]],
        floor_points=[tuple(p) for p in raw["floor_points"]],
        H=np.asarray(raw["H"], dtype=np.float64),
        error_m=float(raw["error_m"]),
    )
