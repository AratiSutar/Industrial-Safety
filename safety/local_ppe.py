"""Local PPE check: a YOLO detector trained on PPE classes, run on this PC (no network, no cost per check).

Same interface as the Jev client (check(item, jpeg) -> PpeAnswer) and fed the same face-blurred crops,
so the inspector and the gate decision work unchanged whichever engine answers.
"""

import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from safety.config import JEV_ITEMS
from safety.jev_client import PpeAnswer


def normalize(name: str) -> str:
    """'NO-Hardhat', 'no_hardhat' and 'No Hardhat' all become 'nohardhat'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def class_ids(names: dict[int, str], classes: dict[str, dict[str, tuple[str, ...]]]) -> dict[str, dict[str, set[int]]]:
    """Model class ids meaning present / missing for each item. Names the model doesn't have are skipped."""
    by_name = {normalize(n): i for i, n in names.items()}
    return {item: {side: {by_name[normalize(n)] for n in wanted.get(side, ()) if normalize(n) in by_name}
                   for side in ("present", "missing")}
            for item, wanted in classes.items()}


def decide(item: str, detections: list[tuple[int, float]], ids: dict[str, set[int]], latency_s: float) -> PpeAnswer:
    """The strongest present / missing detection wins; a tie counts as missing.

    Nothing found is "unclear", never "no": the detector not seeing a helmet doesn't prove there is none.
    """
    present = max((conf for cls, conf in detections if cls in ids["present"]), default=0.0)
    missing = max((conf for cls, conf in detections if cls in ids["missing"]), default=0.0)
    if present == missing == 0.0:
        return PpeAnswer(item, "unclear", 0.0, latency_s, error="nothing detected")
    if present > missing:
        return PpeAnswer(item, "yes", present, latency_s)
    return PpeAnswer(item, "no", missing, latency_s)


class LocalPpeChecker:
    """One model shared by the inspector's threads, behind a lock (YOLO predict isn't thread-safe)."""

    def __init__(self, model, classes: dict[str, dict[str, tuple[str, ...]]], imgsz: int = 320,
                 device: str = "cpu", min_detection_conf: float = 0.25):
        self.model = model
        ids = class_ids(model.names, classes)
        self.ids = {item: ids.get(item, {"present": set(), "missing": set()}) for item in JEV_ITEMS}
        self.imgsz = imgsz
        self.device = device
        self.min_detection_conf = min_detection_conf
        self._lock = threading.Lock()

    @classmethod
    def load(cls, weights: str, classes: dict[str, dict[str, tuple[str, ...]]], imgsz: int, device: str = "cpu",
             backend: str = "pytorch", min_detection_conf: float = 0.25) -> "LocalPpeChecker":
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"local PPE model {path} not found (put YOLO PPE weights there, see README)")
        from ultralytics import YOLO  # heavy import, kept out of module load

        from safety.detector import ensure_openvino

        if backend == "openvino":
            path = ensure_openvino(path, imgsz)
        return cls(YOLO(str(path), task="detect"), classes, imgsz, device, min_detection_conf)

    def unmapped_items(self) -> list[str]:
        """Items the model has no class for: they will always be answered "unclear"."""
        return [item for item, ids in self.ids.items() if not ids["present"] and not ids["missing"]]

    def check(self, item: str, jpeg: bytes) -> PpeAnswer:
        start = time.perf_counter()
        ids = self.ids.get(item)
        if not ids or not (ids["present"] or ids["missing"]):
            return PpeAnswer(item, "unclear", 0.0, 0.0, f"model has no class for {item}")
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return PpeAnswer(item, "error", 0.0, time.perf_counter() - start, "could not decode crop")
        try:
            with self._lock:
                result = self.model.predict(img, imgsz=self.imgsz, device=self.device,
                                            conf=self.min_detection_conf, verbose=False)[0]
        except Exception as e:
            # Any failure counts as "error", which keeps the gate locked (same as the Jev client)
            return PpeAnswer(item, "error", 0.0, time.perf_counter() - start, f"{type(e).__name__}: {e}")
        boxes = result.boxes
        detections = ([] if boxes is None else
                      [(int(c), float(p)) for c, p in zip(boxes.cls.tolist(), boxes.conf.tolist(), strict=True)])
        return decide(item, detections, ids, time.perf_counter() - start)
