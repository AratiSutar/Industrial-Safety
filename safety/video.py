"""Open a webcam, video file or RTSP stream and yield frames to process."""

from collections.abc import Iterator

import cv2
import numpy as np


def open_source(source: str) -> cv2.VideoCapture:
    # A bare number means a local webcam index
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def frames(source: str, every_n: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) for every Nth frame. Frames are never written to disk."""
    cap = open_source(source)
    index = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % every_n == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, frame
            index += 1
    finally:
        cap.release()
