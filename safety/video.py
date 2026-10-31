"""Open a webcam, video file or RTSP stream and yield frames to process."""

import threading
from collections.abc import Iterator

import cv2
import numpy as np

LIVE_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://")


def is_live(source: str) -> bool:
    return source.isdigit() or source.lower().startswith(LIVE_PREFIXES)


def open_source(source: str) -> cv2.VideoCapture:
    # A bare number means a local webcam index
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


class LatestFrameReader(threading.Thread):
    """Reads a live camera continuously and keeps only the newest frame.

    The camera sends ~30 fps but the CPU analyses ~5, so without this the backlog grows
    and alerts would arrive later and later.
    """

    def __init__(self, cap):
        super().__init__(daemon=True, name="camera-reader")
        self.cap = cap
        self.running = True
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._frame: np.ndarray | None = None
        self._index = -1

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            with self._lock:
                self._frame = frame
                self._index += 1
            self._new.set()
        self.running = False
        self._new.set()  # wake the consumer so it can see the stream ended

    def latest(self, timeout: float = 5.0) -> tuple[int, np.ndarray] | None:
        """Newest unseen frame, or None if the stream ended or stalled."""
        if not self._new.wait(timeout):
            return None
        with self._lock:
            self._new.clear()
            if self._frame is None:
                return None
            item = (self._index, self._frame)
            self._frame = None  # never hand out the same frame twice
            return item

    def stop(self) -> None:
        self.running = False


def frames(source: str, every_n: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame). Frames are never written to disk.

    Video files: every Nth frame, in order. Live cameras: always the newest frame.
    """
    cap = open_source(source)
    if is_live(source):
        reader = LatestFrameReader(cap)
        reader.start()
        try:
            while (item := reader.latest()) is not None:
                yield item
        finally:
            reader.stop()
            reader.join(timeout=1.0)
            cap.release()
        return

    index = 0
    try:
        while cap.grab():
            if index % every_n == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, frame
            index += 1
    finally:
        cap.release()
