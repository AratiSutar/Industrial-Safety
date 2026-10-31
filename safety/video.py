"""Open a webcam, video file or RTSP stream and yield frames to process."""

import threading
import time
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

    def __init__(self, cap, pace_fps: float | None = None):
        super().__init__(daemon=True, name="camera-reader")
        self.cap = cap
        self.pace_fps = pace_fps  # video files: play at their real speed, like a live camera
        self.running = True
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._frame: np.ndarray | None = None
        self._index = -1

    def run(self) -> None:
        start = time.perf_counter()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.pace_fps:
                wait = start + (self._index + 1) / self.pace_fps - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            with self._lock:
                self._frame = frame
                self._index += 1
                self._new.set()
        with self._lock:
            self.running = False
            self._new.set()  # wake the consumer so it can see the stream ended

    def latest(self, timeout: float = 5.0) -> tuple[int, np.ndarray] | None:
        """Newest unseen frame, or None if the stream ended or stalled for `timeout` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            # State is only changed under the lock (frame, running and the wake-up signal together),
            # so a wake-up can't be lost and an empty wake-up never ends the stream early.
            with self._lock:
                if self._frame is not None:
                    item = (self._index, self._frame)
                    self._frame = None  # never hand out the same frame twice
                    self._new.clear()
                    return item
                if not self.running:
                    return None
                self._new.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._new.wait(remaining):
                return None

    def stop(self) -> None:
        self.running = False


def frames(source: str, every_n: int = 1, realtime: bool = False) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame). Frames are never written to disk.

    Video files: every Nth frame, in order (slow motion if analysis is slower than the video).
    Live cameras, or files with realtime=True: always the newest frame, skipping what we can't keep up with.
    """
    cap = open_source(source)
    if is_live(source) or realtime:
        reader = LatestFrameReader(cap, pace_fps=None if is_live(source) else cap.get(cv2.CAP_PROP_FPS) or 25)
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
