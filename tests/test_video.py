import time

import numpy as np

from safety.video import LatestFrameReader, is_live


class FakeCamera:
    """Sends 60 frames at ~200 fps, much faster than the consumer reads them."""

    def __init__(self, n=60):
        self.n, self.i = n, 0

    def read(self):
        if self.i >= self.n:
            return False, None
        time.sleep(0.005)
        self.i += 1
        return True, np.full((4, 4, 3), self.i, np.uint8)


def test_slow_consumer_gets_newest_frames_without_backlog():
    reader = LatestFrameReader(FakeCamera())
    reader.start()
    seen = []
    while (item := reader.latest(timeout=1.0)) is not None:
        seen.append(item[0])
        time.sleep(0.05)  # slow "model"
    reader.join(timeout=1.0)

    assert seen == sorted(set(seen))      # always newer, never the same frame twice
    assert len(seen) < 30                 # old frames were dropped, not queued
    assert seen[-1] >= 50                 # still reached the end of the stream


def test_live_source_detection():
    assert is_live("0") and is_live("rtsp://10.0.0.5/stream") and is_live("HTTP://cam/live")
    assert not is_live("videos/test.mp4")
