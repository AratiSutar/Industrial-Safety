import threading
import time

import numpy as np

from safety.video import LatestFrameReader, is_live


class FakeCamera:
    """Sends 60 frames at ~200 fps, much faster than the consumer reads them."""

    def __init__(self, n=60, delay=0.005):
        self.n, self.i, self.delay = n, 0, delay

    def read(self):
        if self.i >= self.n:
            return False, None
        time.sleep(self.delay)
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


def test_fast_consumer_never_ends_a_running_stream_early():
    # Consumer polls as fast as it can: an empty wake-up must not be mistaken for "stream ended"
    for _ in range(20):
        reader = LatestFrameReader(FakeCamera(n=30, delay=0.001))
        reader.start()
        last = -1
        while (item := reader.latest(timeout=1.0)) is not None:
            last = item[0]
        reader.join(timeout=1.0)
        assert last == 29


def test_end_of_stream_is_noticed_immediately():
    reader = LatestFrameReader(FakeCamera(n=5, delay=0.001))
    reader.start()
    t0 = time.perf_counter()
    while reader.latest(timeout=2.0) is not None:
        pass
    assert time.perf_counter() - t0 < 0.5   # not stuck waiting for the timeout


def test_stalled_camera_times_out():
    class Stalled:
        def read(self):
            threading.Event().wait(5)
            return False, None

    reader = LatestFrameReader(Stalled())
    reader.start()
    t0 = time.perf_counter()
    assert reader.latest(timeout=0.3) is None
    assert 0.25 < time.perf_counter() - t0 < 1.0
    reader.stop()


def test_live_source_detection():
    assert is_live("0") and is_live("rtsp://10.0.0.5/stream") and is_live("HTTP://cam/live")
    assert not is_live("videos/test.mp4")
