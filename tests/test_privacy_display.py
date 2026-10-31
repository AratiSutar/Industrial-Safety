import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from safety.alerts import late_alarm
from safety.config import AlertSettings, Machine
from safety.detector import L_ANKLE, R_ANKLE, Person
from safety.display import feet_box, ui_scale
from safety.privacy import YUNET_SHA256, blur_faces, ensure_face_model
from safety.video import LatestFrameReader

BOX = (100.0, 100.0, 200.0, 500.0)  # 400 px tall person


def person(ankles: bool):
    kp, kc = np.zeros((17, 2)), np.full(17, 0.1)
    if ankles:
        kp[L_ANKLE], kp[R_ANKLE] = (130, 480), (170, 485)
        kc[L_ANKLE] = kc[R_ANKLE] = 0.9
    return Person(1, BOX, 0.9, kp, kc, (150, 482), ankles)


def test_feet_box_surrounds_both_ankles_and_reaches_the_floor():
    x1, y1, x2, y2 = feet_box(person(ankles=True))
    assert x1 < 130 and x2 > 170 and y1 < 480 and y2 >= 500


def test_feet_box_without_ankles_sits_at_bottom_of_body():
    x1, y1, x2, y2 = feet_box(person(ankles=False))
    assert y2 == 500 and 100 < x1 < x2 < 200 and y1 > 400


def test_extra_face_boxes_are_blurred_even_without_a_tracked_person():
    frame = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    out = blur_faces(frame, [], 0.5, faces=[(50, 50, 110, 120)])
    assert not np.array_equal(out[50:120, 50:110], frame[50:120, 50:110])
    assert np.array_equal(out[150:, :], frame[150:, :])


def test_face_model_with_wrong_checksum_is_refused(tmp_path):
    bad = tmp_path / "yunet.onnx"
    bad.write_bytes(b"not the real model")
    with pytest.raises(RuntimeError, match="checksum"):
        ensure_face_model(bad)
    assert len(YUNET_SHA256) == 64


def test_labels_scale_with_resolution():
    assert ui_scale(np.zeros((2160, 3840, 3), np.uint8)) == 3.0
    assert ui_scale(np.zeros((720, 1280, 3), np.uint8)) == 1.0


def test_small_danger_zone_is_flagged_large_one_is_not():
    s = AlertSettings()  # 5 fps, very close after 0.3 s, at half of danger_m
    small = Machine("m", "Press", "c", [(0, 0), (1, 0), (1, 1)], warn_m=1.5, danger_m=0.8)
    big = Machine("m", "Press", "c", [(0, 0), (1, 0), (1, 1)], warn_m=3.0, danger_m=2.0)
    walked, suggested = late_alarm(small, s)
    assert walked == pytest.approx(1.6 * 0.5) and suggested == pytest.approx(1.6)
    assert late_alarm(big, s) is None


class FastFile:
    """A 'video file' that decodes instantly: 50 frames at 50 fps = 1 s of video."""

    def __init__(self):
        self.i = 0

    def read(self):
        if self.i >= 50:
            return False, None
        self.i += 1
        return True, np.zeros((2, 2, 3), np.uint8)


def test_realtime_file_plays_at_its_own_speed():
    reader = LatestFrameReader(FastFile(), pace_fps=50)
    t0 = time.perf_counter()
    reader.start()
    seen = []
    while (item := reader.latest(timeout=1.0)) is not None:
        seen.append(item[0])
        time.sleep(0.1)  # slow analysis: ~10 fps
    elapsed = time.perf_counter() - t0
    assert 0.9 < elapsed < 1.6            # took about as long as the video, not instant
    assert seen == sorted(set(seen)) and len(seen) <= 15   # frames we couldn't keep up with were skipped


def test_downloaded_face_model_matches_checksum():
    path = Path("models/face_detection_yunet_2023mar.onnx")
    if not path.exists():
        pytest.skip("face model not downloaded yet")
    assert ensure_face_model(path) == path
    assert cv2.FaceDetectorYN.create(str(path), "", (320, 320)) is not None
