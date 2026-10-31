import cv2
import numpy as np
import pytest

from safety import homography

# A camera looking at the floor at an angle: floor meters -> pixels
TRUE_FLOOR_TO_IMAGE = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [4, 0], [4, 6], [0, 6]]),
    np.float32([[200, 700], [1000, 700], [800, 250], [400, 250]]),
)


def to_image(floor_pts):
    pts = np.asarray(floor_pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, TRUE_FLOOR_TO_IMAGE).reshape(-1, 2)


FLOOR = [(0, 0), (4, 0), (4, 6), (0, 6), (2, 3)]


def test_pixels_map_back_to_meters():
    cal = homography.compute("cam", to_image(FLOOR), FLOOR)
    assert cal.error_m < 1e-6
    check = [(1.0, 1.0), (3.5, 5.0), (2.2, 0.4)]
    assert np.allclose(cal.to_floor(to_image(check)), check, atol=1e-6)
    assert np.allclose(cal.to_image(check), to_image(check), atol=1e-4)


def test_bad_measurement_shows_up_as_error():
    floor = list(FLOOR)
    floor[4] = (2.5, 3.0)  # someone measured the 5th point 50 cm off
    cal = homography.compute("cam", to_image(FLOOR), floor)
    assert cal.error_m > homography.MAX_GOOD_ERROR_M / 2


def test_needs_four_points():
    with pytest.raises(ValueError):
        homography.compute("cam", to_image(FLOOR[:3]), FLOOR[:3])


def test_points_on_one_line_are_rejected():
    line = [(0, 0), (1, 0), (2, 0), (3, 0)]
    with pytest.raises(ValueError):
        homography.compute("cam", to_image(line), line)


def test_save_and_load(tmp_path):
    cal = homography.compute("cam_floor_1", to_image(FLOOR), FLOOR)
    homography.save(cal, tmp_path)
    loaded = homography.load("cam_floor_1", tmp_path)
    assert loaded is not None and np.allclose(loaded.H, cal.H)
    assert homography.load("other_cam", tmp_path) is None
