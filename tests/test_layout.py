import numpy as np
import pytest

from safety import homography, layout
from safety.config import ConfigError, load_machines

# Camera where 1 pixel = 1 cm on the floor (top-down), so results are easy to check
CAL = homography.compute("cam1", [(0, 0), (400, 0), (400, 300), (0, 300)], [(0, 0), (4, 0), (4, 3), (0, 3)])


def test_clicked_pixels_become_floor_meters():
    assert layout.floor_polygon(CAL, [(100, 100), (200, 100), (200, 150), (100, 150)]) == \
        [[1.0, 1.0], [2.0, 1.0], [2.0, 1.5], [1.0, 1.5]]


def test_crossing_or_too_few_clicks_rejected():
    with pytest.raises(ConfigError):
        layout.floor_polygon(CAL, [(100, 100), (200, 100)])
    with pytest.raises(ConfigError, match="order"):
        layout.floor_polygon(CAL, [(100, 100), (200, 150), (200, 100), (100, 150)])  # bow-tie


def test_entry_is_validated():
    poly = [[1, 1], [2, 1], [2, 2]]
    with pytest.raises(ConfigError):
        layout.machine_entry("m1", "M1", "cam1", poly, warn_m=0.5, danger_m=1.0)  # danger > warn


def test_ring_is_at_the_right_distance():
    pts = np.array(layout.ring([[0, 0], [2, 0], [2, 2], [0, 2]], 1.0))
    assert pts[:, 0].min() == pytest.approx(-1.0) and pts[:, 1].max() == pytest.approx(3.0)


def test_save_keeps_other_cameras_and_makes_backup(tmp_path):
    path = tmp_path / "machines.yaml"
    other = layout.machine_entry("drill", "Drill", "cam2", [[0, 0], [1, 0], [1, 1]], 1.0, 0.5)
    layout.save(path, "cam2", [other])

    press = layout.machine_entry("press", "Press 3", "cam1", [[1, 1], [2, 1], [2, 2]], 1.5, 0.8,
                                 noisy=True, operator_spot=[[1, 0.3], [1.5, 0.3], [1.5, 0.8]])
    backup = layout.save(path, "cam1", [press])

    assert backup is not None and backup.exists()
    machines = {m.id: m for m in load_machines(path)}
    assert set(machines) == {"drill", "press"}
    assert machines["press"].noisy and machines["press"].operator_spot is not None


def test_duplicate_id_across_cameras_rejected(tmp_path):
    path = tmp_path / "machines.yaml"
    layout.save(path, "cam2", [layout.machine_entry("press", "P", "cam2", [[0, 0], [1, 0], [1, 1]], 1.0, 0.5)])
    with pytest.raises(ConfigError, match="twice"):
        layout.save(path, "cam1", [layout.machine_entry("press", "P", "cam1", [[0, 0], [1, 0], [1, 1]], 1.0, 0.5)])


def test_floor_points_behind_the_camera_are_nan():
    tilted = homography.compute("cam", [(0, 700), (800, 700), (600, 300), (200, 300)],
                                [(0, 0), (4, 0), (4, 6), (0, 6)])
    assert np.isnan(tilted.to_image([(2, -50)])).all()               # behind the camera: no image position
    far_ahead = tilted.to_image([(2, 1000)])[0]
    assert -100 < far_ahead[1] < 300                                  # converges toward the vanishing point
    assert tilted.to_image([(2, 3)])[0] == pytest.approx([400, 433.3], abs=0.1)
