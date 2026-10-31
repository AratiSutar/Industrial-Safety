import pytest

from safety.config import Machine
from safety.zones import MachineZones


def machine(id_, polygon, camera="cam1"):
    return Machine(id_, id_.title(), camera, polygon, warn_m=1.5, danger_m=0.8)


PRESS = machine("press", [(2, 2), (4, 2), (4, 4), (2, 4)])
LATHE = machine("lathe", [(8, 2), (9, 2), (9, 3), (8, 3)])
OTHER_CAMERA = machine("drill", [(0, 0), (1, 0), (1, 1)], camera="cam2")


def test_distance_to_nearest_edge():
    zones = MachineZones([PRESS, LATHE], "cam1")
    n = zones.nearest((5.0, 3.0))
    assert n.machine.id == "press" and n.distance_m == pytest.approx(1.0)


def test_inside_machine_area_is_zero():
    assert MachineZones([PRESS], "cam1").nearest((3, 3)).distance_m == 0


def test_picks_closer_machine():
    assert MachineZones([PRESS, LATHE], "cam1").nearest((7.5, 2.5)).machine.id == "lathe"


def test_only_machines_of_this_camera():
    zones = MachineZones([PRESS, OTHER_CAMERA], "cam1")
    assert [m.id for m in zones.machines] == ["press"]
    assert MachineZones([OTHER_CAMERA], "cam1").nearest((0, 0)) is None


def test_self_crossing_polygon_rejected():
    bowtie = machine("bad", [(0, 0), (2, 2), (2, 0), (0, 2)])
    with pytest.raises(ValueError):
        MachineZones([bowtie], "cam1")
