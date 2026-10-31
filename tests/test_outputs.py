import pytest

from safety.alerts import Level, MachineChange
from safety.config import ConfigError, parse_machine
from safety.outputs import OutputRouter

BASE = {"id": "press", "name": "Press 3", "camera": "cam1",
        "polygon": [[0, 0], [2, 0], [2, 2]], "warn_m": 1.5, "danger_m": 0.8}


def change(machine, old, new):
    return MachineChange(machine, old, new, 1, "msg")


def test_levels_switch_outputs_on_and_off():
    m = parse_machine(BASE)
    router = OutputRouter()
    router.apply(change(m, Level.SAFE, Level.WARN))
    assert router.adapters["light_amber"].active == {"press": "msg"}

    line = router.apply(change(m, Level.WARN, Level.ALERT))
    assert router.adapters["light_amber"].active == {}                   # amber off
    assert "press" in router.adapters["operator_screen"].active
    assert line.startswith("[ALERT] Press 3 (up from WARN)")

    router.apply(change(m, Level.ALERT, Level.SAFE))
    assert all(not a.active for a in router.adapters.values())


def test_noisy_machine_defaults_to_headset_radio_not_voice():
    quiet, noisy = parse_machine(BASE), parse_machine({**BASE, "noisy": True})
    assert "voice" in quiet.outputs["ALARM"]
    assert "voice" not in noisy.outputs["ALARM"] and "headset_radio" in noisy.outputs["ALARM"]


def test_every_level_needs_something_the_worker_can_see():
    with pytest.raises(ConfigError, match="visual"):
        parse_machine({**BASE, "outputs": {"ALARM": ["voice", "operator_screen"]}})


def test_unknown_output_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        parse_machine({**BASE, "outputs": {"WARN": ["light_amber", "fireworks"]}})


def test_example_machines_file_is_valid():
    from pathlib import Path

    from safety.config import load_machines
    machines = load_machines(Path(__file__).resolve().parent.parent / "config" / "machines.example.yaml")
    press = next(m for m in machines if m.id == "press_03")
    assert press.noisy and press.operator_spot is not None
