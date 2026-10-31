from pathlib import Path

import pytest

from safety.config import ConfigError, load_machines, load_settings

ROOT = Path(__file__).resolve().parent.parent


def test_default_settings_load():
    s = load_settings(ROOT / "config", env_file=ROOT / "missing.env")
    assert s.mode in ("GATE", "FACTORY")
    assert s.process_every_n_frames >= 1
    assert not s.jev_model.endswith("latest")


def test_example_machines_are_valid():
    machines = load_machines(ROOT / "config" / "machines.example.yaml")
    assert len(machines) == 2
    assert all(0 < m.danger_m < m.warn_m for m in machines)


def test_danger_must_be_smaller_than_warn(tmp_path):
    bad = tmp_path / "machines.yaml"
    bad.write_text(
        "machines:\n"
        "  - {id: m1, camera: c1, polygon: [[0,0],[1,0],[1,1]], warn_m: 0.5, danger_m: 1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_machines(bad)


def test_jev_latest_is_rejected(tmp_path):
    (tmp_path / "settings.yaml").write_text("jev:\n  model: jev-latest\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(tmp_path, env_file=tmp_path / "missing.env")
