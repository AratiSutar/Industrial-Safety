from pathlib import Path

import pytest

from safety.config import ConfigError, load_machines, load_settings

ROOT = Path(__file__).resolve().parent.parent


def test_default_settings_load():
    s = load_settings(ROOT / "config", env_file=ROOT / "missing.env")
    assert s.mode in ("GATE", "FACTORY")
    assert s.process_every_n_frames >= 1
    assert not s.jev.model.endswith("latest")
    assert s.jev.required_items == ("hard_hat", "vest", "boots")
    assert s.gate.box[0] < s.gate.box[2]


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


def test_inspection_engine_defaults_to_jev_and_local_ml_settings_load():
    s = load_settings(ROOT / "config", env_file=ROOT / "missing.env")
    assert s.engine == "jev_vlm"
    assert set(s.local_ml.classes) == {"hard_hat", "vest", "boots"}
    assert "no-hardhat" in s.local_ml.classes["hard_hat"]["missing"]


@pytest.mark.parametrize("yaml_text", [
    "inspection:\n  engine: gpt\n",
    "local_ml:\n  classes:\n    gloves: {present: [gloves]}\n",
    "local_ml:\n  min_confidence: 2\n",
])
def test_invalid_engine_settings_are_rejected(tmp_path, yaml_text):
    (tmp_path / "settings.yaml").write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(tmp_path, env_file=tmp_path / "missing.env")


def test_jev_latest_is_rejected(tmp_path):
    (tmp_path / "settings.yaml").write_text("jev:\n  model: jev-latest\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(tmp_path, env_file=tmp_path / "missing.env")
