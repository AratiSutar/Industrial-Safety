"""Load and validate settings.yaml, machines.yaml and environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODES = ("GATE", "FACTORY")
RUNNING_SOURCES = ("always", "motion", "plc")


class ConfigError(ValueError):
    pass


@dataclass
class Machine:
    id: str
    name: str
    camera: str
    polygon: list[tuple[float, float]]
    warn_m: float
    danger_m: float
    running_source: str = "always"


@dataclass
class Settings:
    mode: str
    video_source: str
    process_every_n_frames: int
    pose_weights: str
    imgsz: int
    device: str
    store_frames: bool
    blur_faces: bool
    snapshot_retention_days: int
    jev_model: str
    jev_timeout_s: float
    typesafe_api_key: str | None = None
    machines: list[Machine] = field(default_factory=list)


def _parse_machine(raw: dict) -> Machine:
    try:
        m = Machine(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            camera=str(raw["camera"]),
            polygon=[(float(x), float(y)) for x, y in raw["polygon"]],
            warn_m=float(raw["warn_m"]),
            danger_m=float(raw["danger_m"]),
            running_source=str(raw.get("running_source", "always")),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"Invalid machine entry {raw!r}: {e}") from e

    if len(m.polygon) < 3:
        raise ConfigError(f"Machine {m.id}: polygon needs at least 3 points")
    if not 0 < m.danger_m < m.warn_m:
        raise ConfigError(f"Machine {m.id}: need 0 < danger_m < warn_m")
    if m.running_source not in RUNNING_SOURCES:
        raise ConfigError(f"Machine {m.id}: running_source must be one of {RUNNING_SOURCES}")
    return m


def load_machines(path: Path) -> list[Machine]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    machines = [_parse_machine(m) for m in raw.get("machines", [])]

    ids = [m.id for m in machines]
    if len(ids) != len(set(ids)):
        raise ConfigError("Machine ids must be unique")
    return machines


def load_settings(config_dir: Path = Path("config"), env_file: Path = Path(".env")) -> Settings:
    load_dotenv(env_file)
    raw = yaml.safe_load((config_dir / "settings.yaml").read_text(encoding="utf-8")) or {}

    mode = str(raw.get("mode", "FACTORY")).upper()
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")

    video = raw.get("video", {})
    model = raw.get("model", {})
    privacy = raw.get("privacy", {})
    jev = raw.get("jev", {})

    every_n = int(video.get("process_every_n_frames", 1))
    if every_n < 1:
        raise ConfigError("video.process_every_n_frames must be >= 1")

    jev_model = str(jev.get("model", "jev-1.14.0"))
    if jev_model.endswith("latest"):
        raise ConfigError("Pin an exact Jev version; do not use jev-latest")

    return Settings(
        mode=mode,
        video_source=str(video.get("source", "0")),
        process_every_n_frames=every_n,
        pose_weights=str(model.get("pose_weights", "models/yolov8n-pose.pt")),
        imgsz=int(model.get("imgsz", 416)),
        device=str(model.get("device", "cpu")),
        store_frames=bool(privacy.get("store_frames", False)),
        blur_faces=bool(privacy.get("blur_faces", True)),
        snapshot_retention_days=int(privacy.get("snapshot_retention_days", 7)),
        jev_model=jev_model,
        jev_timeout_s=float(jev.get("timeout_s", 2.0)),
        typesafe_api_key=os.getenv("TYPESAFE_API_KEY") or None,
        machines=load_machines(config_dir / "machines.yaml"),
    )
