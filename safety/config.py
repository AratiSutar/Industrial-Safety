"""Load and validate settings.yaml, machines.yaml and environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODES = ("GATE", "FACTORY")
RUNNING_SOURCES = ("always", "motion", "plc")
BACKENDS = ("openvino", "pytorch")
JEV_ITEMS = ("hard_hat", "vest", "boots")  # safety glasses are checked locally, never sent out


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
class GateSettings:
    box: tuple[float, float, float, float]  # inspection area as fractions of the image
    still_seconds: float
    still_tolerance: float                  # max foot movement, fraction of person height


@dataclass
class JevSettings:
    model: str
    timeout_s: float
    max_workers: int
    min_confidence: float
    required_items: tuple[str, ...]
    recheck_s: float                        # FACTORY mode re-check interval, 0 = off


@dataclass
class Settings:
    mode: str
    video_source: str
    camera: str
    process_every_n_frames: int
    pose_weights: str
    imgsz: int
    device: str
    backend: str
    tracker: str
    min_keypoint_conf: float
    calibration_dir: Path
    store_frames: bool
    blur_faces: bool
    snapshot_retention_days: int
    gate: GateSettings
    jev: JevSettings
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


def _parse_gate(raw: dict) -> GateSettings:
    box = tuple(float(v) for v in raw.get("box", (0.25, 0.05, 0.75, 0.98)))
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ConfigError("gate.box must be [x1, y1, x2, y2] fractions with x1 < x2 and y1 < y2")
    return GateSettings(
        box=box,
        still_seconds=float(raw.get("still_seconds", 0.5)),
        still_tolerance=float(raw.get("still_tolerance", 0.05)),
    )


def _parse_jev(raw: dict) -> JevSettings:
    model = str(raw.get("model", "jev-1.14.0"))
    if model.endswith("latest"):
        raise ConfigError("Pin an exact Jev version; do not use jev-latest")

    items = tuple(raw.get("required_items", JEV_ITEMS))
    unknown = set(items) - set(JEV_ITEMS)
    if unknown:
        raise ConfigError(f"jev.required_items: unknown {sorted(unknown)}; allowed {JEV_ITEMS}")

    min_conf = float(raw.get("min_confidence", 0.7))
    if not 0 <= min_conf <= 1:
        raise ConfigError("jev.min_confidence must be between 0 and 1")

    return JevSettings(
        model=model,
        timeout_s=float(raw.get("timeout_s", 2.0)),
        max_workers=int(raw.get("max_workers", 3)),
        min_confidence=min_conf,
        required_items=items,
        recheck_s=float(raw.get("recheck_s", 60)),
    )


def load_settings(config_dir: Path = Path("config"), env_file: Path = Path(".env")) -> Settings:
    load_dotenv(env_file)
    raw = yaml.safe_load((config_dir / "settings.yaml").read_text(encoding="utf-8")) or {}

    mode = str(raw.get("mode", "FACTORY")).upper()
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")

    video = raw.get("video", {})
    model = raw.get("model", {})
    privacy = raw.get("privacy", {})

    every_n = int(video.get("process_every_n_frames", 1))
    if every_n < 1:
        raise ConfigError("video.process_every_n_frames must be >= 1")

    backend = str(model.get("backend", "openvino")).lower()
    if backend not in BACKENDS:
        raise ConfigError(f"model.backend must be one of {BACKENDS}, got {backend!r}")

    return Settings(
        mode=mode,
        video_source=str(video.get("source", "0")),
        camera=str(video.get("camera", "cam_1")),
        process_every_n_frames=every_n,
        pose_weights=str(model.get("pose_weights", "models/yolov8n-pose.pt")),
        imgsz=int(model.get("imgsz", 416)),
        device=str(model.get("device", "cpu")),
        backend=backend,
        tracker=str(model.get("tracker", "bytetrack.yaml")),
        min_keypoint_conf=float(model.get("min_keypoint_conf", 0.5)),
        calibration_dir=Path(raw.get("calibration_dir", "calibration")),
        store_frames=bool(privacy.get("store_frames", False)),
        blur_faces=bool(privacy.get("blur_faces", True)),
        snapshot_retention_days=int(privacy.get("snapshot_retention_days", 7)),
        gate=_parse_gate(raw.get("gate", {})),
        jev=_parse_jev(raw.get("jev", {})),
        typesafe_api_key=os.getenv("TYPESAFE_API_KEY") or None,
        machines=load_machines(config_dir / "machines.yaml"),
    )
