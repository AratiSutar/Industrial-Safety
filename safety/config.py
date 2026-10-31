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

ALERT_LEVELS = ("WARN", "ALERT", "ALARM")
OUTPUT_TYPES = (
    "light_amber", "light_red", "light_red_flashing", "floor_projection",  # seen by the worker
    "operator_screen", "supervisor_radio",                                # reach the operator / supervisor
    "voice", "headset_radio", "wristband",                                # heard / felt by the worker
)
# Every level needs at least one of these: workers with ear defenders may not hear anything
WORKER_VISUAL = frozenset({"light_amber", "light_red", "light_red_flashing", "floor_projection"})

# Layered defaults: WARN is a quiet nudge, ALERT brings in people, ALARM uses everything.
# Wristbands are opt-in only (they need Legal sign-off).
DEFAULT_OUTPUTS = {
    "WARN": ("light_amber",),
    "ALERT": ("light_red", "floor_projection", "operator_screen", "supervisor_radio"),
    "ALARM": ("light_red_flashing", "floor_projection", "operator_screen", "supervisor_radio", "voice"),
}
# In loud areas voice is unreliable; use the radio in the workers' ear defenders instead
NOISY_DEFAULT_OUTPUTS = {
    **DEFAULT_OUTPUTS,
    "ALARM": ("light_red_flashing", "floor_projection", "operator_screen", "supervisor_radio", "headset_radio"),
}


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
    noisy: bool = False                                    # ear defenders required here
    operator_spot: list[tuple[float, float]] | None = None  # where one operator may stand
    outputs: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_OUTPUTS))


@dataclass
class AlertSettings:
    warn_after_s: float = 0.6        # inside warn_m this long -> WARN
    alert_after_s: float = 1.0       # inside danger_m this long -> ALERT
    critical_fraction: float = 0.5   # inside danger_m * this -> ALARM straight away
    critical_after_s: float = 0.3
    approach_speed_mps: float = 0.3  # moving toward the machine faster than this during ALERT -> ALARM
    ack_timeout_s: float = 10.0      # ALERT not acknowledged by the operator -> ALARM
    ignored_after_s: float = 20.0    # still in danger this long after the announcement -> ALARM
    clear_margin_m: float = 0.2      # anti-flicker: must move this far past the line ...
    clear_after_s: float = 1.0       # ... for this long before the level drops
    lost_grace_s: float = 3.0        # keep alerting this long after a person disappears from view
    event_log: Path = Path("logs/events.jsonl")


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
    alerts: AlertSettings = field(default_factory=AlertSettings)
    typesafe_api_key: str | None = None
    machines: list[Machine] = field(default_factory=list)


def parse_machine(raw: dict) -> Machine:
    try:
        noisy = bool(raw.get("noisy", False))
        defaults = NOISY_DEFAULT_OUTPUTS if noisy else DEFAULT_OUTPUTS
        outputs = {level: tuple(raw.get("outputs", {}).get(level, defaults[level])) for level in ALERT_LEVELS}
        spot = raw.get("operator_spot")
        m = Machine(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            camera=str(raw["camera"]),
            polygon=[(float(x), float(y)) for x, y in raw["polygon"]],
            warn_m=float(raw["warn_m"]),
            danger_m=float(raw["danger_m"]),
            running_source=str(raw.get("running_source", "always")),
            noisy=noisy,
            operator_spot=[(float(x), float(y)) for x, y in spot] if spot else None,
            outputs=outputs,
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"Invalid machine entry {raw!r}: {e}") from e

    if len(m.polygon) < 3:
        raise ConfigError(f"Machine {m.id}: polygon needs at least 3 points")
    if m.operator_spot is not None and len(m.operator_spot) < 3:
        raise ConfigError(f"Machine {m.id}: operator_spot needs at least 3 points")
    if not 0 < m.danger_m < m.warn_m:
        raise ConfigError(f"Machine {m.id}: need 0 < danger_m < warn_m")
    if m.running_source not in RUNNING_SOURCES:
        raise ConfigError(f"Machine {m.id}: running_source must be one of {RUNNING_SOURCES}")
    for level, outs in m.outputs.items():
        unknown = set(outs) - set(OUTPUT_TYPES)
        if unknown:
            raise ConfigError(f"Machine {m.id}: unknown {level} outputs {sorted(unknown)}; allowed {OUTPUT_TYPES}")
        if not WORKER_VISUAL & set(outs):
            raise ConfigError(f"Machine {m.id}: {level} needs a visual output the worker can see "
                              f"({', '.join(sorted(WORKER_VISUAL))}); sound alone may not be heard")
    return m


def load_machines(path: Path) -> list[Machine]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    machines = [parse_machine(m) for m in raw.get("machines", [])]

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


def _parse_alerts(raw: dict) -> AlertSettings:
    defaults = AlertSettings()
    unknown = set(raw) - set(defaults.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"alerts: unknown settings {sorted(unknown)}")
    try:
        values = {k: (Path(v) if k == "event_log" else float(v)) for k, v in raw.items()}
        s = AlertSettings(**values)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"alerts: {e}") from e
    if not 0 < s.critical_fraction < 1:
        raise ConfigError("alerts.critical_fraction must be between 0 and 1")
    if s.ignored_after_s <= 0 or s.ack_timeout_s <= 0:
        raise ConfigError("alerts.ack_timeout_s and alerts.ignored_after_s must be positive")
    return s


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
        alerts=_parse_alerts(raw.get("alerts", {})),
        typesafe_api_key=os.getenv("TYPESAFE_API_KEY") or None,
        machines=load_machines(config_dir / "machines.yaml"),
    )
