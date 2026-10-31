"""Build and save machine layouts (config/machines.yaml) from points clicked on a camera image."""

import shutil
from pathlib import Path

import yaml
from shapely.geometry import Polygon

from safety.config import ConfigError, parse_machine
from safety.homography import Calibration

HEADER = """\
# Factory layout: machine floor areas in meters (same coordinates as the camera calibration).
# Made with tools/draw_machines.py; safe to edit by hand. See machines.example.yaml for all options.
"""


def floor_polygon(cal: Calibration, image_points) -> list[list[float]]:
    """Clicked pixels -> floor polygon in meters (rounded to cm). Raises ConfigError if it isn't a usable shape."""
    if len(image_points) < 3:
        raise ConfigError("Click at least 3 corners")
    pts = [[round(float(x), 2), round(float(y), 2)] for x, y in cal.to_floor(image_points)]
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 0.01:
        raise ConfigError("Shape is invalid: click the corners in order around the edge, without crossing")
    return pts


def machine_entry(id_: str, name: str, camera: str, polygon, warn_m: float, danger_m: float,
                  noisy: bool = False, operator_spot=None) -> dict:
    entry = {"id": id_, "name": name, "camera": camera, "polygon": polygon,
             "warn_m": warn_m, "danger_m": danger_m, "running_source": "always", "noisy": noisy}
    if operator_spot:
        entry["operator_spot"] = operator_spot
    parse_machine(entry)  # validate now, not when the monitor starts
    return entry


def ring(polygon, distance_m: float) -> list[tuple[float, float]]:
    """Outline at a given distance around a machine (where WARN / ALERT start)."""
    return list(Polygon(polygon).buffer(distance_m, quad_segs=8).exterior.coords)


def load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("machines", [])


def merge(existing: list[dict], camera: str, entries: list[dict]) -> list[dict]:
    """Replace this camera's machines with `entries`; keep other cameras' machines untouched."""
    others = [m for m in existing if m.get("camera") != camera]
    merged = others + entries
    ids = [m["id"] for m in merged]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ConfigError(f"Machine ids used twice: {dupes}")
    return merged


def save(path: Path, camera: str, entries: list[dict]) -> Path | None:
    """Write the layout. Returns the backup path if an existing file was replaced."""
    machines = merge(load_raw(path), camera, entries)
    for m in machines:
        parse_machine(m)
    backup = None
    if path.exists():
        backup = path.with_suffix(".yaml.bak")
        shutil.copyfile(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump({"machines": machines}, sort_keys=False, default_flow_style=None),
                    encoding="utf-8")
    return backup
