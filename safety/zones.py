"""Distance from a person's floor position to each machine's floor area."""

from dataclasses import dataclass

from shapely.geometry import Point, Polygon

from safety.config import Machine


@dataclass(frozen=True)
class Nearest:
    machine: Machine
    distance_m: float  # 0 when standing inside the machine's area


class MachineZones:
    def __init__(self, machines: list[Machine], camera: str):
        # Only machines this camera can see
        self.machines = [m for m in machines if m.camera == camera]
        self._polygons = [Polygon(m.polygon) for m in self.machines]
        for m, poly in zip(self.machines, self._polygons, strict=True):
            if not poly.is_valid:
                raise ValueError(f"Machine {m.id}: polygon edges cross each other")

    def nearest(self, floor_xy: tuple[float, float]) -> Nearest | None:
        if not self.machines:
            return None
        p = Point(floor_xy)
        distances = [poly.distance(p) for poly in self._polygons]
        i = min(range(len(distances)), key=distances.__getitem__)
        return Nearest(self.machines[i], float(distances[i]))
