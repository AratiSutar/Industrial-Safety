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
        self._spots = {m.id: Polygon(m.operator_spot) for m in self.machines if m.operator_spot}
        for m, poly in zip(self.machines, self._polygons, strict=True):
            if not poly.is_valid:
                raise ValueError(f"Machine {m.id}: polygon edges cross each other")
            if m.id in self._spots and not self._spots[m.id].is_valid:
                raise ValueError(f"Machine {m.id}: operator_spot edges cross each other")

    def distances(self, floor_xy: tuple[float, float]) -> list[Nearest]:
        """Distance to every machine on this camera."""
        p = Point(floor_xy)
        return [Nearest(m, float(poly.distance(p)))
                for m, poly in zip(self.machines, self._polygons, strict=True)]

    def nearest(self, floor_xy: tuple[float, float]) -> Nearest | None:
        return min(self.distances(floor_xy), key=lambda n: n.distance_m, default=None)

    def in_operator_spot(self, machine: Machine, floor_xy: tuple[float, float]) -> bool:
        spot = self._spots.get(machine.id)
        return spot is not None and spot.covers(Point(floor_xy))
