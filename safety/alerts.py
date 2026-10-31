"""Proximity alert levels: SAFE -> WARN -> ALERT -> ALARM, per person and per machine.

Each person near each machine has its own state. The machine shows the highest level of
anyone near it, and announcements name only the machine ("person at Press 3"), never a person.
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

from safety.config import AlertSettings, Machine

WALKING_SPEED_MPS = 1.6  # approach speed of a walking person used in ISO 13855


def late_alarm(machine: Machine, s: AlertSettings) -> tuple[float, float] | None:
    """Is the 'very close' line far enough out for the ALARM to fire before a fast walker gets there?

    Returns (distance walked before ALARM, suggested danger_m) when it isn't, else None.
    """
    response_s = 1.0 / s.expected_fps + s.critical_after_s
    walked = WALKING_SPEED_MPS * response_s
    if machine.danger_m * s.critical_fraction >= walked:
        return None
    return walked, walked / s.critical_fraction


class Level(IntEnum):
    SAFE = 0
    WARN = 1
    ALERT = 2
    ALARM = 3


@dataclass(frozen=True)
class Observation:
    track_id: int        # internal only, never logged or announced
    machine_id: str
    distance_m: float
    in_operator_spot: bool = False


@dataclass(frozen=True)
class MachineChange:
    machine: Machine
    old: Level
    new: Level
    people: int          # how many people are at ALERT or above (WARN or above when the machine is at WARN)
    message: str


@dataclass
class Incident:
    """One continuous period where a machine was above SAFE. Logged as text when it ends."""
    machine: Machine
    started: float
    started_wall: datetime
    peak: Level = Level.WARN
    min_distance_m: float = float("inf")
    max_people: int = 0
    reasons: set[str] = field(default_factory=set)
    acked_at: float | None = None

    def record(self, ended: float, ended_wall: datetime, end_reason: str) -> dict:
        return {
            "machine_id": self.machine.id,
            "machine": self.machine.name,
            "started": self.started_wall.isoformat(timespec="seconds"),
            "ended": ended_wall.isoformat(timespec="seconds"),
            "duration_s": round(ended - self.started, 1),
            "peak_level": self.peak.name,
            "min_distance_m": round(self.min_distance_m, 2),
            "max_people": self.max_people,
            "alarm_reasons": sorted(self.reasons),
            "acknowledged_after_s": None if self.acked_at is None else round(self.acked_at - self.started, 1),
            "end_reason": end_reason,
        }


@dataclass
class _Pair:
    """One person near one machine."""
    level: Level = Level.SAFE
    warn_since: float | None = None
    danger_since: float | None = None
    critical_since: float | None = None
    clear_since: float | None = None
    alert_since: float | None = None
    last_seen: float = 0.0
    distance_m: float = float("inf")
    history: deque = field(default_factory=lambda: deque(maxlen=30))  # (time, distance)
    reasons: set[str] = field(default_factory=set)                    # why it reached ALARM


def _since(prev: float | None, condition: bool, now: float) -> float | None:
    if not condition:
        return None
    return now if prev is None else prev


def message_for(machine: Machine, level: Level, people: int) -> str:
    who = "person" if people <= 1 else f"{people} people"
    if level == Level.ALARM:
        return f"Warning: clearance breach at {machine.name}. Step back now."
    if level == Level.ALERT:
        return f"Attention: {who} at {machine.name}, please step back {machine.warn_m:g} meters."
    if level == Level.WARN:
        return f"Caution: {who} close to {machine.name}."
    return f"{machine.name} clear."


class AlertEngine:
    def __init__(self, machines: list[Machine], settings: AlertSettings,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], datetime] = datetime.now):
        self.machines = {m.id: m for m in machines}
        self.s = settings
        self.clock = clock
        self.wall = wall
        self._pairs: dict[tuple[int, str], _Pair] = {}
        self._levels = {m.id: Level.SAFE for m in machines}
        self._people = {m.id: 0 for m in machines}
        self._incidents: dict[str, Incident] = {}
        self._spot_occupant: dict[str, int] = {}

    # --- operator actions -------------------------------------------------------------------------
    def acknowledge(self, machine_id: str) -> bool:
        """Operator announced the alert. Starts the 'ignored announcement' timer."""
        inc = self._incidents.get(machine_id)
        if inc is None or self._levels[machine_id] < Level.ALERT or inc.acked_at is not None:
            return False
        inc.acked_at = self.clock()
        return True

    def level(self, machine_id: str) -> Level:
        return self._levels[machine_id]

    # --- main update ------------------------------------------------------------------------------
    def update(self, observations: list[Observation],
               running: Callable[[Machine], bool] = lambda m: True) -> tuple[list[MachineChange], list[dict]]:
        now = self.clock()
        stopped = {mid for mid, m in self.machines.items() if not running(m)}
        self._update_spot_occupants(observations)

        for ob in observations:
            m = self.machines.get(ob.machine_id)
            if m is None or ob.machine_id in stopped:
                continue
            key = (ob.track_id, ob.machine_id)
            pair = self._pairs.get(key)
            if pair is None:
                if ob.distance_m > m.warn_m + self.s.clear_margin_m:
                    continue  # far away: no state needed
                pair = self._pairs[key] = _Pair()
            exempt = (ob.in_operator_spot and ob.distance_m > 0
                      and self._spot_occupant.get(ob.machine_id) == ob.track_id)
            self._step(pair, m, now, ob.distance_m, exempt)

        # Forget pairs that are safe, belong to a stopped machine, or were lost from view too long
        lost_from = set()
        for key, pair in list(self._pairs.items()):
            _, mid = key
            lost = now - pair.last_seen > self.s.lost_grace_s
            if lost and pair.level > Level.SAFE:
                lost_from.add(mid)
            if mid in stopped or lost or (pair.level == Level.SAFE and pair.warn_since is None):
                del self._pairs[key]

        return self._aggregate(now, stopped, lost_from)

    def close_all(self, reason: str = "shutdown") -> list[dict]:
        now, wall = self.clock(), self.wall()
        records = [inc.record(now, wall, reason) for inc in self._incidents.values()]
        self._incidents.clear()
        return records

    # --- internals --------------------------------------------------------------------------------
    def _update_spot_occupants(self, observations: list[Observation]) -> None:
        """Only one person per operator spot is exempt: whoever was there first."""
        in_spot: dict[str, set[int]] = {}
        for ob in observations:
            if ob.in_operator_spot:
                in_spot.setdefault(ob.machine_id, set()).add(ob.track_id)
        for mid in list(self._spot_occupant):
            if self._spot_occupant[mid] not in in_spot.get(mid, set()):
                del self._spot_occupant[mid]
        for mid, tracks in in_spot.items():
            self._spot_occupant.setdefault(mid, min(tracks))

    def _approach_speed(self, pair: _Pair, now: float) -> float:
        """Meters per second toward the machine over the last ~1 s (positive = getting closer)."""
        old = next(((t, d) for t, d in pair.history if now - t <= 1.0), None)
        if old is None or now - old[0] < 0.5:
            return 0.0
        return (old[1] - pair.distance_m) / (now - old[0])

    def _step(self, pair: _Pair, m: Machine, now: float, d: float, exempt: bool) -> None:
        s = self.s
        pair.last_seen = now
        pair.distance_m = d
        pair.history.append((now, d))

        warn_in = not exempt and d <= m.warn_m
        danger_in = not exempt and d <= m.danger_m
        # The operator spot never exempts stepping into the machine's own area
        critical_in = d == 0 if exempt else d <= m.danger_m * s.critical_fraction

        pair.warn_since = _since(pair.warn_since, warn_in, now)
        pair.danger_since = _since(pair.danger_since, danger_in, now)
        pair.critical_since = _since(pair.critical_since, critical_in, now)

        want, reasons = Level.SAFE, set()
        if warn_in and now - pair.warn_since >= s.warn_after_s:
            want = Level.WARN
        if danger_in and now - pair.danger_since >= s.alert_after_s:
            want = Level.ALERT
        if critical_in and now - pair.critical_since >= s.critical_after_s:
            want, reasons = Level.ALARM, {"very close"}

        if pair.level >= Level.ALERT and danger_in:
            inc = self._incidents.get(m.id)
            if self._approach_speed(pair, now) >= s.approach_speed_mps:
                reasons.add("moving toward machine")
            if inc is not None and inc.acked_at is None and now - pair.alert_since >= s.ack_timeout_s:
                reasons.add("not acknowledged")
            if inc is not None and inc.acked_at is not None and now - inc.acked_at >= s.ignored_after_s:
                reasons.add("ignored announcement")
            if reasons:
                want = Level.ALARM

        pair.reasons |= reasons

        if want > pair.level:
            pair.level = want
            pair.clear_since = None
            if want >= Level.ALERT and pair.alert_since is None:
                pair.alert_since = now
        elif want < pair.level:
            # Anti-flicker: only drop after being clearly past the line for a while
            limit = (m.warn_m if pair.level == Level.WARN else m.danger_m) + s.clear_margin_m
            if exempt or d > limit:
                pair.clear_since = _since(pair.clear_since, True, now)
                if now - pair.clear_since >= s.clear_after_s:
                    still_near = not exempt and pair.level > Level.WARN and d <= m.warn_m + s.clear_margin_m
                    pair.level = max(want, Level.WARN if still_near else Level.SAFE)
                    pair.clear_since = None
                    if pair.level < Level.ALERT:
                        pair.alert_since = None
            else:
                pair.clear_since = None

    def _aggregate(self, now: float, stopped: set[str],
                   lost_from: set[str]) -> tuple[list[MachineChange], list[dict]]:
        changes, closed = [], []
        for mid, m in self.machines.items():
            pairs = [p for (_, pm), p in self._pairs.items() if pm == mid and p.level > Level.SAFE]
            level = max((p.level for p in pairs), default=Level.SAFE)
            count_from = Level.ALERT if level >= Level.ALERT else Level.WARN
            people = sum(1 for p in pairs if p.level >= count_from)

            inc = self._incidents.get(mid)
            if level > Level.SAFE:
                if inc is None:
                    inc = self._incidents[mid] = Incident(m, now, self.wall())
                inc.peak = max(inc.peak, level)
                inc.min_distance_m = min([inc.min_distance_m, *(p.distance_m for p in pairs)])
                inc.max_people = max(inc.max_people, people)
                for p in pairs:
                    inc.reasons |= p.reasons
            elif inc is not None:
                reason = ("machine stopped" if mid in stopped
                          else "person lost from view" if mid in lost_from else "cleared")
                closed.append(inc.record(now, self.wall(), reason))
                del self._incidents[mid]

            old = self._levels[mid]
            if level != old or (level >= Level.ALERT and people != self._people[mid]):
                changes.append(MachineChange(m, old, level, people, message_for(m, level, people)))
            self._levels[mid] = level
            self._people[mid] = people
        return changes, closed
