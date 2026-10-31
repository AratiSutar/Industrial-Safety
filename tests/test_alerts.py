from datetime import datetime

import pytest

from safety.alerts import AlertEngine, Level, Observation
from safety.config import AlertSettings, Machine

# warn 1.5 m, danger 0.8 m, "very close" = 0.4 m
PRESS = Machine("press", "Press 3", "cam1", [(0, 0), (2, 0), (2, 2), (0, 2)], warn_m=1.5, danger_m=0.8)
FPS = 5


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(**settings):
    clock = Clock()
    engine = AlertEngine([PRESS], AlertSettings(**settings), clock=clock,
                         wall=lambda: datetime(2026, 10, 31, 12, 0, 0))
    return engine, clock


def run(engine, clock, distance, seconds, track=1, spot=False, running=True):
    """Feed one person at a fixed distance for `seconds` at 5 fps. Returns (changes, closed)."""
    changes, closed = [], []
    for _ in range(round(seconds * FPS)):
        clock.t += 1 / FPS
        c, cl = engine.update([Observation(track, "press", distance, spot)], lambda m: running)
        changes += c
        closed += cl
    return changes, closed


def test_walking_past_does_not_warn():
    engine, clock = make()
    run(engine, clock, 1.2, 0.4)    # 2 frames inside warn_m
    run(engine, clock, 3.0, 1.0)
    assert engine.level("press") == Level.SAFE


def test_lingering_gives_warn():
    engine, clock = make()
    changes, _ = run(engine, clock, 1.2, 1.0)
    assert engine.level("press") == Level.WARN
    assert changes[0].message == "Caution: person close to Press 3."


def test_danger_zone_gives_alert_with_location_announcement():
    engine, clock = make()
    changes, _ = run(engine, clock, 0.7, 1.4)
    assert engine.level("press") == Level.ALERT
    assert changes[-1].message == "Attention: person at Press 3, please step back 1.5 meters."


def test_very_close_skips_straight_to_alarm():
    engine, clock = make()
    changes, _ = run(engine, clock, 0.3, 0.6)
    assert engine.level("press") == Level.ALARM
    assert [c.new for c in changes] == [Level.ALARM]
    _, closed = run(engine, clock, 3.0, 1.4)
    assert closed[0]["alarm_reasons"] == ["very close"]


def test_unacknowledged_alert_escalates_10s_after_alert_started():
    engine, clock = make()
    run(engine, clock, 0.7, 1.2)                        # ALERT starts at ~1.2 s
    assert engine.level("press") == Level.ALERT
    run(engine, clock, 0.7, 9.6)                        # 9.6 s into the ALERT
    assert engine.level("press") == Level.ALERT
    run(engine, clock, 0.7, 0.6)                        # past 10 s
    assert engine.level("press") == Level.ALARM


def test_acknowledged_alert_waits_then_escalates_if_ignored():
    engine, clock = make()
    run(engine, clock, 0.7, 2.0)
    assert engine.acknowledge("press")
    run(engine, clock, 0.7, 15.0)
    assert engine.level("press") == Level.ALERT         # acknowledged: no 10 s escalation
    run(engine, clock, 0.7, 6.0)
    assert engine.level("press") == Level.ALARM         # still there 20 s after the announcement
    _, closed = run(engine, clock, 3.0, 2.4)
    assert closed[0]["alarm_reasons"] == ["ignored announcement"]
    assert closed[0]["acknowledged_after_s"] is not None


def test_moving_toward_machine_during_alert_escalates():
    engine, clock = make()
    run(engine, clock, 0.78, 1.4)
    assert engine.level("press") == Level.ALERT
    for d in (0.7, 0.6, 0.5, 0.45, 0.42):               # ~0.4 m/s toward the machine
        run(engine, clock, d, 0.2)
    assert engine.level("press") == Level.ALARM


def test_anti_flicker_on_the_line():
    engine, clock = make()
    run(engine, clock, 0.7, 1.4)
    run(engine, clock, 0.85, 3.0)                       # just outside danger_m, inside the margin
    assert engine.level("press") == Level.ALERT
    run(engine, clock, 1.1, 0.6)                        # past the margin, but not for 1 s yet
    assert engine.level("press") == Level.ALERT
    run(engine, clock, 1.1, 0.8)
    assert engine.level("press") == Level.WARN
    run(engine, clock, 2.0, 1.4)
    assert engine.level("press") == Level.SAFE


def test_stopped_machine_never_alerts_and_closes_incident():
    engine, clock = make()
    run(engine, clock, 0.3, 2.0, running=False)
    assert engine.level("press") == Level.SAFE
    run(engine, clock, 0.7, 1.4)
    _, closed = run(engine, clock, 0.7, 0.2, running=False)
    assert engine.level("press") == Level.SAFE
    assert closed[0]["end_reason"] == "machine stopped"


def test_operator_spot_exempts_one_person_only():
    engine, clock = make()
    for _ in range(10):                                   # 2 s
        clock.t += 1 / FPS
        engine.update([Observation(1, "press", 0.5, True), Observation(2, "press", 0.6, True)])
    assert engine.level("press") == Level.ALERT          # person 2 is the second one in the spot
    engine2, clock2 = make()
    run(engine2, clock2, 0.5, 2.0, spot=True)
    assert engine2.level("press") == Level.SAFE           # operator alone at their spot
    run(engine2, clock2, 0.0, 0.6, spot=True)
    assert engine2.level("press") == Level.ALARM          # but stepping into the machine is never exempt


def test_person_lost_from_view_keeps_alert_for_grace_period():
    engine, clock = make()
    run(engine, clock, 0.7, 1.4)
    for _ in range(10):                                   # 2 s with nobody detected
        clock.t += 1 / FPS
        engine.update([])
    assert engine.level("press") == Level.ALERT           # could be hidden behind the machine
    closed = []
    for _ in range(10):
        clock.t += 1 / FPS
        closed += engine.update([])[1]
    assert engine.level("press") == Level.SAFE
    assert closed[0]["end_reason"] == "person lost from view"


def test_two_people_announced_by_count_not_identity():
    engine, clock = make()
    changes = []
    for _ in range(8):
        clock.t += 1 / FPS
        changes += engine.update([Observation(1, "press", 0.7), Observation(2, "press", 0.6)])[0]
    assert changes[-1].message == "Attention: 2 people at Press 3, please step back 1.5 meters."


def test_incident_record_is_text_only():
    engine, clock = make()
    run(engine, clock, 0.7, 1.4)
    _, closed = run(engine, clock, 3.0, 2.4)
    record = closed[0]
    assert set(record) == {"machine_id", "machine", "started", "ended", "duration_s", "peak_level",
                           "min_distance_m", "max_people", "alarm_reasons", "acknowledged_after_s", "end_reason"}
    assert record["peak_level"] == "ALERT" and record["min_distance_m"] == pytest.approx(0.7)


def test_acknowledge_only_during_alert():
    engine, clock = make()
    assert not engine.acknowledge("press")
    run(engine, clock, 1.2, 1.0)                          # WARN only
    assert not engine.acknowledge("press")
