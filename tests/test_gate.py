import numpy as np

from safety.config import GateSettings
from safety.detector import L_ANKLE, NOSE, R_ANKLE, Person
from safety.gate import GateWatcher

FRAME = (1000, 1000, 3)
SETTINGS = GateSettings(box=(0.2, 0.0, 0.8, 1.0), still_seconds=0.5, still_tolerance=0.05)


def person(x=500.0, track_id=1, full_body=True):
    kp, kc = np.zeros((17, 2)), np.full(17, 0.9 if full_body else 0.1)
    kp[NOSE], kp[L_ANKLE], kp[R_ANKLE] = (x, 150), (x - 20, 890), (x + 20, 890)
    return Person(track_id, (x - 100, 100.0, x + 100, 900.0), 0.9, kp, kc, (x, 890), full_body)


def watcher():
    now = [0.0]
    return GateWatcher(SETTINGS, 0.5, clock=lambda: now[0]), now


def test_ready_after_standing_still():
    w, now = watcher()
    assert w.ready([person()], FRAME) == []
    now[0] = 0.3
    assert w.ready([person(x=505)], FRAME) == []      # small sway is fine
    now[0] = 0.6
    assert len(w.ready([person(x=503)], FRAME)) == 1


def test_walking_restarts_timer():
    w, now = watcher()
    w.ready([person(x=400)], FRAME)
    now[0] = 0.6
    assert w.ready([person(x=500)], FRAME) == []      # moved 100 px: not still
    now[0] = 1.2
    assert len(w.ready([person(x=500)], FRAME)) == 1


def test_outside_box_or_partial_body_is_ignored():
    w, now = watcher()
    w.ready([person(x=150), person(track_id=2, full_body=False)], FRAME)
    now[0] = 1.0
    assert w.ready([person(x=150), person(track_id=2, full_body=False)], FRAME) == []
