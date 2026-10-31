"""Decide when a person at the gate is ready to be inspected:
whole body inside the inspection box and standing still for a moment."""

import time
from collections.abc import Callable

import numpy as np

from safety.config import GateSettings
from safety.detector import Person, is_full_body


class GateWatcher:
    def __init__(self, settings: GateSettings, min_keypoint_conf: float,
                 clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.min_conf = min_keypoint_conf
        self.clock = clock
        self._still_since: dict[int, tuple[tuple[float, float], float]] = {}  # track -> (anchor foot, since)

    def _inside_box(self, person: Person, frame_shape) -> bool:
        h, w = frame_shape[:2]
        bx1, by1, bx2, by2 = self.settings.box
        x1, y1, x2, y2 = person.box
        return x1 >= bx1 * w and y1 >= by1 * h and x2 <= bx2 * w and y2 <= by2 * h

    def ready(self, people: list[Person], frame_shape) -> list[Person]:
        now = self.clock()
        seen, ready = set(), []
        for p in people:
            if p.track_id is None:
                continue
            seen.add(p.track_id)
            if not (self._inside_box(p, frame_shape) and is_full_body(p, self.min_conf)):
                self._still_since.pop(p.track_id, None)
                continue

            height = p.box[3] - p.box[1]
            anchor, since = self._still_since.get(p.track_id, (p.foot, now))
            if np.hypot(p.foot[0] - anchor[0], p.foot[1] - anchor[1]) > self.settings.still_tolerance * height:
                anchor, since = p.foot, now  # moved: restart the timer
            self._still_since[p.track_id] = (anchor, since)

            if now - since >= self.settings.still_seconds:
                ready.append(p)

        for tid in set(self._still_since) - seen:  # forget people who left
            del self._still_since[tid]
        return ready
