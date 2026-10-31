"""Draw tracking results on a (face-blurred) frame for the local preview."""

import cv2
import numpy as np

from safety.detector import Person

GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)


def draw_people(frame: np.ndarray, people: list[Person]) -> np.ndarray:
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        label = f"#{p.track_id}" if p.track_id is not None else "?"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        # Green dot = ankles visible, orange = fell back to bottom of the box
        color = GREEN if p.foot_from_ankles else ORANGE
        cv2.circle(frame, (int(p.foot[0]), int(p.foot[1])), 6, color, -1)
    return frame
