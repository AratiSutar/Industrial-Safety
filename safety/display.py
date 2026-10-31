"""Draw tracking results on a (face-blurred) frame for the local preview."""

import cv2
import numpy as np

from safety.detector import Person
from safety.homography import Calibration
from safety.inspector import DENY, UNLOCK, Inspection
from safety.zones import Nearest

GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
GREY = (160, 160, 160)

PPE_LABELS = {"hard_hat": "HAT", "vest": "VEST", "boots": "BOOTS"}
ANSWER_MARKS = {"yes": "ok", "no": "NO", "unclear": "?", "error": "err"}


def ppe_label(ins: Inspection) -> tuple[str, tuple[int, int, int]]:
    """Jev's latest answers as text, e.g. 'HAT ok VEST NO BOOTS ?', coloured by the decision."""
    text = " ".join(f"{PPE_LABELS.get(item, item)} {ANSWER_MARKS.get(a.choice, a.choice)}"
                    for item, a in ins.answers.items())
    color = GREEN if ins.decision == UNLOCK else RED if ins.decision == DENY else ORANGE
    return text, color


def draw_people(frame: np.ndarray, people: list[Person],
                nearest: dict[int | None, Nearest] | None = None,
                ppe: dict[int | None, Inspection] | None = None) -> np.ndarray:
    nearest = nearest or {}
    ppe = ppe or {}
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        label = f"#{p.track_id}" if p.track_id is not None else "?"
        if (n := nearest.get(p.track_id)) is not None:
            label += f" {n.machine.name}: {n.distance_m:.1f} m"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        if (ins := ppe.get(p.track_id)) is not None:
            text, color = ppe_label(ins)
            cv2.putText(frame, text, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Green dot = ankles visible, orange = fell back to bottom of the box
        color = GREEN if p.foot_from_ankles else ORANGE
        cv2.circle(frame, (int(p.foot[0]), int(p.foot[1])), 6, color, -1)
    return frame


def draw_machines(frame: np.ndarray, cal: Calibration, machines) -> np.ndarray:
    """Outline each machine's floor area on the video."""
    for m in machines:
        pts = cal.to_image(m.polygon).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=RED, thickness=2)
        cv2.putText(frame, m.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    return frame


def draw_gate_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), GREY, 2)
    return frame
