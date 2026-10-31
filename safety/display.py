"""Draw tracking results on a (face-blurred) frame for the local preview."""

import time

import cv2
import numpy as np

from safety.alerts import Level
from safety.detector import Person
from safety.homography import Calibration
from safety.inspector import DENY, UNLOCK, Inspection
from safety.zones import Nearest

GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
AMBER = (0, 191, 255)
WHITE = (255, 255, 255)
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


def _level_color(level: Level) -> tuple[int, int, int]:
    if level == Level.ALARM:
        return RED if int(time.monotonic() * 4) % 2 == 0 else WHITE  # flashing
    return {Level.SAFE: GREY, Level.WARN: AMBER, Level.ALERT: RED}[level]


def draw_floor_outline(frame: np.ndarray, cal: Calibration, floor_pts, color, thickness: int = 2,
                       closed: bool = True) -> None:
    """Draw a floor-coordinate outline, skipping any part beyond the camera's horizon."""
    pts = cal.to_image(list(floor_pts) + ([floor_pts[0]] if closed else []))
    run: list = []
    for p in [*pts, (np.nan, np.nan)]:
        if np.isnan(p).any() or np.abs(p).max() > 1e5:
            if len(run) >= 2:
                cv2.polylines(frame, [np.array(run, np.int32)], isClosed=False, color=color, thickness=thickness)
            run = []
        else:
            run.append(p)


def draw_machines(frame: np.ndarray, cal: Calibration, machines,
                  levels: dict[str, Level] | None = None, rings: bool = True) -> np.ndarray:
    """Outline each machine's floor area, coloured by its alert level, plus its WARN / ALERT rings."""
    from safety.layout import ring

    levels = levels or {}
    for m in machines:
        level = levels.get(m.id, Level.SAFE)
        color = _level_color(level)
        if rings:
            draw_floor_outline(frame, cal, ring(m.polygon, m.warn_m), AMBER, 1)
            draw_floor_outline(frame, cal, ring(m.polygon, m.danger_m), RED, 1)
        draw_floor_outline(frame, cal, m.polygon, color, 4 if level >= Level.ALERT else 2)
        anchor = cal.to_image([m.polygon[0]])[0]
        if not np.isnan(anchor).any():
            label = m.name if level == Level.SAFE else f"{m.name}: {level.name}"
            cv2.putText(frame, label, (int(anchor[0]), int(anchor[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_banner(frame: np.ndarray, announcements: list[tuple[Level, str]]) -> np.ndarray:
    """Current alert messages across the top, highest level first."""
    for i, (level, text) in enumerate(sorted(announcements, key=lambda a: -a[0])):
        y = 28 + i * 30
        cv2.rectangle(frame, (0, y - 22), (frame.shape[1], y + 8), (0, 0, 0), -1)
        cv2.putText(frame, f"{level.name}: {text}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _level_color(level), 2)
    return frame


def draw_gate_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), GREY, 2)
    return frame
