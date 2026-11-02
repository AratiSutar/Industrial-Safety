"""Draw tracking results on a (face-blurred) frame for the local preview."""

import time

import cv2
import numpy as np

from safety.alerts import Level
from safety.crops import head_region
from safety.detector import L_ANKLE, R_ANKLE, Box, Person
from safety.homography import Calibration
from safety.inspector import DENY, UNLOCK, Inspection
from safety.zones import Nearest

GREEN = (0, 200, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
AMBER = (0, 191, 255)
BLUE = (255, 128, 0)
WHITE = (255, 255, 255)
GREY = (160, 160, 160)

PPE_LABELS = {"hard_hat": "HAT", "vest": "VEST", "boots": "BOOTS"}
ANSWER_MARKS = {"yes": "ok", "no": "NO", "unclear": "?", "error": "err"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ui_scale(frame: np.ndarray) -> float:
    """Text and line sizes grow with the video resolution (1.0 at 1280 px wide)."""
    return max(0.5, frame.shape[1] / 1280)


def _text(frame, text, org, color, size: float = 0.6) -> None:
    s = ui_scale(frame)
    cv2.putText(frame, text, (int(org[0]), int(org[1])), FONT, size * s, color, max(1, round(2 * s)))


def _thick(frame, base: int = 2) -> int:
    return max(1, round(base * ui_scale(frame)))


def feet_box(p: Person, min_conf: float = 0.5) -> Box:
    """Box around the shoes: from the ankles when visible, otherwise the bottom of the body box."""
    x1, y1, x2, y2 = p.box
    h = y2 - y1
    ankles = [p.keypoints[i] for i in (L_ANKLE, R_ANKLE) if p.keypoint_conf[i] >= min_conf]
    if ankles:
        xs, ys = [float(a[0]) for a in ankles], [float(a[1]) for a in ankles]
        pad = 0.07 * h  # roughly half a shoe length
        return (min(xs) - pad, min(ys) - pad * 0.6, max(xs) + pad, max(max(ys) + pad, y2))
    cx, half = (x1 + x2) / 2, (x2 - x1) * 0.3
    return (cx - half, y2 - 0.1 * h, cx + half, y2)


def ppe_label(ins: Inspection) -> tuple[str, tuple[int, int, int]]:
    """Jev's latest answers as text, e.g. 'HAT ok VEST NO BOOTS ?', coloured by the decision."""
    text = " ".join(f"{PPE_LABELS.get(item, item)} {ANSWER_MARKS.get(a.choice, a.choice)}"
                    for item, a in ins.answers.items())
    color = GREEN if ins.decision == UNLOCK else RED if ins.decision == DENY else ORANGE
    return text, color


def draw_people(frame: np.ndarray, people: list[Person],
                nearest: dict[int | None, Nearest] | None = None,
                ppe: dict[int | None, Inspection] | None = None,
                min_conf: float = 0.5) -> np.ndarray:
    nearest = nearest or {}
    ppe = ppe or {}
    s = ui_scale(frame)
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, _thick(frame))
        label = f"#{p.track_id}" if p.track_id is not None else "?"
        if (n := nearest.get(p.track_id)) is not None:
            label += f" {n.machine.name}: {n.distance_m:.1f} m"
        _text(frame, label, (x1, max(18 * s, y1 - 8 * s)), GREEN)
        if (ins := ppe.get(p.track_id)) is not None:
            text, color = ppe_label(ins)
            _text(frame, text, (x1 + 4 * s, y1 + 24 * s), color)
        # Yellow box = feet found from the ankles, orange = ankles hidden, estimated from the body box
        fx1, fy1, fx2, fy2 = (int(v) for v in feet_box(p, min_conf))
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), YELLOW if p.foot_from_ankles else ORANGE, _thick(frame))
        # Blue box = the area Jev is shown for the hard hat (it stops above the eyebrows)
        hx1, hy1, hx2, hy2 = (int(v) for v in head_region(p, min_conf))
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), BLUE, _thick(frame))
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
            draw_floor_outline(frame, cal, ring(m.polygon, m.warn_m), AMBER, _thick(frame, 1))
            draw_floor_outline(frame, cal, ring(m.polygon, m.danger_m), RED, _thick(frame, 1))
        draw_floor_outline(frame, cal, m.polygon, color, _thick(frame, 4 if level >= Level.ALERT else 2))
        anchor = cal.to_image([m.polygon[0]])[0]
        if not np.isnan(anchor).any():
            _text(frame, m.name if level == Level.SAFE else f"{m.name}: {level.name}", anchor, color)
    return frame


def draw_banner(frame: np.ndarray, announcements: list[tuple[Level, str]]) -> np.ndarray:
    """Current alert messages across the top, highest level first."""
    s = ui_scale(frame)
    for i, (level, text) in enumerate(sorted(announcements, key=lambda a: -a[0])):
        y = (28 + i * 30) * s
        cv2.rectangle(frame, (0, int(y - 22 * s)), (frame.shape[1], int(y + 8 * s)), (0, 0, 0), -1)
        _text(frame, f"{level.name}: {text}", (8 * s, y), _level_color(level))
    return frame


def draw_gate_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), GREY, _thick(frame))
    return frame
