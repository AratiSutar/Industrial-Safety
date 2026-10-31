"""Draw machine floor areas by clicking on a camera image, and save them to config/machines.yaml.

Usage:
    python tools/draw_machines.py --camera cam_floor_1 --source 0
    python tools/draw_machines.py --camera cam_warehouse --source data/videos/warehouse.mp4 --frame 120

Needs the camera's calibration first (tools/calibrate.py).

Click the corners where the machine MEETS THE FLOOR, in order around its base. Not the top of
the machine: only points on the floor convert to correct floor positions.

Keys:  n = finish this shape     u = undo last click     o = add an operator spot to the last machine
       x = delete last machine   s = save and quit       q = quit without saving
Amber ring = where WARN starts, red ring = where ALERT starts. Only coordinates are saved, never the image.
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety import homography, layout  # noqa: E402
from safety.config import ConfigError, parse_machine  # noqa: E402
from safety.display import AMBER, GREEN, RED, draw_floor_outline  # noqa: E402
from safety.video import open_source  # noqa: E402

WINDOW = "Draw machines: click base corners | n=finish u=undo o=operator spot x=delete s=save q=quit"
CYAN = (255, 255, 0)


def ask(prompt: str, default, cast=str):
    while True:
        text = input(f"  {prompt} [{default}]: ").strip()
        if not text:
            return default
        try:
            return cast(text)
        except ValueError:
            print("  Please try again.")


def yes_no(text: str) -> bool:
    if text.lower() in ("y", "yes"):
        return True
    if text.lower() in ("n", "no"):
        return False
    raise ValueError(text)


def draw(frame, cal, entries, clicks, mode):
    view = frame.copy()
    for e in entries:
        m = parse_machine(e)
        draw_floor_outline(view, cal, layout.ring(m.polygon, m.warn_m), AMBER, 1)
        draw_floor_outline(view, cal, layout.ring(m.polygon, m.danger_m), RED, 1)
        draw_floor_outline(view, cal, m.polygon, RED, 2)
        if m.operator_spot:
            draw_floor_outline(view, cal, m.operator_spot, GREEN, 2)
        x, y = cal.to_image([m.polygon[0]])[0]
        cv2.putText(view, m.name, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    color = GREEN if mode == "spot" else CYAN
    for i, (x, y) in enumerate(clicks):
        cv2.circle(view, (x, y), 5, color, -1)
        if i:
            cv2.line(view, clicks[i - 1], (x, y), color, 2)
    status = "OPERATOR SPOT: click its corners, then n" if mode == "spot" else f"{len(entries)} machine(s)"
    cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return view


def read_frame(source: str, frame_no: int):
    cap = open_source(source)
    if frame_no:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read frame {frame_no} from {source}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="camera id, as used for the calibration")
    parser.add_argument("--source", required=True, help="webcam index, video file or rtsp:// URL")
    parser.add_argument("--frame", type=int, default=0, help="video files: which frame to draw on")
    parser.add_argument("--calibration-dir", default="calibration")
    parser.add_argument("--machines", default="config/machines.yaml", help="layout file to update")
    args = parser.parse_args()

    cal = homography.load(args.camera, Path(args.calibration_dir))
    if cal is None:
        sys.exit(f"No calibration for {args.camera}. Run tools/calibrate.py first.")
    frame = read_frame(args.source, args.frame)
    path = Path(args.machines)

    # Start from this camera's machines already in the file
    entries = [m for m in layout.load_raw(path) if m.get("camera") == args.camera]
    print(f"{len(entries)} existing machine(s) for {args.camera}. Click the base corners of a machine.")

    clicks: list[tuple[int, int]] = []
    mode = "machine"
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    while True:
        cv2.imshow(WINDOW, draw(frame, cal, entries, clicks, mode))
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u") and clicks:
            clicks.pop()
        elif key == ord("x") and entries and not clicks:
            print(f"Deleted {entries.pop()['name']}")
        elif key == ord("o"):
            if entries:
                mode, clicks = "spot", []
                print(f"Operator spot for {entries[-1]['name']}: click its corners, then press n")
            else:
                print("Draw a machine first.")
        elif key == ord("n"):
            try:
                polygon = layout.floor_polygon(cal, clicks)
                if mode == "spot":
                    candidate = {**entries[-1], "operator_spot": polygon}
                    parse_machine(candidate)
                    entries[-1] = candidate
                    print(f"  Operator spot added to {candidate['name']}")
                else:
                    prev = entries[-1] if entries else {}
                    n = len(entries) + 1
                    print(f"Machine {n}: floor area {polygon}")
                    id_ = ask("id (short, no spaces)", f"machine_{n}")
                    name = ask("name used in announcements", id_.replace("_", " ").title())
                    warn = ask("warn distance, meters", prev.get("warn_m", 1.5), float)
                    danger = ask("danger distance, meters", prev.get("danger_m", 0.8), float)
                    noisy = ask("noisy area, ear defenders required? (y/n)", "n", yes_no)
                    entry = layout.machine_entry(id_, name, args.camera, polygon, warn, danger, noisy)
                    layout.merge([], args.camera, [*entries, entry])  # catches duplicate ids
                    entries.append(entry)
                    print(f"  Added {name}. Press o to add an operator spot, or click the next machine.")
                clicks, mode = [], "machine"
            except ConfigError as e:
                print(f"  {e}")
        elif key == ord("s"):
            try:
                backup = layout.save(path, args.camera, entries)
            except ConfigError as e:
                print(f"  Not saved: {e}")
                continue
            print(f"Saved {len(entries)} machine(s) to {path}" + (f" (previous version: {backup})" if backup else ""))
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
