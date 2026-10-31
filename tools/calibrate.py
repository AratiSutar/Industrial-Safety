"""Camera calibration: click floor points in the image and type their real positions in meters.

Usage:
    python tools/calibrate.py --camera cam_floor_1 --source 0
    python tools/calibrate.py --camera cam_floor_1 --source path/to/video.mp4

Before you start, mark at least 4 points on the floor with tape and measure their positions
from one corner (x to the right, y forward, in meters). Spread them out; never 3 on one line.

In the window: left-click a point, then type its floor position in the terminal.
Keys: u = undo last point, s = save (needs 4+ points), q = quit without saving.
Only the point coordinates are saved, never the image.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety import homography  # noqa: E402
from safety.video import open_source  # noqa: E402

WINDOW = "Calibration: click floor points (u=undo, s=save, q=quit)"


def ask_floor_point(n: int) -> tuple[float, float] | None:
    while True:
        text = input(f"Point {n}: floor position in meters as x,y (blank = cancel): ").strip()
        if not text:
            return None
        try:
            x, y = (float(v) for v in text.replace(" ", "").split(","))
            return x, y
        except ValueError:
            print("  Please type two numbers like 2.5,4")


def draw(frame, image_pts, floor_pts, cal=None):
    view = frame.copy()
    if cal is not None:
        # 1 m floor grid projected back onto the image: lines should follow the floor
        xs, ys = zip(*floor_pts, strict=True)
        for gx in np.arange(np.floor(min(xs)) - 1, np.ceil(max(xs)) + 2):
            line = cal.to_image([(gx, min(ys) - 1), (gx, max(ys) + 1)]).astype(int)
            cv2.line(view, tuple(line[0]), tuple(line[1]), (255, 200, 0), 1)
        for gy in np.arange(np.floor(min(ys)) - 1, np.ceil(max(ys)) + 2):
            line = cal.to_image([(min(xs) - 1, gy), (max(xs) + 1, gy)]).astype(int)
            cv2.line(view, tuple(line[0]), tuple(line[1]), (255, 200, 0), 1)
    for (x, y), (fx, fy) in zip(image_pts, floor_pts, strict=True):
        cv2.circle(view, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(view, f"({fx:g},{fy:g})", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", required=True, help="camera id, as used in settings.yaml and machines.yaml")
    parser.add_argument("--source", required=True, help="webcam index, video file or rtsp:// URL")
    parser.add_argument("--out-dir", default="calibration", help="where to save <camera>.json")
    args = parser.parse_args()

    cap = open_source(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame from {args.source}")

    image_pts: list[tuple[float, float]] = []
    floor_pts: list[tuple[float, float]] = []
    clicked: list[tuple[int, int]] = []
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda e, x, y, *_: clicked.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    cal = None
    while True:
        cv2.imshow(WINDOW, draw(frame, image_pts, floor_pts, cal))
        key = cv2.waitKey(30) & 0xFF

        if clicked:
            x, y = clicked.pop(0)
            floor = ask_floor_point(len(image_pts) + 1)
            if floor is not None:
                image_pts.append((float(x), float(y)))
                floor_pts.append(floor)
        elif key == ord("u") and image_pts:
            image_pts.pop()
            floor_pts.pop()
        elif key == ord("q"):
            print("Quit without saving.")
            break
        elif key == ord("s"):
            if cal is None:
                print("Need at least 4 valid points before saving.")
                continue
            path = homography.save(cal, Path(args.out_dir))
            print(f"Saved {path}")
            break
        else:
            continue

        cal = None
        if len(image_pts) >= 4:
            try:
                cal = homography.compute(args.camera, image_pts, floor_pts)
                if len(image_pts) == 4:
                    quality = "4 points always fit exactly; add a 5th or 6th to check accuracy"
                elif cal.error_m <= homography.MAX_GOOD_ERROR_M:
                    quality = "good"
                else:
                    quality = "CHECK YOUR MEASUREMENTS"
                print(f"  {len(image_pts)} points, error {100 * cal.error_m:.1f} cm ({quality}). "
                      "The blue 1 m grid should follow the floor. Press s to save.")
            except ValueError as e:
                print(f"  {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
