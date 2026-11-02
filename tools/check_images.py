"""Check still images: people, face blur, hat and shoe boxes and the PPE crops Jev would receive.

Usage:
    python tools/check_images.py --source data/images            # every image in a folder
    python tools/check_images.py --source data/images/worker.jpg --show
    python tools/check_images.py --source data/images/worker.jpg --jev

For each image an annotated copy is saved to data/output/<name>_checked.jpg (git-ignored):
faces blurred, green person boxes, blue hard-hat boxes, yellow shoe boxes (orange = ankles hidden),
and underneath the hard-hat / vest / boots crops exactly as they would be sent to Jev.
Nothing is sent anywhere, unless --jev: then those crops are sent to Jev and its answers are
written on each person (green = confident yes, red = confident no, orange = unclear).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.config import load_settings  # noqa: E402
from safety.crops import ppe_crops  # noqa: E402
from safety.detector import PoseTracker  # noqa: E402
from safety.display import PPE_LABELS, draw_people, ui_scale  # noqa: E402
from safety.inspector import DENY, UNLOCK, gate_decision  # noqa: E402
from safety.jev_client import JevPpeClient, PpeAnswer  # noqa: E402
from safety.privacy import FaceDetector, blur_faces  # noqa: E402

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CROP_HEIGHT = 300
DECISION_TEXT = {UNLOCK: "ALL PPE OK", DENY: "MISSING PPE"}  # anything else: not sure, check again


def ask_jev(checker: JevPpeClient, crops: dict[str, bytes | None], required: tuple[str, ...],
            min_conf: float) -> tuple[dict[str, PpeAnswer], str]:
    answers = {}
    for item in required:
        jpeg = crops.get(item)
        answers[item] = (checker.check(item, jpeg) if jpeg is not None
                         else PpeAnswer(item, "unclear", 0.0, error="body part not visible"))
    return answers, gate_decision(answers, required, min_conf)


def draw_answers(view: np.ndarray, box, answers: dict[str, PpeAnswer], decision: str, min_conf: float) -> None:
    """One line per item inside the top of the person box, e.g. 'HAT yes 0.92', then the decision."""
    s = ui_scale(view)
    lines = []
    for item, a in answers.items():
        sure = a.confidence >= min_conf
        color = (0, 200, 0) if a.choice == "yes" and sure else (0, 0, 255) if a.choice == "no" and sure \
            else (0, 140, 255)
        lines.append((f"{PPE_LABELS.get(item, item)} {a.choice} {a.confidence:.2f}", color))
    color = (0, 200, 0) if decision == UNLOCK else (0, 0, 255) if decision == DENY else (0, 140, 255)
    lines.append((DECISION_TEXT.get(decision, "CHECK AGAIN"), color))
    for i, (text, color) in enumerate(lines):
        org = (int(box[0] + 6 * s), int(box[1] + 26 * s * (i + 1)))
        # Black outline first, so the text reads on any background
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), max(2, round(5 * s)))
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, color, max(1, round(2 * s)))


def crop_strip(blurred: np.ndarray, people, min_conf: float, width: int) -> np.ndarray | None:
    """One row per person: hard hat, vest, boots crops side by side, labelled."""
    rows = []
    for i, p in enumerate(people, start=1):
        tiles = []
        for item, jpeg in ppe_crops(blurred, p, min_conf).items():
            if jpeg is None:
                img = np.full((CROP_HEIGHT, CROP_HEIGHT, 3), 60, np.uint8)
                label = f"#{i} {item}: not visible"
            else:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                label = f"#{i} {item}"
            img = cv2.resize(img, (round(img.shape[1] * CROP_HEIGHT / img.shape[0]), CROP_HEIGHT))
            cv2.putText(img, label, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(img)
        row = np.hstack(tiles)
        if row.shape[1] > width:
            row = cv2.resize(row, (width, round(row.shape[0] * width / row.shape[1])))
        rows.append(cv2.copyMakeBorder(row, 0, 8, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)))
    return np.vstack(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="an image file or a folder of images")
    parser.add_argument("--out-dir", default="data/output", help="where to save the checked images")
    parser.add_argument("--show", action="store_true", help="also show each result in a window (any key = next)")
    parser.add_argument("--jev", action="store_true", help="send the crops to Jev and write its answers on the image")
    args = parser.parse_args()

    src = Path(args.source)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_TYPES) if src.is_dir() else [src]
    if not paths:
        sys.exit(f"No images found in {src}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = load_settings()
    if args.jev and not s.typesafe_api_key:
        sys.exit("--jev needs TYPESAFE_API_KEY in .env")
    checker = JevPpeClient(s.typesafe_api_key, s.jev.model, s.jev.timeout_s) if args.jev else None
    faces = FaceDetector(Path(s.face_model), s.face_detector_width)
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path.name}: could not read, skipped")
            continue
        # A new tracker per image: these are separate photos, not frames of one video
        people = PoseTracker(s.pose_weights, s.imgsz, s.device, s.backend, s.tracker, s.min_keypoint_conf).update(frame)
        face_boxes = faces.detect(frame)
        blurred = blur_faces(frame, people, s.min_keypoint_conf, face_boxes)

        view = draw_people(blurred.copy(), people, min_conf=s.min_keypoint_conf)
        if checker is not None:
            for i, p in enumerate(people, start=1):
                crops = ppe_crops(blurred, p, s.min_keypoint_conf)
                answers, decision = ask_jev(checker, crops, s.jev.required_items, s.jev.min_confidence)
                draw_answers(view, p.box, answers, decision, s.jev.min_confidence)
                summary = ", ".join(f"{item} {a.choice} {a.confidence:.2f}" + (f" ({a.error})" if a.error else "")
                                    for item, a in answers.items())
                print(f"  person #{i}: {summary} -> {DECISION_TEXT.get(decision, 'CHECK AGAIN')}")
        if view.shape[1] > 1920:  # keep files a sensible size; crops are added after, at full size
            view = cv2.resize(view, (1920, round(view.shape[0] * 1920 / view.shape[1])), interpolation=cv2.INTER_AREA)
        if (strip := crop_strip(blurred, people, s.min_keypoint_conf, view.shape[1])) is not None:
            view = np.vstack([view, strip])
        out = out_dir / f"{path.stem}_checked.jpg"
        cv2.imwrite(str(out), view, [cv2.IMWRITE_JPEG_QUALITY, 90])

        ankles = sum(p.foot_from_ankles for p in people)
        print(f"{path.name}: {len(people)} people ({ankles} with ankles visible), "
              f"{len(face_boxes)} faces found by the face detector -> {out}")

        if args.show:
            h = min(view.shape[0], 1000)  # fit on a laptop screen
            cv2.imshow("Checked image (any key = next, q = quit)",
                       cv2.resize(view, (round(view.shape[1] * h / view.shape[0]), h)))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
