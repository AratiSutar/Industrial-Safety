"""One-off Jev test: send the 3 PPE crops of ONE person from ONE frame and print exactly what comes back.

Usage:
    python tools/jev_check.py --source data/videos/warehouse_forklift.mp4 --frame 300
    python tools/jev_check.py --source data/videos/warehouse_forklift.mp4 --frame 300 --dry-run
    python tools/jev_check.py --source data/images/worker.jpg

Without TYPESAFE_API_KEY in .env (or with --dry-run) nothing is sent: the crops that WOULD be sent
are saved to data/jev_check/ so you can see them. Only use stock or consented footage.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.config import load_settings  # noqa: E402
from safety.crops import ppe_crops  # noqa: E402
from safety.detector import PoseTracker  # noqa: E402
from safety.jev_client import CRITERIA, DANGER_ZONE_HEADERS, QUESTIONS  # noqa: E402
from safety.privacy import FaceDetector, blur_faces  # noqa: E402
from safety.video import open_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="image, video file or camera")
    parser.add_argument("--list-models", action="store_true", help="only show which Jev models your key can use")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="save the crops but don't call Jev")
    parser.add_argument("--out-dir", default="data/jev_check", help="where to save the crops (git-ignored)")
    args = parser.parse_args()

    settings = load_settings()
    if args.list_models:
        if not settings.typesafe_api_key:
            sys.exit("No TYPESAFE_API_KEY in .env")
        from typesafe_sdk import TypeSafeClient

        with TypeSafeClient(api_key=settings.typesafe_api_key) as client:
            print(client.models.list())
        print(f"settings.yaml uses: {settings.jev.model}")
        return
    if not args.source:
        parser.error("--source is required (or use --list-models)")
    if Path(args.source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        frame = cv2.imread(args.source)
        if frame is None:
            sys.exit(f"Could not read image {args.source}")
    else:
        cap = open_source(args.source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Could not read frame {args.frame}")

    tracker = PoseTracker(settings.pose_weights, settings.imgsz, settings.device, settings.backend,
                          settings.tracker, settings.min_keypoint_conf)
    people = tracker.update(frame)
    if not people:
        sys.exit("No person found in this frame; try another --frame")
    person = max(people, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
    print(f"Using the largest of {len(people)} people (box {[round(v) for v in person.box]})")

    face_boxes = FaceDetector(Path(settings.face_model), settings.face_detector_width).detect(frame)
    blurred = blur_faces(frame, people, settings.min_keypoint_conf, face_boxes)
    crops = ppe_crops(blurred, person, settings.min_keypoint_conf)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for item, jpeg in crops.items():
        if jpeg is None:
            print(f"  {item:8}: body part not visible, would answer 'unclear' without asking Jev")
        else:
            (out / f"{item}.jpg").write_bytes(jpeg)
            print(f"  {item:8}: {len(jpeg) // 1024} KB -> {out / (item + '.jpg')}")

    if args.dry_run or not settings.typesafe_api_key:
        reason = "--dry-run" if args.dry_run else "no TYPESAFE_API_KEY in .env"
        print(f"Nothing sent ({reason}). Open the images above to see exactly what would go to Jev.")
        return

    from typesafe_sdk import Choice, Image, RetryPolicy, TypeSafeClient, TypeSafeError

    client = TypeSafeClient(api_key=settings.typesafe_api_key, model=settings.jev.model,
                            timeout=settings.jev.timeout_s, retry=RetryPolicy(max_retries=0),
                            headers=DANGER_ZONE_HEADERS)
    print(f"Calling {settings.jev.model} once per item (no retries)...")
    for item, jpeg in crops.items():
        if jpeg is None:
            continue
        start = time.perf_counter()
        try:
            r = client.system_one({"worker": Image(jpeg)},
                                  {item: Choice(instructions=QUESTIONS[item], criteria=CRITERIA)})
        except TypeSafeError as e:
            print(f"  {item:8}: ERROR {type(e).__name__}: {e}")
            continue
        a = r.choices[item]
        probs = ", ".join(f"{k} {v:.2f}" for k, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1]))
        print(f"  {item:8}: {a.choice:8} confidence {a.confidence:.2f} | {probs} | "
              f"{1000 * (time.perf_counter() - start):.0f} ms | model {r.model} | "
              f"tokens in/out {r.usage.input_tokens}/{r.usage.output_tokens} | request {r.request_id}")
    client.close()


if __name__ == "__main__":
    main()
