"""Orchestrator. Milestone 2: tracks people and finds where each one stands."""

import argparse
import time

import cv2

from safety.config import load_settings
from safety.detector import PoseTracker
from safety.display import draw_people
from safety.privacy import blur_faces
from safety.video import frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial safety monitor")
    parser.add_argument("--source", help="override video.source from settings.yaml")
    parser.add_argument("--backend", choices=("openvino", "pytorch"), help="override model.backend")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = run forever)")
    parser.add_argument("--show", action="store_true", help="open a local preview window (faces blurred)")
    args = parser.parse_args()

    settings = load_settings()
    source = args.source or settings.video_source
    backend = args.backend or settings.backend

    print(f"Mode: {settings.mode}")
    print(f"Video source: {source} (processing 1 of every {settings.process_every_n_frames} frames)")
    print(f"Machines configured: {len(settings.machines)}")
    print(f"Jev model: {settings.jev_model} | API key set: {'yes' if settings.typesafe_api_key else 'no'}")
    print(f"Pose model: {settings.pose_weights} | backend: {backend} | imgsz: {settings.imgsz}")

    tracker = PoseTracker(settings.pose_weights, settings.imgsz, settings.device, backend,
                          settings.tracker, settings.min_keypoint_conf)

    processed = 0
    infer_s = 0.0
    try:
        for index, frame in frames(source, settings.process_every_n_frames):
            t0 = time.perf_counter()
            people = tracker.update(frame)
            infer_s += time.perf_counter() - t0
            processed += 1

            if processed % 30 == 0:
                feet = ", ".join(f"#{p.track_id}@({p.foot[0]:.0f},{p.foot[1]:.0f})"
                                 f"{'' if p.foot_from_ankles else '*'}" for p in people)
                print(f"frame {index}: {len(people)} people {feet}  "
                      f"[{1000 * infer_s / processed:.0f} ms/frame]")

            if args.show:
                # Faces are always blurred before anything is displayed
                view = draw_people(blur_faces(frame, people, settings.min_keypoint_conf), people)
                cv2.imshow("Industrial Safety (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        if args.show:
            cv2.destroyAllWindows()

    if processed:
        ms = 1000 * infer_s / processed
        print(f"Done. {processed} frames, {ms:.0f} ms per frame = {1000 / ms:.1f} analysed fps "
              f"(* = ankles hidden, used bottom of box)")


if __name__ == "__main__":
    main()
