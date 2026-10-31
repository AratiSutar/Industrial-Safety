"""Orchestrator. For now it only loads config and reads video (milestone 1)."""

import argparse
import time

from safety.config import load_settings
from safety.video import frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial safety monitor")
    parser.add_argument("--source", help="override video.source from settings.yaml")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = run forever)")
    args = parser.parse_args()

    settings = load_settings()
    source = args.source or settings.video_source

    print(f"Mode: {settings.mode}")
    print(f"Video source: {source} (processing 1 of every {settings.process_every_n_frames} frames)")
    print(f"Machines configured: {len(settings.machines)}")
    print(f"Jev model: {settings.jev_model} | API key set: {'yes' if settings.typesafe_api_key else 'no'}")

    processed = 0
    start = time.perf_counter()
    for index, frame in frames(source, settings.process_every_n_frames):
        processed += 1
        if processed % 30 == 0:
            fps = processed / (time.perf_counter() - start)
            print(f"frame {index}: {frame.shape[1]}x{frame.shape[0]}, {fps:.1f} processed fps")
        if args.max_frames and processed >= args.max_frames:
            break

    print(f"Done. Processed {processed} frames.")


if __name__ == "__main__":
    main()
