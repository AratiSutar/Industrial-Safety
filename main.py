"""Orchestrator: tracking -> floor distance -> proximity alerts, plus background Jev PPE checks."""

import argparse
import time
from pathlib import Path

import cv2

from safety import homography
from safety.alerts import AlertEngine, Level, Observation
from safety.config import load_settings
from safety.crops import ppe_crops
from safety.detector import PoseTracker
from safety.display import draw_banner, draw_gate_box, draw_machines, draw_people
from safety.events import EventLog
from safety.gate import GateWatcher
from safety.inspector import AsyncInspector, Inspection
from safety.jev_client import FakePpeClient, JevPpeClient
from safety.outputs import OutputRouter
from safety.privacy import blur_faces
from safety.video import frames, is_live
from safety.webview import WebView
from safety.zones import MachineZones


def describe(ins: Inspection) -> str:
    parts = []
    for item, a in ins.answers.items():
        text = f"{item}={a.choice}"
        if a.confidence:
            text += f"({a.confidence:.2f})"
        if a.error:
            text += f"[{a.error}]"
        parts.append(text)
    return f"Jev #{ins.track_id}: {ins.decision} after {ins.latency_s:.2f}s  " + " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial safety monitor")
    parser.add_argument("--source", help="override video.source from settings.yaml")
    parser.add_argument("--backend", choices=("openvino", "pytorch"), help="override model.backend")
    parser.add_argument("--mode", choices=("GATE", "FACTORY"), help="override mode")
    parser.add_argument("--camera", help="override video.camera (picks the calibration and machines)")
    parser.add_argument("--config-dir", default="config", help="folder with settings.yaml and machines.yaml")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = run forever)")
    parser.add_argument("--show", action="store_true", help="open a local preview window (faces blurred)")
    parser.add_argument("--web", type=int, metavar="PORT",
                        help="serve the preview in a browser on this port, e.g. 8000 (for Docker / no screen)")
    parser.add_argument("--loop", action="store_true", help="video files: start again at the end (demos)")
    parser.add_argument("--fake-jev", action="store_true",
                        help="offline Jev stand-in (always 'unclear', never unlocks) to test the pipeline")
    args = parser.parse_args()

    settings = load_settings(Path(args.config_dir))
    if args.camera:
        settings.camera = args.camera
    source = args.source or settings.video_source
    backend = args.backend or settings.backend
    mode = args.mode or settings.mode

    print(f"Mode: {mode} | camera: {settings.camera}")
    reading = ("live: newest frame" if is_live(source)
               else f"file: 1 of every {settings.process_every_n_frames} frames")
    print(f"Video source: {source} ({reading})")
    print(f"Pose model: {settings.pose_weights} | backend: {backend} | imgsz: {settings.imgsz}")

    # Block 2: floor distance (needs a calibration for this camera)
    cal = homography.load(settings.camera, settings.calibration_dir)
    zones = MachineZones(settings.machines, settings.camera)
    if cal is None:
        print(f"Floor distance: OFF (no calibration for {settings.camera}; run tools/calibrate.py)")
    else:
        print(f"Floor distance: ON | calibration error {100 * cal.error_m:.1f} cm | "
              f"{len(zones.machines)} machines on this camera")

    # Block 3: Jev, in background threads
    inspector = None
    if args.fake_jev:
        checker = FakePpeClient()
        print("Jev: FAKE client (offline test, always 'unclear')")
    elif settings.typesafe_api_key:
        checker = JevPpeClient(settings.typesafe_api_key, settings.jev.model, settings.jev.timeout_s)
        print(f"Jev: {settings.jev.model} | items: {', '.join(settings.jev.required_items)}")
    else:
        checker = None
        print("Jev: OFF (TYPESAFE_API_KEY not set; use --fake-jev to test offline)")
    if checker is not None:
        cooldown = 10.0 if mode == "GATE" else settings.jev.recheck_s
        if mode == "FACTORY" and cooldown <= 0:
            print("Jev: factory re-checks disabled (jev.recheck_s = 0)")
        else:
            inspector = AsyncInspector(checker, settings.jev.required_items, settings.jev.min_confidence,
                                       settings.jev.timeout_s, settings.jev.max_workers, cooldown)

    tracker = PoseTracker(settings.pose_weights, settings.imgsz, settings.device, backend,
                          settings.tracker, settings.min_keypoint_conf)
    gate = GateWatcher(settings.gate, settings.min_keypoint_conf)

    # Proximity alerts
    alerts = AlertEngine(zones.machines, settings.alerts)
    router = OutputRouter()
    events = EventLog(settings.alerts.event_log)
    announcements: dict[str, tuple[Level, str]] = {}  # machine id -> current level and message
    unsupported = sorted({m.running_source for m in zones.machines} - {"always"})
    if unsupported:
        print(f"Machine running state: {', '.join(unsupported)} not connected yet; "
              "treating those machines as running (fail-safe)")

    def machine_running(machine) -> bool:
        return True  # 'always' for now; motion detection / PLC signal plug in here

    web = None
    if args.web:
        web = WebView(args.web)
        print(f"Browser preview: http://localhost:{args.web} (faces blurred)")

    def video():
        while True:
            yield from frames(source, settings.process_every_n_frames)
            if not args.loop or is_live(source):
                return
            print("Video ended, starting again (--loop)")

    processed = 0
    infer_s = loop_s = 0.0
    last_ppe: dict[int | None, Inspection] = {}  # latest Jev result per person, for the display
    try:
        for index, frame in video():
            t0 = time.perf_counter()
            people = tracker.update(frame)
            t1 = time.perf_counter()

            nearest, observations = {}, []
            if cal is not None and people:
                floor = cal.to_floor([p.foot for p in people])
                for p, xy in zip(people, floor, strict=True):
                    xy = (float(xy[0]), float(xy[1]))
                    if (n := zones.nearest(xy)) is not None:
                        nearest[p.track_id] = n
                    if p.track_id is None:
                        continue  # can't time how long an untracked person stays
                    observations += [Observation(p.track_id, d.machine.id, d.distance_m,
                                                 zones.in_operator_spot(d.machine, xy))
                                     for d in zones.distances(xy)]

            changes, closed = alerts.update(observations, machine_running)
            for change in changes:
                print(router.apply(change))
                if change.new > Level.SAFE:
                    announcements[change.machine.id] = (change.new, change.message)
                else:
                    announcements.pop(change.machine.id, None)
            events.write(closed)

            # Faces are always blurred before anything is shown or sent out
            blurred = blur_faces(frame, people, settings.min_keypoint_conf)

            if inspector is not None:
                candidates = gate.ready(people, frame.shape) if mode == "GATE" else people
                for p in candidates:
                    if p.track_id is not None and inspector.can_submit(p.track_id):
                        inspector.submit(p.track_id, ppe_crops(frame, blurred, p, settings.min_keypoint_conf))
                for ins in inspector.poll():
                    print(describe(ins))
                    last_ppe[ins.track_id] = ins

            t2 = time.perf_counter()
            infer_s += t1 - t0
            loop_s += t2 - t0
            processed += 1

            if processed % 30 == 0:
                feet = ", ".join(
                    f"#{p.track_id}"
                    + (f" {nearest[p.track_id].machine.name} {nearest[p.track_id].distance_m:.2f}m"
                       if p.track_id in nearest else f"@({p.foot[0]:.0f},{p.foot[1]:.0f})")
                    + ("" if p.foot_from_ankles else "*")
                    for p in people)
                jev = f" | Jev pending {inspector.pending}" if inspector else ""
                print(f"frame {index}: {len(people)} people {feet}  "
                      f"[{1000 * loop_s / processed:.0f} ms/frame{jev}]")
                # Forget results for people no longer in view
                current = {p.track_id for p in people}
                last_ppe = {tid: ins for tid, ins in last_ppe.items() if tid in current}

            if args.show or web is not None:
                view = draw_people(blurred.copy(), people, nearest, last_ppe)
                if cal is not None:
                    draw_machines(view, cal, zones.machines, {m.id: alerts.level(m.id) for m in zones.machines})
                if mode == "GATE":
                    draw_gate_box(view, settings.gate.box)
                draw_banner(view, list(announcements.values()))
                if web is not None:
                    web.publish(view)

            if args.show:
                cv2.imshow("Industrial Safety (q = quit, a = operator announced)", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):  # stand-in for the operator screen's "Announced" button
                    for m in zones.machines:
                        if alerts.acknowledge(m.id):
                            print(f"[ACK  ] {m.name}: operator announced")

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        if inspector is not None:
            # Let in-flight checks finish briefly so their results are reported
            deadline = time.monotonic() + inspector.deadline_s
            while inspector.pending and time.monotonic() < deadline:
                for ins in inspector.poll():
                    print(describe(ins))
                time.sleep(0.05)
            inspector.close()
        events.write(alerts.close_all())
        if web is not None:
            web.close()
        if args.show:
            cv2.destroyAllWindows()

    if processed:
        print(f"Done. {processed} frames | pose {1000 * infer_s / processed:.0f} ms | "
              f"whole loop {1000 * loop_s / processed:.0f} ms = {processed / loop_s:.1f} fps "
              f"(* = ankles hidden, used bottom of box)")


if __name__ == "__main__":
    main()
