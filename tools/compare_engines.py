"""Benchmark the PPE inspection engines side by side on labelled photos: Jev (VLM) vs the local YOLO model.

Usage:
    python tools/compare_engines.py                         # data/eval/labels.csv, both engines
    python tools/compare_engines.py --engines local_ml      # offline: nothing is sent
    python tools/compare_engines.py --list-classes          # class names of the local model

data/eval/labels.csv (git-ignored, like the photos next to it), one row per photo of ONE worker:
    image,hat,vest,boots
    worker_001.jpg,yes,yes,no
    worker_002.jpg,yes,unclear,yes
yes = worn, no = missing, unclear = a person can't tell either (not scored), empty = not labelled.

Both engines get exactly the same face-blurred crops, cut as in the live pipeline, and are scored the
way the gate uses them (below the engine's min_confidence counts as unclear). The report is written
to data/output/compare_report.json. With jev_vlm, 3 crops per photo are sent to Jev: only use stock
or consented photos.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.config import ENGINES, load_settings  # noqa: E402
from safety.crops import ppe_crops, to_jpeg  # noqa: E402
from safety.detector import PoseTracker  # noqa: E402
from safety.engines import EngineUnavailable, make_engine  # noqa: E402
from safety.evaluation import Sample, engine_report, expected_gate, load_labels  # noqa: E402
from safety.inspector import gate_decision  # noqa: E402
from safety.jev_client import PpeAnswer  # noqa: E402
from safety.privacy import FaceDetector, blur_faces  # noqa: E402

DEFINITIONS = {
    "positive": "the PPE item is present",
    "accuracy": "correct confident answers / labelled items (unclear counts as wrong)",
    "accuracy_when_answered": "correct / confident answers (ignores unclear)",
    "false_positive_rate": "item missing but the engine said present / items labelled missing (the dangerous error)",
    "false_negative_rate": "item present but the engine said missing / items labelled present",
    "unclear_rate": "unclear, error or below min_confidence / labelled items (the gate stays locked)",
    "gate.false_unlocks": "people with missing PPE the gate would have let in",
    "latency": "time per engine call; Jev includes the network. per_person_sequential = 3 calls one after another "
               "(the live pipeline sends Jev's 3 calls in parallel)",
    "tokens": "billable Jev input tokens; a worker check = all items of one person. "
              "tools/post_figures.py turns them into cost",
}


def list_classes(settings) -> None:
    from safety.local_ppe import LocalPpeChecker

    lm = settings.local_ml
    try:
        checker = LocalPpeChecker.load(lm.weights, lm.classes, lm.imgsz, settings.device, "pytorch")
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(f"{lm.weights} classes: {', '.join(checker.model.names.values())}")
    for item, ids in checker.ids.items():
        names = {side: [checker.model.names[i] for i in sorted(ids[side])] for side in ("present", "missing")}
        print(f"  {item:8}: present {names['present'] or '-'} | missing {names['missing'] or '-'}")
    if unmapped := checker.unmapped_items():
        print(f"No class for {', '.join(unmapped)}: add the model's names to local_ml.classes in settings.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default="data/eval/labels.csv", help="CSV with image, hat, vest, boots")
    parser.add_argument("--images", help="folder with the photos (default: the labels file's folder)")
    parser.add_argument("--engines", default=",".join(ENGINES), help=f"comma-separated, from {ENGINES}")
    parser.add_argument("--out", default="data/output/compare_report.json", help="JSON report (git-ignored)")
    parser.add_argument("--list-classes", action="store_true", help="only show the local model's class names")
    args = parser.parse_args()

    s = load_settings()
    if args.list_classes:
        list_classes(s)
        return

    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    if unknown := set(names) - set(ENGINES):
        parser.error(f"unknown engines {sorted(unknown)}; allowed {ENGINES}")
    engines = {}
    for name in names:
        try:
            engines[name] = make_engine(name, s)
        except EngineUnavailable as e:
            sys.exit(f"{name}: {e}")
        print(f"{name}: {engines[name].label} | min_confidence {engines[name].min_confidence}")

    labels_path = Path(args.labels)
    if not labels_path.exists():
        sys.exit(f"{labels_path} not found; see the format at the top of this script (--help)")
    try:
        rows = load_labels(labels_path)
    except ValueError as e:
        sys.exit(str(e))
    images_dir = Path(args.images) if args.images else labels_path.parent
    items = s.jev.required_items

    # The first YOLO call loads torch / OpenVINO; keep that out of the timings
    if "local_ml" in engines:
        engines["local_ml"].checker.check(items[0], to_jpeg(np.full((280, 280, 3), 128, np.uint8)))

    faces = FaceDetector(Path(s.face_model), s.face_detector_width)
    samples: dict[str, list[Sample]] = {n: [] for n in engines}
    gates: dict[str, list[tuple[str | None, str]]] = {n: [] for n in engines}
    per_image, skipped = [], []
    for image, truth in rows:
        frame = cv2.imread(str(images_dir / image))
        if frame is None:
            skipped.append({"image": image, "reason": "could not read"})
            print(f"{image}: could not read, skipped")
            continue
        # A new tracker per photo: these are separate photos, not frames of one video
        people = PoseTracker(s.pose_weights, s.imgsz, s.device, s.backend, s.tracker, s.min_keypoint_conf).update(frame)
        if not people:
            # Same for every engine, so skipping keeps the comparison fair
            skipped.append({"image": image, "reason": "no person found"})
            print(f"{image}: no person found, skipped")
            continue
        person = max(people, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
        blurred = blur_faces(frame, people, s.min_keypoint_conf, faces.detect(frame))
        crops = ppe_crops(blurred, person, s.min_keypoint_conf)
        expected = expected_gate(truth, items)

        record = {"image": image, "labels": truth, "expected_gate": expected}
        line = []
        for name, engine in engines.items():
            # The local model answers items it has no class for without running: not a timed call
            no_class = set(getattr(engine.checker, "unmapped_items", list)())
            answers = {}
            for item in items:
                jpeg = crops.get(item)
                answers[item] = (engine.checker.check(item, jpeg) if jpeg is not None
                                 else PpeAnswer(item, "unclear", 0.0, error="body part not visible"))
                called = jpeg is not None and item not in no_class
                samples[name].append(Sample(image, item, truth.get(item, ""), answers[item], called))
            decision = gate_decision(answers, items, engine.min_confidence)
            gates[name].append((expected, decision))
            record[name] = {
                "decision": decision,
                "answers": {i: {"choice": a.choice, "confidence": round(a.confidence, 3),
                                "latency_ms": round(1000 * a.latency_s, 1), "error": a.error,
                                "input_tokens": a.input_tokens}
                            for i, a in answers.items()},
            }
            line.append(f"{name} {decision}")
        per_image.append(record)
        print(f"{image}: expected {expected or 'not scored'} | " + " | ".join(line))

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"labels": str(labels_path), "rows": len(rows), "evaluated": len(per_image), "skipped": skipped},
        "items": list(items),
        "definitions": DEFINITIONS,
        "engines": {name: {"label": engine.label,
                           **engine_report(samples[name], gates[name], engine.min_confidence, items)}
                    for name, engine in engines.items()},
        "images": per_image,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'engine':10} {'accuracy':>9} {'answered':>9} {'FPR':>6} {'unclear':>8} {'false unlocks':>14} "
          f"{'p50 ms':>8} {'p95 ms':>8}")
    for name, r in report["engines"].items():
        o, g, lat = r["overall"], r["gate"], r["latency"]["per_call"]
        cells = [o["accuracy"], o["accuracy_when_answered"], o["false_positive_rate"], o["unclear_rate"]]
        print(f"{name:10} " + " ".join(f"{'-' if v is None else f'{v:.2f}':>{w}}"
                                       for v, w in zip(cells, (9, 9, 6, 8), strict=True))
              + f" {g['false_unlocks']:>6} of {g['people_missing_ppe']:<5}"
              + f" {lat.get('p50_ms', '-'):>8} {lat.get('p95_ms', '-'):>8}")
    for name, r in report["engines"].items():
        if tokens := r["tokens"].get("mean_input_per_worker_check"):
            print(f"{name}: {tokens:.0f} input tokens per worker check")
    print(f"Report: {out}  ({len(per_image)} of {len(rows)} photos evaluated)")
    print("Post table and cost graph: python tools/post_figures.py --jev-usd-per-mtok <price>")


if __name__ == "__main__":
    main()
