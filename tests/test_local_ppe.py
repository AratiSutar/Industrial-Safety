import time

import numpy as np

from safety.config import DEFAULT_LOCAL_CLASSES
from safety.crops import to_jpeg
from safety.inspector import RETRY, AsyncInspector
from safety.local_ppe import LocalPpeChecker, class_ids, decide, normalize

# Class names as in a typical public construction-site PPE dataset (no boots class)
NAMES = {0: "Hardhat", 1: "NO-Hardhat", 2: "Safety Vest", 3: "NO-Safety Vest", 4: "Person"}
JPEG = to_jpeg(np.full((300, 300, 3), 128, np.uint8))


class Boxes:
    def __init__(self, detections):
        self.cls = np.array([c for c, _ in detections], float)
        self.conf = np.array([p for _, p in detections], float)


class Result:
    def __init__(self, detections):
        self.boxes = Boxes(detections)


class FakeYolo:
    names = NAMES

    def __init__(self, detections):
        self.detections = detections

    def predict(self, img, **kwargs):
        return [Result(self.detections)]


def test_class_names_match_ignoring_case_and_separators():
    assert normalize("NO-Hardhat") == normalize("no_hardhat") == normalize("No Hardhat") == "nohardhat"
    ids = class_ids(NAMES, DEFAULT_LOCAL_CLASSES)
    assert ids["hard_hat"] == {"present": {0}, "missing": {1}}
    assert ids["vest"] == {"present": {2}, "missing": {3}}
    assert ids["boots"] == {"present": set(), "missing": set()}


def test_strongest_detection_wins_and_tie_counts_as_missing():
    ids = {"present": {0}, "missing": {1}}
    assert decide("hard_hat", [(0, 0.8), (1, 0.3)], ids, 0.0).choice == "yes"
    assert decide("hard_hat", [(0, 0.3), (1, 0.8)], ids, 0.0).choice == "no"
    assert decide("hard_hat", [(0, 0.5), (1, 0.5)], ids, 0.0).choice == "no"


def test_nothing_detected_is_unclear_not_no():
    a = decide("hard_hat", [(4, 0.9)], {"present": {0}, "missing": {1}}, 0.0)
    assert a.choice == "unclear" and a.confidence == 0.0


def test_checker_answers_from_model_detections():
    checker = LocalPpeChecker(FakeYolo([(0, 0.91), (4, 0.95)]), DEFAULT_LOCAL_CLASSES)
    a = checker.check("hard_hat", JPEG)
    assert a.choice == "yes" and abs(a.confidence - 0.91) < 1e-6 and a.latency_s >= 0


def test_item_without_a_model_class_is_always_unclear():
    checker = LocalPpeChecker(FakeYolo([(0, 0.9)]), DEFAULT_LOCAL_CLASSES)
    assert checker.unmapped_items() == ["boots"]
    assert checker.check("boots", JPEG).choice == "unclear"
    # An item left out of local_ml.classes altogether is reported too
    partial = LocalPpeChecker(FakeYolo([]), {"hard_hat": DEFAULT_LOCAL_CLASSES["hard_hat"]})
    assert partial.unmapped_items() == ["vest", "boots"]


def test_model_crash_is_an_error_answer():
    class Broken(FakeYolo):
        def predict(self, img, **kwargs):
            raise RuntimeError("boom")

    a = LocalPpeChecker(Broken([]), DEFAULT_LOCAL_CLASSES).check("vest", JPEG)
    assert a.choice == "error" and "boom" in a.error


def test_local_checker_plugs_into_the_inspector_and_fails_closed():
    checker = LocalPpeChecker(FakeYolo([(0, 0.9), (2, 0.9)]), DEFAULT_LOCAL_CLASSES)
    insp = AsyncInspector(checker, ("hard_hat", "vest", "boots"), 0.5, timeout_s=2.0)
    insp.submit(1, {i: JPEG for i in ("hard_hat", "vest", "boots")})
    end = time.monotonic() + 3
    while not (done := insp.poll()) and time.monotonic() < end:
        time.sleep(0.01)
    [result] = done
    assert result.answers["hard_hat"].choice == "yes" and result.answers["boots"].choice == "unclear"
    assert result.decision == RETRY   # no boots class -> never unlocks
    insp.close()
