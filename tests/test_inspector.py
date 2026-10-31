import threading
import time

import pytest

from safety.inspector import DENY, RETRY, UNLOCK, AsyncInspector, gate_decision
from safety.jev_client import FakePpeClient, PpeAnswer

ITEMS = ("hard_hat", "vest", "boots")
JPEG = b"fake-jpeg"


def answers(**choices):
    return {item: PpeAnswer(item, c, 0.9) for item, c in choices.items()}


@pytest.mark.parametrize("given, expected", [
    (dict(hard_hat="yes", vest="yes", boots="yes"), UNLOCK),
    (dict(hard_hat="no", vest="yes", boots="yes"), DENY),
    (dict(hard_hat="yes", vest="unclear", boots="yes"), RETRY),
    (dict(hard_hat="yes", vest="error", boots="yes"), RETRY),
    (dict(hard_hat="yes", vest="yes"), RETRY),                    # boots missing -> never unlock
    (dict(hard_hat="no", vest="unclear", boots="error"), DENY),
])
def test_gate_decision(given, expected):
    assert gate_decision(answers(**given), ITEMS, 0.7) == expected


def test_low_confidence_yes_does_not_unlock():
    low = {i: PpeAnswer(i, "yes", 0.5) for i in ITEMS}
    assert gate_decision(low, ITEMS, 0.7) == RETRY


class ScriptedChecker:
    def __init__(self, choice="yes", delay=0.3):
        self.choice, self.delay = choice, delay
        self.threads = set()

    def check(self, item, jpeg):
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        return PpeAnswer(item, self.choice, 0.95, self.delay)


def wait_for(inspector, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if done := inspector.poll():
            return done
        time.sleep(0.01)
    raise AssertionError("inspection did not finish")


def test_submit_does_not_block_and_items_run_in_parallel():
    checker = ScriptedChecker(delay=0.3)
    insp = AsyncInspector(checker, ITEMS, 0.7, timeout_s=2.0)
    t0 = time.perf_counter()
    assert insp.submit(7, {i: JPEG for i in ITEMS})
    assert time.perf_counter() - t0 < 0.05          # returned immediately
    assert insp.poll() == []                         # not finished yet, and poll didn't block
    [result] = wait_for(insp)
    assert result.track_id == 7 and result.decision == UNLOCK
    assert result.latency_s < 0.6                    # 3 x 0.3 s ran in parallel, not 0.9 s
    assert len(checker.threads) == 3
    insp.close()


def test_missing_crop_counts_as_unclear():
    insp = AsyncInspector(ScriptedChecker(delay=0.0), ITEMS, 0.7, timeout_s=2.0)
    insp.submit(1, {"hard_hat": JPEG, "vest": JPEG, "boots": None})
    [result] = wait_for(insp)
    assert result.answers["boots"].choice == "unclear" and result.decision == RETRY
    insp.close()


def test_slow_api_times_out_and_keeps_gate_locked():
    now = [0.0]
    insp = AsyncInspector(ScriptedChecker(delay=1.0), ITEMS, 0.7, timeout_s=0.1, clock=lambda: now[0])
    insp.submit(1, {i: JPEG for i in ITEMS})
    now[0] = 5.0  # well past the deadline
    [result] = insp.poll()
    assert all(a.choice == "error" for a in result.answers.values())
    assert result.decision == RETRY
    insp.close()


def test_cooldown_and_one_pending_per_person():
    now = [0.0]
    insp = AsyncInspector(ScriptedChecker(delay=0.0), ITEMS, 0.7, timeout_s=2.0, cooldown_s=10, clock=lambda: now[0])
    assert insp.submit(1, {i: JPEG for i in ITEMS})
    assert not insp.can_submit(1)                   # still pending
    time.sleep(0.05)
    insp.poll()
    now[0] = 5.0
    assert not insp.can_submit(1)                   # in cooldown
    assert insp.can_submit(2)                       # other people are not affected
    now[0] = 10.0
    assert insp.can_submit(1)
    insp.close()


def test_fake_client_never_unlocks():
    insp = AsyncInspector(FakePpeClient(latency_s=0.0), ITEMS, 0.7, timeout_s=2.0)
    insp.submit(1, {i: JPEG for i in ITEMS})
    [result] = wait_for(insp)
    assert result.decision == RETRY
    insp.close()
