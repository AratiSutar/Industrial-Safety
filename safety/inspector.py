"""Run Jev PPE checks in background threads so the video loop never waits for the network."""

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from safety.jev_client import PpeAnswer, PpeChecker

UNLOCK = "UNLOCK_GATE"
DENY = "DENY_MISSING_PPE"
RETRY = "RETRY"  # unclear / error / low confidence: gate stays locked, ask the worker to step closer


def gate_decision(answers: dict[str, PpeAnswer], required: tuple[str, ...], min_conf: float) -> str:
    """Fail-closed: unlock only when every required item is a confident "yes"."""
    if any(a.choice == "no" and a.confidence >= min_conf for i, a in answers.items() if i in required):
        return DENY
    if all(i in answers and answers[i].choice == "yes" and answers[i].confidence >= min_conf for i in required):
        return UNLOCK
    return RETRY


@dataclass
class Inspection:
    track_id: int | None
    answers: dict[str, PpeAnswer]
    decision: str
    latency_s: float


@dataclass
class _Pending:
    track_id: int | None
    started: float
    futures: dict[str, Future] = field(default_factory=dict)
    skipped: dict[str, PpeAnswer] = field(default_factory=dict)


class AsyncInspector:
    """Non-blocking: submit() returns immediately, poll() collects finished inspections."""

    def __init__(self, checker: PpeChecker, required: tuple[str, ...], min_conf: float, timeout_s: float,
                 max_workers: int = 3, cooldown_s: float = 10.0, max_pending: int = 4,
                 clock: Callable[[], float] = time.monotonic):
        self.checker = checker
        self.required = required
        self.min_conf = min_conf
        self.deadline_s = timeout_s * 2 + 1.0  # SDK timeout + one retry + margin
        self.cooldown_s = cooldown_s
        self.max_pending = max_pending
        self.clock = clock
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="jev")
        self._pending: dict[int | None, _Pending] = {}
        self._last_submit: dict[int | None, float] = {}

    def can_submit(self, track_id: int | None) -> bool:
        if track_id in self._pending or len(self._pending) >= self.max_pending:
            return False
        last = self._last_submit.get(track_id)
        return last is None or self.clock() - last >= self.cooldown_s

    def submit(self, track_id: int | None, crops: dict[str, bytes | None]) -> bool:
        if not self.can_submit(track_id):
            return False
        now = self.clock()
        job = _Pending(track_id, now)
        for item in self.required:
            jpeg = crops.get(item)
            if jpeg is None:
                job.skipped[item] = PpeAnswer(item, "unclear", 0.0, error="body part not visible")
            else:
                job.futures[item] = self._pool.submit(self.checker.check, item, jpeg)
        self._pending[track_id] = job
        self._last_submit[track_id] = now
        return True

    def poll(self) -> list[Inspection]:
        """Finished (or timed-out) inspections. Never blocks."""
        done = []
        now = self.clock()
        for track_id, job in list(self._pending.items()):
            timed_out = now - job.started > self.deadline_s
            if not timed_out and not all(f.done() for f in job.futures.values()):
                continue
            answers = dict(job.skipped)
            for item, fut in job.futures.items():
                if fut.done() and fut.exception() is None:
                    answers[item] = fut.result()
                else:
                    fut.cancel()
                    reason = "timeout" if not fut.done() else f"crashed: {fut.exception()!r}"
                    answers[item] = PpeAnswer(item, "error", 0.0, error=reason)
            decision = gate_decision(answers, self.required, self.min_conf)
            done.append(Inspection(track_id, answers, decision, now - job.started))
            del self._pending[track_id]
        return done

    @property
    def pending(self) -> int:
        return len(self._pending)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
