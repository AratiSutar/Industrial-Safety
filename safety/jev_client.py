"""Block 3: ask Jev one yes/no/unclear question per PPE item.

All TypeSafe-specific code lives here, so a model version change touches only this file.
"""

import threading
import time
from dataclasses import dataclass
from typing import Protocol

CHOICES = ("yes", "no", "unclear", "error")

QUESTIONS = {
    "hard_hat": "Is the person in the image wearing a hard hat (safety helmet) on their head?",
    "vest": "Is the person in the image wearing a high-visibility safety vest?",
    "boots": "Is the person in the image wearing safety boots (sturdy work boots, not trainers or sandals)?",
}

CRITERIA = {
    "yes": "The item is clearly visible and worn correctly.",
    "no": "The item is clearly missing or not worn.",
    "unclear": "The image is too blurry, dark, small, cut off or blocked to tell.",
}

# Images and Box are still "danger zone" features in Jev 1.14 and need this header
DANGER_ZONE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}


@dataclass(frozen=True)
class PpeAnswer:
    item: str
    choice: str          # yes | no | unclear | error
    confidence: float
    latency_s: float = 0.0
    error: str | None = None


class PpeChecker(Protocol):
    def check(self, item: str, jpeg: bytes) -> PpeAnswer: ...


class JevPpeClient:
    """Real client. One TypeSafeClient per worker thread (the SDK makes no thread-safety promise)."""

    def __init__(self, api_key: str, model: str, timeout_s: float):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self._local = threading.local()

    def _client(self):
        if not hasattr(self._local, "client"):
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            self._local.client = TypeSafeClient(
                api_key=self.api_key,
                model=self.model,
                timeout=self.timeout_s,
                retry=RetryPolicy(max_retries=1, timeout=self.timeout_s),
                headers=DANGER_ZONE_HEADERS,
            )
        return self._local.client

    def check(self, item: str, jpeg: bytes) -> PpeAnswer:
        from typesafe_sdk import Choice, Image, TypeSafeError

        start = time.perf_counter()
        try:
            result = self._client().system_one(
                {"worker": Image(jpeg)},
                {item: Choice(instructions=QUESTIONS[item], criteria=CRITERIA)},
            )
            answer = result.choices[item]
            return PpeAnswer(item, answer.choice, float(answer.confidence), time.perf_counter() - start)
        except (TypeSafeError, KeyError) as e:
            # Any failure counts as "error", which keeps the gate locked
            return PpeAnswer(item, "error", 0.0, time.perf_counter() - start, f"{type(e).__name__}: {e}")


class FakePpeClient:
    """Offline stand-in for demos and tests. Always answers "unclear", so it can never unlock a gate."""

    def __init__(self, latency_s: float = 0.4):
        self.latency_s = latency_s

    def check(self, item: str, jpeg: bytes) -> PpeAnswer:
        time.sleep(self.latency_s)
        return PpeAnswer(item, "unclear", 0.0, self.latency_s)
