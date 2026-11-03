"""Score PPE answers against human labels, to compare inspection engines (tools/compare_engines.py).

"Positive" means the PPE item is present. A false positive is the dangerous error: the item is
missing but the engine says it is there, which could unlock the gate.
Answers are scored exactly as the gate uses them: "unclear", "error" and anything below the
engine's min_confidence all count as unclear.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safety.inspector import DENY, RETRY, UNLOCK
from safety.jev_client import PpeAnswer

LABELS = ("yes", "no", "unclear")
OUTCOMES = ("yes", "no", "unclear")
LABEL_COLUMNS = {"hat": "hard_hat", "hard_hat": "hard_hat", "vest": "vest", "boots": "boots"}


@dataclass
class Sample:
    """One engine's answer about one item of one labelled person."""
    image: str
    item: str
    label: str         # yes | no | unclear (a person couldn't tell either)
    answer: PpeAnswer
    called: bool       # False when the body part wasn't visible and the engine was never asked


def load_labels(path: Path) -> list[tuple[str, dict[str, str]]]:
    """Rows of labels.csv: image, hat, vest, boots with yes / no / unclear (empty = not labelled)."""
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = {c: c.strip().lower() for c in reader.fieldnames or ()}
        if "image" not in columns.values():
            raise ValueError(f"{path}: needs an 'image' column (got {list(columns.values())})")
        for line, raw in enumerate(reader, start=2):
            row = {columns[k]: (v or "").strip() for k, v in raw.items() if k is not None}
            if not row.get("image"):
                continue
            labels = {}
            for col, item in LABEL_COLUMNS.items():
                value = row.get(col, "").lower()
                if not value:
                    continue
                if value not in LABELS:
                    raise ValueError(f"{path} line {line}: {col}={value!r}, expected one of {LABELS}")
                labels[item] = value
            rows.append((row["image"], labels))
    return rows


def outcome(answer: PpeAnswer, min_conf: float) -> str:
    """What the gate makes of an answer: a confident yes / no, otherwise unclear."""
    return answer.choice if answer.choice in ("yes", "no") and answer.confidence >= min_conf else "unclear"


def ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def item_metrics(samples: list[Sample], min_conf: float) -> dict:
    scored = [(s, outcome(s.answer, min_conf)) for s in samples if s.label in ("yes", "no")]
    present = [o for s, o in scored if s.label == "yes"]
    missing = [o for s, o in scored if s.label == "no"]
    correct = sum(o == s.label for s, o in scored)
    answered = sum(o != "unclear" for _, o in scored)
    return {
        "labelled": len(scored),
        "labelled_present": len(present),
        "labelled_missing": len(missing),
        "accuracy": ratio(correct, len(scored)),                   # unclear counts as wrong
        "accuracy_when_answered": ratio(correct, answered),        # ignores unclear answers
        "false_positive_rate": ratio(missing.count("yes"), len(missing)),
        "false_negative_rate": ratio(present.count("no"), len(present)),
        "unclear_rate": ratio(len(scored) - answered, len(scored)),
        "errors": sum(s.answer.choice == "error" for s, _ in scored),
        "confusion": {label: {o: sum(s.label == label and out == o for s, out in scored) for o in OUTCOMES}
                      for label in ("yes", "no")},
        # Where a person couldn't tell either: ideally the engine says unclear too
        "human_unclear": {o: sum(s.label == "unclear" and outcome(s.answer, min_conf) == o for s in samples)
                          for o in OUTCOMES},
    }


def expected_gate(labels: dict[str, str], required: tuple[str, ...]) -> str | None:
    """The right gate decision for a labelled person, or None when a person couldn't tell."""
    if any(labels.get(i) == "no" for i in required):
        return DENY
    if all(labels.get(i) == "yes" for i in required):
        return UNLOCK
    return None


def gate_metrics(results: list[tuple[str | None, str]]) -> dict:
    """(expected, actual) gate decisions per person. A false unlock lets someone in without PPE."""
    scored = [(e, d) for e, d in results if e is not None]
    should_deny = sum(e == DENY for e, _ in scored)
    should_unlock = sum(e == UNLOCK for e, _ in scored)
    false_unlocks = sum(e == DENY and d == UNLOCK for e, d in scored)
    false_denies = sum(e == UNLOCK and d == DENY for e, d in scored)
    return {
        "people": len(scored),
        "people_missing_ppe": should_deny,
        "accuracy": ratio(sum(e == d for e, d in scored), len(scored)),
        "false_unlocks": false_unlocks,
        "false_unlock_rate": ratio(false_unlocks, should_deny),
        "false_denies": false_denies,
        "false_deny_rate": ratio(false_denies, should_unlock),
        "retry_rate": ratio(sum(d == RETRY for _, d in scored), len(scored)),
    }


def latency_stats(seconds: list[float]) -> dict:
    if not seconds:
        return {"count": 0}
    ms = np.array(seconds) * 1000
    return {
        "count": len(seconds),
        "mean_ms": round(float(ms.mean()), 1),
        "p50_ms": round(float(np.percentile(ms, 50)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "max_ms": round(float(ms.max()), 1),
    }


def token_stats(calls: list[Sample]) -> dict:
    """Billable input tokens (Jev only), per call and per worker check (all items of one person)."""
    counted = [s for s in calls if s.answer.input_tokens is not None]
    if not counted:
        return {"calls_counted": 0}
    per_person: dict[str, int] = {}
    for s in counted:
        per_person[s.image] = per_person.get(s.image, 0) + s.answer.input_tokens
    return {
        "calls_counted": len(counted),
        "mean_input_per_call": round(sum(s.answer.input_tokens for s in counted) / len(counted), 1),
        "mean_input_per_worker_check": round(sum(per_person.values()) / len(per_person), 1),
    }


def usd_per_check(input_tokens_per_check: float, usd_per_million_tokens: float) -> float:
    return input_tokens_per_check * usd_per_million_tokens / 1_000_000


def break_even_checks(local_usd_per_day: float, jev_usd_per_check: float) -> float | None:
    """Worker checks per day above which the local model costs less than Jev.

    0 = the local model is cheaper at any volume; None = Jev costs nothing, so it never is.
    """
    if jev_usd_per_check <= 0:
        return None
    return max(local_usd_per_day, 0.0) / jev_usd_per_check


def engine_report(samples: list[Sample], gates: list[tuple[str | None, str]], min_conf: float,
                  items: tuple[str, ...]) -> dict:
    calls = [s for s in samples if s.called]
    per_person: dict[str, float] = {}
    for s in calls:
        per_person[s.image] = per_person.get(s.image, 0.0) + s.answer.latency_s
    return {
        "min_confidence": min_conf,
        "overall": item_metrics(samples, min_conf),
        "per_item": {item: item_metrics([s for s in samples if s.item == item], min_conf) for item in items},
        "gate": gate_metrics(gates),
        "latency": {
            "per_call": latency_stats([s.answer.latency_s for s in calls]),
            "per_person_sequential": latency_stats(list(per_person.values())),
        },
        "tokens": token_stats(calls),
    }
