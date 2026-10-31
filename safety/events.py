"""Append-only event log: one JSON line per incident. Text only, no images, names or tracker numbers."""

import json
from pathlib import Path


class EventLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, records: list[dict]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
