"""Pick the PPE inspection engine (inspection.engine in settings.yaml).

Every engine answers check(item, jpeg) -> PpeAnswer, so AsyncInspector, the fail-closed gate
decision and the display work the same whichever one is chosen.
"""

from dataclasses import dataclass

from safety.config import ENGINES, Settings
from safety.jev_client import FakePpeClient, JevPpeClient, PpeChecker


class EngineUnavailable(RuntimeError):
    pass


@dataclass
class Engine:
    name: str              # jev_vlm | local_ml
    checker: PpeChecker
    min_confidence: float  # below this an answer counts as unclear; each engine's scores run differently
    label: str             # for log lines and reports


def make_engine(name: str, settings: Settings, fake_jev: bool = False) -> Engine:
    if name == "jev_vlm":
        if fake_jev:
            return Engine(name, FakePpeClient(), settings.jev.min_confidence,
                          "Jev FAKE client (offline test, always 'unclear')")
        if not settings.typesafe_api_key:
            raise EngineUnavailable("TYPESAFE_API_KEY not set; use --fake-jev to test offline")
        return Engine(name, JevPpeClient(settings.typesafe_api_key, settings.jev.model, settings.jev.timeout_s),
                      settings.jev.min_confidence, f"Jev {settings.jev.model}")
    if name == "local_ml":
        from safety.local_ppe import LocalPpeChecker

        lm = settings.local_ml
        try:
            checker = LocalPpeChecker.load(lm.weights, lm.classes, lm.imgsz, settings.device, settings.backend,
                                           lm.min_detection_conf)
        except FileNotFoundError as e:
            raise EngineUnavailable(str(e)) from e
        label = f"local YOLO {lm.weights}"
        if unmapped := checker.unmapped_items():
            label += f" (no class for {', '.join(unmapped)}: always 'unclear')"
        return Engine(name, checker, lm.min_confidence, label)
    raise ValueError(f"unknown engine {name!r}; allowed {ENGINES}")
