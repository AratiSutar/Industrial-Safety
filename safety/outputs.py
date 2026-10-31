"""Send alert levels to lights, projectors, speakers, radios and the operator screen.

Every output is an adapter with on()/off(). For now all of them are simulated (shown on screen
and in the terminal). Real hardware (USB relay, PLC, MQTT, speaker) replaces one adapter at a time
without touching the alert rules.
"""

from typing import Protocol

from safety.alerts import Level, MachineChange
from safety.config import OUTPUT_TYPES, Machine


class OutputAdapter(Protocol):
    def on(self, machine: Machine, level: Level, message: str) -> None: ...
    def off(self, machine: Machine) -> None: ...


class SimulatedOutput:
    """Remembers what is on, so the preview and tests can show it."""

    def __init__(self, name: str):
        self.name = name
        self.active: dict[str, str] = {}  # machine id -> message

    def on(self, machine: Machine, level: Level, message: str) -> None:
        self.active[machine.id] = message

    def off(self, machine: Machine) -> None:
        self.active.pop(machine.id, None)


class OutputRouter:
    def __init__(self, adapters: dict[str, OutputAdapter] | None = None):
        self.adapters: dict[str, OutputAdapter] = {name: SimulatedOutput(name) for name in OUTPUT_TYPES}
        self.adapters.update(adapters or {})
        self.active: dict[str, tuple[str, ...]] = {}  # machine id -> outputs currently on

    def apply(self, change: MachineChange) -> str:
        m = change.machine
        wanted = m.outputs.get(change.new.name, ()) if change.new > Level.SAFE else ()
        for name in self.active.get(m.id, ()):
            if name not in wanted:
                self.adapters[name].off(m)
        for name in wanted:
            self.adapters[name].on(m, change.new, change.message)
        self.active[m.id] = tuple(wanted)
        if not wanted:
            self.active.pop(m.id, None)

        arrow = "up" if change.new > change.old else "down" if change.new < change.old else "update"
        outs = ", ".join(wanted) if wanted else "all off"
        return f"[{change.new.name:5}] {m.name} ({arrow} from {change.old.name}) -> {outs} | \"{change.message}\""
