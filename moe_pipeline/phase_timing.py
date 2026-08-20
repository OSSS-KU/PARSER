"""Lightweight wall-clock phase timing for the compression pipeline.

The pipeline runs across two separate processes — ``save_blocks.py``
(calibration) and ``compress_blocks.py`` (backbone generation + compression) —
so timing cannot be aggregated in memory. Instead, each process accumulates
into a process-global :class:`PhaseTimer` and writes a JSON sidecar.

Two ways to record a phase:
  - ``with timer.timed(name): ...`` for a single contiguous block.
  - ``timer.add(name, dt)`` for a phase entered repeatedly across a loop
    (e.g. backbone generation, called once per layer).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


class PhaseTimer:
    def __init__(self) -> None:
        self._elapsed: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    @contextmanager
    def timed(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        self._elapsed[name] = self._elapsed.get(name, 0.0) + float(seconds)
        self._counts[name] = self._counts.get(name, 0) + 1

    def reset(self) -> None:
        self._elapsed.clear()
        self._counts.clear()

    def as_dict(self) -> dict:
        return {
            "phases": {
                name: {
                    "seconds": round(self._elapsed[name], 4),
                    "calls": self._counts[name],
                }
                for name in sorted(self._elapsed)
            },
            "total_seconds": round(sum(self._elapsed.values()), 4),
        }

    def write(self, path, extra: Optional[dict] = None, merge: bool = False) -> None:
        """Write timing as JSON.

        With ``merge=True``, phases from an existing file at ``path`` are kept
        unless this timer redefines them. This lets multiple processes that
        share one sidecar (e.g. activation calibrations that land in
        ``blocks/calibration_timing.json``) accumulate their phases instead of
        clobbering each other.
        """
        payload = self.as_dict()
        path = Path(path)
        if merge and path.exists():
            try:
                old = json.loads(path.read_text())
                old_phases = old.get("phases", {})
            except Exception:
                old_phases = {}
            if isinstance(old_phases, dict):
                merged = {**old_phases, **payload["phases"]}
                payload["phases"] = merged
                payload["total_seconds"] = round(
                    sum(
                        p.get("seconds", 0.0)
                        for p in merged.values()
                        if isinstance(p, dict)
                    ),
                    4,
                )
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))


_GLOBAL_TIMER = PhaseTimer()


def global_timer() -> PhaseTimer:
    """Return the process-global timer shared across pipeline modules."""
    return _GLOBAL_TIMER
