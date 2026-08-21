"""Criterios de error y detención."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Metrics:
    absolute: float
    relative: float
    residual: float

def metrics(previous: float | None, current: float, fx: float) -> Metrics:
    absolute = math.inf if previous is None else abs(current - previous)
    relative = math.inf if previous is None or current == 0 else absolute / abs(current)
    return Metrics(absolute, relative, abs(fx))

def should_stop(m: Metrics, tolerance: float, criteria: set[str], mode: str = "alguno",
                iteration: int | None = None, max_iterations: int | None = None) -> tuple[bool, str]:
    checks = {"Error absoluto": m.absolute <= tolerance,
              "Error relativo": m.relative <= tolerance,
              "Residuo": m.residual <= tolerance,
              "Iteraciones": iteration is not None and max_iterations is not None and iteration >= max_iterations}
    selected = [(name, checks[name]) for name in criteria]
    stop = all(v for _, v in selected) if mode == "todos" else any(v for _, v in selected)
    reason = (" y " if mode == "todos" else " o ").join(name.lower() for name, ok in selected if ok)
    return stop, f"Criterio alcanzado: {reason}." if stop else ""
