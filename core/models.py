"""Estructuras de resultados independientes de la interfaz."""
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass
class IterationResult:
    iteration: int
    x_previous: float | None = None
    x_current: float | None = None
    x_next: float | None = None
    fx: float | None = None
    absolute_error: float | None = None
    relative_error: float | None = None
    residual: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        metadata = row.pop("metadata")
        row.update(metadata)
        return row

@dataclass
class MethodResult:
    method: str
    converged: bool
    root: float | None
    iterations: int
    stop_reason: str
    history: list[IterationResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

