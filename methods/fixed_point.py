"""Iteración de punto fijo."""
from core.models import IterationResult, MethodResult
from core.errors import metrics, should_stop
from core.parser import safe_float
import mpmath as mp

def fixed_point(f, g, x0: float, tolerance: float = 1e-8, max_iterations: int = 100,
                criteria: set[str] | None = None, mode: str = "alguno") -> MethodResult:
    criteria = criteria or {"Error absoluto", "Residuo"}; history=[]; x=mp.mpf(x0); tolerance=mp.mpf(tolerance); rising=0
    seen: list[float] = []
    for n in range(1, max_iterations + 1):
        x_next = safe_float(g, x); fx = safe_float(f, x_next); m = metrics(x, x_next, fx)
        history.append(IterationResult(n, x, x, x_next, fx, m.absolute, m.relative, m.residual))
        stop, reason = should_stop(m, tolerance, criteria, mode, n, max_iterations)
        if stop:
            converged=any(label in reason for label in ("error absoluto","error relativo","residuo"))
            return MethodResult("Punto Fijo",converged,x_next,n,reason,history)
        if abs(x_next) > 1e12:
            return MethodResult("Punto Fijo", False, x_next, n, "Divergencia aparente.", history, ["Las aproximaciones crecieron excesivamente."])
        if any(abs(x_next-v) <= tolerance/10 for v in seen[-6:-1]):
            return MethodResult("Punto Fijo", False, x_next, n, "Se detectó un ciclo corto.", history, ["La sucesión repite valores sin alcanzar el criterio."])
        if len(history)>2 and history[-1].absolute_error > history[-2].absolute_error: rising += 1
        else: rising=0
        if rising >= 8:
            return MethodResult("Punto Fijo", False, x_next, n, "Divergencia aparente.", history, ["El error aumentó durante varias iteraciones."])
        seen.append(x); x=x_next
    return MethodResult("Punto Fijo", False, x, max_iterations, "Se alcanzó el máximo de iteraciones.", history)
