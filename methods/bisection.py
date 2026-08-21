"""Método de bisección."""
import math
import mpmath as mp
from core.models import IterationResult, MethodResult
from core.errors import metrics, should_stop
from core.parser import safe_float

def theoretical_iterations(a: float, b: float, tolerance: float) -> int:
    return max(0, math.ceil(math.log2((b - a) / tolerance)))

def bisection(f, a: float, b: float, tolerance: float = 1e-8, max_iterations: int = 100,
              criteria: set[str] | None = None, mode: str = "alguno") -> MethodResult:
    criteria = criteria or {"Error absoluto", "Residuo"}
    a, b, tolerance = mp.mpf(a), mp.mpf(b), mp.mpf(tolerance)
    if not a < b or tolerance <= 0:
        raise ValueError("Se requiere a < b y una tolerancia positiva.")
    fa, fb = safe_float(f, a), safe_float(f, b)
    if fa == 0: return MethodResult("Bisección", True, a, 0, "El extremo a es raíz.")
    if fb == 0: return MethodResult("Bisección", True, b, 0, "El extremo b es raíz.")
    if fa * fb > 0:
        raise ValueError(f"No puede aplicarse Bisección: f(a)={mp.nstr(fa,8)} y f(b)={mp.nstr(fb,8)} tienen el mismo signo; Bolzano no garantiza una raíz.")
    history, previous = [], None
    for n in range(1, max_iterations + 1):
        c, fc = (a + b) / 2, safe_float(f, (a + b) / 2)
        m = metrics(previous, c, fc)
        if fa * fc <= 0: new_a, new_b, chosen = a, c, "[a,c]"
        else: new_a, new_b, chosen = c, b, "[c,b]"
        history.append(IterationResult(n, previous, c, c, fc, m.absolute, m.relative, m.residual,
            {"a": a, "b": b, "c": c, "f(a)": fa, "f(c)": fc, "f(b)": fb,
             "ancho": b-a, "nuevo_intervalo": chosen, "new_a": new_a, "new_b": new_b}))
        stop, reason = should_stop(m, tolerance, criteria, mode, n, max_iterations)
        if stop or fc == 0:
            converged = fc == 0 or any(label in reason for label in ("error absoluto", "error relativo", "residuo"))
            return MethodResult("Bisección", converged, c, n, reason or "Raíz exacta encontrada.", history)
        previous, a, b = c, new_a, new_b
        fa, fb = safe_float(f, a), safe_float(f, b)
    return MethodResult("Bisección", False, history[-1].x_current, max_iterations,
                        "Se alcanzó el máximo de iteraciones.", history)
