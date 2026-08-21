"""Herramientas experimentales de convergencia."""
import numpy as np
import mpmath as mp

def observed_order(errors: list[float]) -> float | None:
    """Estima p aceptando tanto floats de NumPy como números mpmath.mpf."""
    threshold = mp.power(10, -(mp.mp.dps - 5))
    clean: list[mp.mpf] = []
    for error in errors:
        try:
            value = mp.mpf(error)
        except (TypeError, ValueError):
            continue
        if mp.isfinite(value) and value > threshold:
            clean.append(value)
    if len(clean) < 4:
        return None
    e0, e1, e2 = clean[-3:]
    denominator = mp.log(e1 / e0)
    if abs(denominator) < threshold:
        return None
    p = mp.log(e2 / e1) / denominator
    return float(p) if mp.isfinite(p) and 0 < p < 5 else None

def contraction_analysis(g, dg, a: float, b: float, samples: int = 600) -> dict:
    xs = np.linspace(a, b, samples)
    with np.errstate(all="ignore"):
        ys, derivatives = np.asarray(g(xs), dtype=float), np.asarray(dg(xs), dtype=float)
    finite = np.isfinite(ys) & np.isfinite(derivatives)
    if not finite.any():
        return {"valid": False, "maps_into_itself": False, "L": None}
    ys, derivatives = ys[finite], derivatives[finite]
    return {"valid": bool(finite.all()), "maps_into_itself": bool(np.all((ys >= a) & (ys <= b))),
            "L": float(np.max(np.abs(derivatives)))}
