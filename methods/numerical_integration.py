"""Reglas cerradas de Newton-Cotes y punto medio para integración numérica."""
from dataclasses import dataclass

import mpmath as mp

from core.parser import safe_float


METHODS = {
    "midpoint": ("Rectángulo por punto medio", 0, 2),
    "left_rectangle": ("Rectángulo izquierdo", 0, 1),
    "right_rectangle": ("Rectángulo derecho", 0, 1),
    "trapezoid": ("Regla del trapecio", 1, 2),
    "simpson_13": ("Regla de Simpson 1/3", 2, 4),
    "simpson_38": ("Regla de Simpson 3/8", 3, 4),
}


@dataclass(frozen=True)
class IntegrationPoint:
    x: mp.mpf
    fx: mp.mpf
    weight: int


@dataclass(frozen=True)
class IntegrationResult:
    method: str
    method_key: str
    approximation: mp.mpf
    a: mp.mpf
    b: mp.mpf
    n: int
    h: mp.mpf
    degree: int
    error_order: int
    points: tuple[IntegrationPoint, ...]
    boundaries: tuple[mp.mpf, ...]


def _finite_number(value, label: str) -> mp.mpf:
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError):
        raise ValueError(f"El límite {label} debe ser un número real.") from None
    if not mp.isfinite(number):
        raise ValueError(f"El límite {label} debe ser finito.")
    return number


def _positive_integer(value) -> int:
    if isinstance(value, bool):
        raise ValueError("n debe ser un número entero positivo.")
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError):
        raise ValueError("n debe ser un número entero positivo.") from None
    if not mp.isfinite(number) or number <= 0 or number != mp.floor(number):
        raise ValueError("n debe ser un número entero positivo.")
    return int(number)


def integrate_newton_cotes(function: callable, a, b, n, method: str) -> IntegrationResult:
    """Aproxima una integral definida y conserva los puntos y pesos utilizados."""
    if method not in METHODS:
        raise ValueError("Elegí un método de integración válido.")
    a_value, b_value, count = _finite_number(a, "inferior a"), _finite_number(b, "superior b"), _positive_integer(n)
    if method == "simpson_13" and count % 2:
        raise ValueError("Para Simpson 1/3, n debe ser par (n=2 corresponde a la fórmula simple).")
    if method == "simpson_38" and count % 3:
        raise ValueError("Para Simpson 3/8, n debe ser múltiplo de 3 (n=3 corresponde a la fórmula simple).")

    h = (b_value - a_value) / count
    boundaries = tuple(a_value + index * h for index in range(count + 1))
    if method == "midpoint":
        xs = tuple(a_value + (index + mp.mpf("0.5")) * h for index in range(count))
        weights = (1,) * count
        factor = h
    elif method == "left_rectangle":
        xs = boundaries[:-1]
        weights = (1,) * count
        factor = h
    elif method == "right_rectangle":
        xs = boundaries[1:]
        weights = (1,) * count
        factor = h
    else:
        xs = boundaries
        if method == "trapezoid":
            weights, factor = (1,) + (2,) * (count - 1) + (1,), h / 2
        elif method == "simpson_13":
            weights = tuple(1 if index in (0, count) else (4 if index % 2 else 2)
                            for index in range(count + 1))
            factor = h / 3
        else:
            weights = tuple(1 if index in (0, count) else (2 if index % 3 == 0 else 3)
                            for index in range(count + 1))
            factor = 3 * h / 8

    points = tuple(IntegrationPoint(x, safe_float(function, x), weight)
                   for x, weight in zip(xs, weights))
    approximation = factor * mp.fsum(point.weight * point.fx for point in points)
    name, degree, order = METHODS[method]
    return IntegrationResult(name, method, approximation, a_value, b_value, count, h,
                             degree, order, points, boundaries)


def integrate_gauss_legendre(function: callable, a, b, nodes: int = 32) -> mp.mpf:
    """Calcula una referencia mediante Gauss–Legendre sobre un intervalo finito."""
    a_value = _finite_number(a, "inferior a")
    b_value = _finite_number(b, "superior b")
    count = _positive_integer(nodes)
    gauss_nodes, gauss_weights = mp.gauss_quadrature(count, "legendre")
    midpoint = (a_value + b_value) / 2
    half_width = (b_value - a_value) / 2
    return half_width * mp.fsum(
        gauss_weights[index] * safe_float(function, midpoint + half_width * gauss_nodes[index])
        for index in range(count)
    )
