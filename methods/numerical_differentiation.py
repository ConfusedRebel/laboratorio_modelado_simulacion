"""Esquemas elementales de derivación numérica con aritmética exacta."""
from dataclasses import dataclass
from typing import Sequence

import sympy as sp

X = sp.Symbol("x", real=True)
SCHEMES = {
    "forward": ("Diferencia hacia adelante", 1),
    "backward": ("Diferencia hacia atrás", 1),
    "centered": ("Diferencia centrada", 2),
}


def _number(value: int | float | str | sp.Expr) -> sp.Expr:
    """Convierte enteros, decimales y fracciones sin perder exactitud."""
    if isinstance(value, sp.Expr):
        result = value
    else:
        try:
            result = sp.Rational(str(value).strip())
        except (TypeError, ValueError, ZeroDivisionError):
            raise ValueError(f"«{value}» no es un número válido. Usá un entero, decimal o fracción como 1/3.") from None
    if result.is_real is False or not result.is_finite:
        raise ValueError("Todos los datos deben ser números reales y finitos.")
    return result


@dataclass(frozen=True)
class DifferentiationValue:
    x: sp.Expr
    y: sp.Expr
    role: str


@dataclass(frozen=True)
class DifferentiationResult:
    scheme: str
    point: sp.Expr
    h: sp.Expr
    approximation: sp.Expr
    order: int
    values: tuple[DifferentiationValue, ...]
    substituted_formula: sp.Expr
    exact_derivative: sp.Expr | None = None
    signed_error: sp.Expr | None = None
    absolute_error: sp.Expr | None = None
    relative_error: sp.Expr | None = None
    expression: sp.Expr | None = None


@dataclass(frozen=True)
class SecondDerivativeResult:
    point: sp.Expr
    h: sp.Expr
    approximation: sp.Expr
    exact_derivative: sp.Expr
    absolute_error: sp.Expr


@dataclass(frozen=True)
class KinematicsRow:
    x: sp.Expr
    position: sp.Expr
    first_derivative: sp.Expr
    second_derivative: sp.Expr


def _finish(scheme: str, point: sp.Expr, h: sp.Expr,
            values: tuple[DifferentiationValue, ...], expression: sp.Expr | None = None) -> DifferentiationResult:
    y = {item.role: item.y for item in values}
    if scheme == "forward":
        approximation = sp.simplify((y["f(x₀+h)"] - y["f(x₀)"]) / h)
    elif scheme == "backward":
        approximation = sp.simplify((y["f(x₀)"] - y["f(x₀−h)"]) / h)
    else:
        approximation = sp.simplify((y["f(x₀+h)"] - y["f(x₀−h)"]) / (2 * h))
    exact = sp.simplify(sp.diff(expression, X).subs(X, point)) if expression is not None else None
    signed = sp.simplify(approximation - exact) if exact is not None else None
    absolute = sp.Abs(signed) if signed is not None else None
    relative = None if exact in (None, 0) else sp.simplify(absolute / sp.Abs(exact))
    return DifferentiationResult(
        SCHEMES[scheme][0], point, h, approximation, SCHEMES[scheme][1], values,
        approximation, exact, signed, absolute, relative, expression,
    )


def differentiate_function(expression: sp.Expr, point, h, scheme: str) -> DifferentiationResult:
    """Aproxima f'(x₀) evaluando una expresión simbólica en dos nodos."""
    if scheme not in SCHEMES:
        raise ValueError("Elegí diferencia hacia adelante, hacia atrás o centrada.")
    point, h = _number(point), _number(h)
    if h <= 0:
        raise ValueError("El paso h debe ser positivo. Probá, por ejemplo, h=1/10.")
    offsets = {"forward": (0, 1), "backward": (-1, 0), "centered": (-1, 1)}[scheme]
    roles = {"forward": ("f(x₀)", "f(x₀+h)"), "backward": ("f(x₀−h)", "f(x₀)"),
             "centered": ("f(x₀−h)", "f(x₀+h)")}[scheme]
    values = []
    for offset, role in zip(offsets, roles):
        x_value = point + offset * h
        y_value = sp.simplify(expression.subs(X, x_value))
        if y_value.is_real is False or y_value.is_finite is False or y_value.has(sp.zoo, sp.nan):
            raise ValueError(f"La función no está definida en x={x_value}. Cambiá x₀, h o el esquema.")
        values.append(DifferentiationValue(x_value, y_value, role))
    return _finish(scheme, point, h, tuple(values), expression)


def differentiate_nodes(x_values: Sequence, y_values: Sequence, point, scheme: str) -> DifferentiationResult:
    """Aproxima la derivada en una tabla igualmente espaciada."""
    if scheme not in SCHEMES:
        raise ValueError("El esquema seleccionado no es válido.")
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Ingresá al menos dos nodos y la misma cantidad de valores x e y.")
    xs, ys = tuple(map(_number, x_values)), tuple(map(_number, y_values))
    if len(set(xs)) != len(xs):
        raise ValueError("Hay nodos x repetidos. Cada x debe aparecer una sola vez.")
    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    xs, ys = tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    steps = [sp.simplify(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    if any(step != steps[0] for step in steps[1:]):
        raise ValueError("Los nodos no están igualmente espaciados. En esta etapa usá una tabla con un mismo paso h.")
    point, h = _number(point), steps[0]
    if point not in xs:
        raise ValueError("El punto elegido debe coincidir exactamente con uno de los nodos de la tabla.")
    lookup = dict(zip(xs, ys))
    required = {"forward": (point, point + h), "backward": (point - h, point),
                "centered": (point - h, point + h)}[scheme]
    missing = [value for value in required if value not in lookup]
    if missing:
        raise ValueError(f"Faltan los nodos {', '.join(map(str, missing))} para aplicar este esquema en x₀={point}.")
    roles = {"forward": ("f(x₀)", "f(x₀+h)"), "backward": ("f(x₀−h)", "f(x₀)"),
             "centered": ("f(x₀−h)", "f(x₀+h)")}[scheme]
    values = tuple(DifferentiationValue(x, lookup[x], role) for x, role in zip(required, roles))
    return _finish(scheme, point, h, values)


def differentiate_second_function(expression: sp.Expr, point, h) -> SecondDerivativeResult:
    """Aproxima f''(x₀) con la diferencia centrada de orden dos."""
    point, h = _number(point), _number(h)
    if h <= 0:
        raise ValueError("El paso h debe ser positivo.")
    samples = [sp.simplify(expression.subs(X, point + offset * h)) for offset in (-1, 0, 1)]
    if any(value.is_real is False or value.is_finite is False or value.has(sp.zoo, sp.nan)
           for value in samples):
        raise ValueError("La función no está definida en todos los puntos necesarios para f''.")
    approximation = sp.simplify((samples[2] - 2 * samples[1] + samples[0]) / h**2)
    exact = sp.simplify(sp.diff(expression, X, 2).subs(X, point))
    return SecondDerivativeResult(point, h, approximation, exact, sp.Abs(approximation - exact))


def differentiate_table_all(x_values: Sequence, y_values: Sequence) -> tuple[KinematicsRow, ...]:
    """Completa primera y segunda derivada en una tabla de nodos equiespaciados."""
    if len(x_values) != len(y_values) or len(x_values) < 3:
        raise ValueError("Para completar velocidad y aceleración se requieren al menos tres pares x,y.")
    xs, ys = tuple(map(_number, x_values)), tuple(map(_number, y_values))
    if len(set(xs)) != len(xs):
        raise ValueError("Los nodos x deben ser distintos.")
    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    xs, ys = tuple(zip(*pairs))
    steps = [sp.simplify(xs[index + 1] - xs[index]) for index in range(len(xs) - 1)]
    if steps[0] <= 0 or any(step != steps[0] for step in steps[1:]):
        raise ValueError("La tabla debe tener un paso constante y positivo.")
    h, last = steps[0], len(xs) - 1
    rows = []
    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        if index == 0:
            first = (ys[1] - ys[0]) / h
            second = (ys[2] - 2 * ys[1] + ys[0]) / h**2
        elif index == last:
            first = (ys[last] - ys[last - 1]) / h
            second = (ys[last] - 2 * ys[last - 1] + ys[last - 2]) / h**2
        else:
            first = (ys[index + 1] - ys[index - 1]) / (2 * h)
            second = (ys[index + 1] - 2 * ys[index] + ys[index - 1]) / h**2
        rows.append(KinematicsRow(x_value, y_value, sp.simplify(first), sp.simplify(second)))
    return tuple(rows)
