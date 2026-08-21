"""Construcción exacta del polinomio interpolante de Lagrange."""
from dataclasses import dataclass
from typing import Sequence

import sympy as sp

X = sp.Symbol("x", real=True)


@dataclass(frozen=True)
class LagrangeBasis:
    """Una base L_i y los índices j que intervienen en su producto."""
    i: int
    j_values: tuple[int, ...]
    expression: sp.Expr
    expanded: sp.Expr
    weighted: sp.Expr


@dataclass(frozen=True)
class LagrangeResult:
    """Resultado simbólico completo de una interpolación."""
    x_values: tuple[sp.Expr, ...]
    y_values: tuple[sp.Expr, ...]
    bases: tuple[LagrangeBasis, ...]
    polynomial: sp.Expr
    degree: int

    def evaluate(self, value: float | str, digits: int = 16) -> sp.Expr:
        """Evalúa el polinomio final conservando precisión simbólica."""
        return sp.N(self.polynomial.subs(X, _number(value)), digits)


def _number(value: float | str | sp.Expr) -> sp.Expr:
    if isinstance(value, sp.Expr):
        return value
    return sp.Rational(str(value))


def build_lagrange(x_values: Sequence[float], y_values: Sequence[float]) -> LagrangeResult:
    """Construye P_n(x)=sum y_i L_i(x) para entre 2 y 20 nodos distintos."""
    if len(x_values) != len(y_values):
        raise ValueError("Las listas de x e y deben tener la misma cantidad de valores.")
    if not 2 <= len(x_values) <= 20:
        raise ValueError("Se requieren entre 2 y 20 nodos.")
    xs = tuple(_number(value) for value in x_values)
    ys = tuple(_number(value) for value in y_values)
    if len(set(xs)) != len(xs):
        raise ValueError("Los valores x_i deben ser distintos; no puede dividirse por x_i-x_j=0.")

    bases: list[LagrangeBasis] = []
    polynomial = sp.Integer(0)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        j_values = tuple(j for j in range(len(xs)) if j != i)
        factors = [(X - xs[j]) / (xi - xs[j]) for j in j_values]
        expression = sp.prod(factors)
        expanded = sp.expand(expression)
        weighted = sp.expand(yi * expression)
        polynomial += weighted
        bases.append(LagrangeBasis(i, j_values, expression, expanded, weighted))

    polynomial = sp.expand(polynomial)
    degree = 0 if polynomial == 0 else int(sp.degree(polynomial, X))
    return LagrangeResult(xs, ys, tuple(bases), polynomial, degree)


def basis_factor(result: LagrangeResult, i: int, j: int) -> sp.Expr:
    """Devuelve el factor individual asociado al par (i,j)."""
    if not 0 <= i < len(result.x_values):
        raise ValueError("El índice i está fuera del rango de nodos.")
    if not 0 <= j < len(result.x_values) or j == i:
        raise ValueError("El índice j debe pertenecer a los nodos y ser distinto de i.")
    return (X - result.x_values[j]) / (result.x_values[i] - result.x_values[j])
