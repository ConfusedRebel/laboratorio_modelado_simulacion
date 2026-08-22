"""Calculadora simbólica segura para asistir la carga de datos."""
from dataclasses import dataclass

import sympy as sp

from core.parser import parse_function


@dataclass(frozen=True)
class CalculatorResult:
    expression: sp.Expr
    exact: sp.Expr
    decimal: sp.Expr


def calculate(text: str, digits: int = 15) -> CalculatorResult:
    """Interpreta una expresión sin variables usando el parser permitido."""
    try:
        parsed = parse_function(text)
    except ValueError as exc:
        if "función real" in str(exc):
            raise ValueError("El cálculo no produjo un número real finito. Revisá raíces, logaritmos y divisiones.") from None
        raise
    if parsed.expression.has(sp.Symbol("x", real=True)):
        raise ValueError("La calculadora trabaja solo con números y constantes; no ingreses x.")
    exact = sp.simplify(parsed.expression)
    if exact.is_real is False or exact.is_finite is False or exact.has(sp.nan, sp.zoo):
        raise ValueError("El cálculo no produjo un número real finito. Revisá raíces, logaritmos y divisiones.")
    return CalculatorResult(parsed.expression, exact, sp.N(exact, digits))
