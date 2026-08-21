"""Análisis seguro de expresiones matemáticas mediante SymPy."""
from dataclasses import dataclass
import numpy as np
import sympy as sp
import mpmath as mp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        convert_xor, implicit_multiplication_application)

X = sp.Symbol("x", real=True)
ALLOWED = {"x": X, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
           "sqrt": sp.sqrt, "root": sp.root, "exp": sp.exp, "log": sp.log, "abs": sp.Abs,
           "pi": sp.pi, "e": sp.E}

@dataclass(frozen=True)
class ParsedFunction:
    expression: sp.Expr
    numeric: callable
    derivative_expression: sp.Expr
    derivative: callable
    high_precision: callable
    derivative_high_precision: callable

def parse_function(text: str) -> ParsedFunction:
    """Convierte texto permitido en expresión y funciones NumPy; rechaza símbolos ajenos."""
    if not text.strip():
        raise ValueError("Ingresá una función.")
    try:
        expr = parse_expr(text, local_dict=ALLOWED, global_dict={"__builtins__": {}, **sp.__dict__},
                          transformations=standard_transformations + (convert_xor, implicit_multiplication_application),
                          evaluate=True)
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar la función: {exc}") from None
    if expr.free_symbols - {X}:
        raise ValueError("Solo se admite la variable x.")
    if not expr.is_real and expr.is_real is False:
        raise ValueError("La expresión no representa una función real.")
    derivative = sp.diff(expr, X)
    return ParsedFunction(
        expr,
        sp.lambdify(X, expr, modules=["numpy"]),
        derivative,
        sp.lambdify(X, derivative, modules=["numpy"]),
        sp.lambdify(X, expr, modules=["mpmath"]),
        sp.lambdify(X, derivative, modules=["mpmath"]),
    )

def safe_float(fn: callable, x: float) -> mp.mpf:
    """Evalúa con precisión arbitraria y controla errores del dominio real."""
    try:
        try:
            with np.errstate(all="ignore"):
                value = fn(x)
        except (TypeError, AttributeError):
            with np.errstate(all="ignore"):
                value = fn(float(x))
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, complex) or getattr(value, "imag", 0) != 0:
            raise ValueError
        value = mp.mpf(value)
    except Exception:
        raise ValueError(f"La función no está definida en los reales para x={mp.nstr(x,10)}.") from None
    if not mp.isfinite(value):
        raise ValueError(f"La evaluación en x={mp.nstr(x,10)} produjo NaN o infinito; revisá el dominio.")
    return value
