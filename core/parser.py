"""Análisis seguro de expresiones matemáticas mediante SymPy."""
from dataclasses import dataclass
import numpy as np
import sympy as sp
import mpmath as mp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        convert_xor, implicit_multiplication_application)

X = sp.Symbol("x", real=True)
ALLOWED = {"x": X, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
           "sqrt": sp.sqrt, "root": sp.root, "exp": sp.exp, "log": sp.log,
           "ln": sp.log, "log10": lambda value: sp.log(value, 10), "abs": sp.Abs,
           "pi": sp.pi, "e": sp.E}

@dataclass(frozen=True)
class ParsedFunction:
    expression: sp.Expr
    numeric: callable
    derivative_expression: sp.Expr
    derivative: callable
    high_precision: callable
    derivative_high_precision: callable


@dataclass(frozen=True)
class ParsedConstant:
    expression: sp.Expr
    value: mp.mpf


def _normalize_math_symbols(text: str) -> str:
    """Accept common symbols users naturally paste from mathematical notation."""
    return (text.replace("π", "pi").replace("ℯ", "e")
            .replace("√", "sqrt").replace("×", "*").replace("·", "*")
            .replace("÷", "/").replace("−", "-"))


def parse_constant(text: str) -> ParsedConstant:
    """Parse a real, finite constant expression such as pi/2 or -sqrt(2)."""
    if not text.strip():
        raise ValueError("Ingresá un límite.")
    try:
        expr = parse_expr(
            _normalize_math_symbols(text),
            local_dict=ALLOWED,
            global_dict={"__builtins__": {}, **sp.__dict__},
            transformations=standard_transformations + (convert_xor, implicit_multiplication_application),
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar el límite: {exc}") from None
    if expr.free_symbols:
        raise ValueError("El límite debe ser una constante; no puede contener variables.")
    try:
        numeric = sp.N(expr, 50)
        if numeric.is_real is not True:
            raise ValueError
        value = mp.mpf(str(numeric))
    except (TypeError, ValueError):
        raise ValueError("El límite debe ser un número real.") from None
    if not mp.isfinite(value):
        raise ValueError("El límite debe ser finito.")
    return ParsedConstant(expr, value)


def _high_precision_callable(expr: sp.Expr) -> callable:
    """Evaluate with mpmath, completing finite removable singularities by continuity."""
    direct = sp.lambdify(X, expr, modules=["mpmath"])

    def evaluate(value):
        try:
            result = direct(value)
            if mp.isfinite(result):
                return result
        except Exception:
            pass

        # Newton-Cotes rules must sample their nodes.  Expressions such as
        # sin(x)/x are undefined syntactically at zero although their limit is
        # finite, so use that limit instead of rejecting the whole integral.
        point = sp.Float(str(mp.mpf(value)), max(mp.mp.dps, 30))
        limit = sp.limit(expr, X, point)
        numeric = sp.N(limit, max(mp.mp.dps, 30))
        if numeric.is_real is not True:
            raise ValueError
        result = mp.mpf(str(numeric))
        if not mp.isfinite(result):
            raise ValueError
        return result

    return evaluate

def parse_function(text: str) -> ParsedFunction:
    """Convierte texto permitido en expresión y funciones NumPy; rechaza símbolos ajenos."""
    if not text.strip():
        raise ValueError("Ingresá una función.")
    try:
        expr = parse_expr(_normalize_math_symbols(text), local_dict=ALLOWED, global_dict={"__builtins__": {}, **sp.__dict__},
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
        _high_precision_callable(expr),
        _high_precision_callable(derivative),
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
