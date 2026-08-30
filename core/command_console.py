"""Parser and dispatcher for the command-line style exercise console."""
import ast
from dataclasses import dataclass
import shlex
import sys

import mpmath as mp
import sympy as sp

from core.parser import parse_constant, parse_function
from methods import (bisection, build_lagrange, differentiate_function, fixed_point,
                     integrate_newton_cotes, newton)


COMMAND_HELP = (
    ("newton", "newton f=x^3-x-2 x0=1.5 tol=8 max=100", "Raíz por Newton–Raphson."),
    ("bisection", "bisection f=x^3-x-2 a=1 b=2 tol=8 max=100", "Raíz por bisección."),
    ("fixed", "fixed f=x-cos(x) g=cos(x) x0=0.5 tol=8 max=100", "Iteración de punto fijo."),
    ("integrate", "integrate f=sin(x) a=0 b=pi n=12 method=simpson_13", "Integral de Newton–Cotes."),
    ("derivative", "derivative f=sin(x) x0=pi/4 h=1/100 method=centered", "Derivada por diferencias finitas."),
    ("lagrange", 'lagrange xs="0,1,2" ys="1,3,0" at=1.5', "Polinomio interpolante y evaluación opcional."),
)

ALIASES = {
    "newton": "newton", "newton-raphson": "newton",
    "bisection": "bisection", "biseccion": "bisection", "bisección": "bisection",
    "fixed": "fixed", "punto-fijo": "fixed", "punto_fijo": "fixed",
    "integrate": "integrate", "integrar": "integrate", "integral": "integrate",
    "derivative": "derivative", "derivar": "derivative", "derivada": "derivative",
    "lagrange": "lagrange", "interpolar": "lagrange",
}


@dataclass(frozen=True)
class ConsoleResult:
    command: str
    expression: sp.Expr | None
    value: object
    details: dict


def _arguments(text: str) -> tuple[str, dict[str, str]]:
    text = text.strip()
    if text.startswith("{"):
        return _dictionary_arguments(text)
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"Comando incompleto: {exc}") from None
    if not tokens:
        raise ValueError("Ingresá un comando. Escribí `help` para ver ejemplos.")
    command = ALIASES.get(tokens[0].lower())
    if not command:
        raise ValueError(f"Comando desconocido: {tokens[0]}. Escribí `help` para ver el diccionario.")
    values: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            raise ValueError(f"No se entiende `{token}`. Cada dato debe escribirse como nombre=valor.")
        key, value = token.split("=", 1)
        if not key or not value:
            raise ValueError(f"El parámetro `{token}` está incompleto.")
        values[key.lower()] = value
    return command, values


def parse_command(text: str) -> tuple[str, dict[str, str]]:
    """Return the normalized command and parameters without executing it."""
    return _arguments(text)


def _dictionary_arguments(text: str) -> tuple[str, dict[str, str]]:
    """Accept a safely parsed Python/JSON-like dictionary pasted into the console."""
    try:
        data = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        raise ValueError("El diccionario no es válido. Usá, por ejemplo, "
                         "{'command': 'newton', 'f': 'x^2-2', 'x0': 1}.") from None
    if not isinstance(data, dict):
        raise ValueError("El prompt pegado debe ser un diccionario.")
    raw_command = data.get("command", data.get("comando", data.get("method", data.get("metodo"))))
    if raw_command is None:
        raise ValueError("El diccionario debe incluir `command` (o `comando`).")
    command = ALIASES.get(str(raw_command).lower())
    if not command:
        raise ValueError(f"Comando desconocido: {raw_command}. Escribí `help` para ver el diccionario.")
    values = {}
    for key, value in data.items():
        if str(key).lower() in {"command", "comando", "method", "metodo"}:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(map(str, value))
        values[str(key).lower()] = str(value)
    return command, values


def _required(values: dict[str, str], *names: str) -> list[str]:
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError("Faltan parámetros: " + ", ".join(missing) + ".")
    return [values[name] for name in names]


def _constant(text: str):
    return parse_constant(text).value


def _settings(values: dict[str, str]) -> tuple[mp.mpf, int, int]:
    try:
        digits = int(values.get("tol", "8"))
        maximum = int(values.get("max", "100"))
    except ValueError:
        raise ValueError("tol y max deben ser números enteros.") from None
    if not 1 <= digits <= 100 or not 1 <= maximum <= 2000:
        raise ValueError("Usá tol entre 1 y 100 decimales y max entre 1 y 2000 iteraciones.")
    return mp.power(10, -digits), maximum, digits


def _list(text: str) -> list[str]:
    result = [item.strip() for item in text.split(",") if item.strip()]
    if not result:
        raise ValueError("La lista no puede estar vacía.")
    return result


def solve_command(text: str) -> ConsoleResult:
    """Parse one console command and solve it using the application's engines."""
    command, values = _arguments(text)
    if command in {"newton", "bisection", "fixed"}:
        tolerance, maximum, digits = _settings(values)
        f_text = _required(values, "f")[0]
        parsed = parse_function(f_text)
        with mp.workdps(max(mp.mp.dps, digits + 20)):
            if command == "newton":
                x0 = _constant(_required(values, "x0")[0])
                result = newton(parsed.high_precision, parsed.derivative_high_precision, x0,
                                tolerance, maximum)
            elif command == "bisection":
                a, b = map(_constant, _required(values, "a", "b"))
                result = bisection(parsed.high_precision, a, b, tolerance, maximum)
            else:
                g_text, x0_text = _required(values, "g", "x0")
                g = parse_function(g_text)
                result = fixed_point(parsed.high_precision, g.high_precision, _constant(x0_text),
                                     tolerance, maximum)
        return ConsoleResult(command, parsed.expression, result, {})

    if command == "integrate":
        f_text, a_text, b_text = _required(values, "f", "a", "b")
        parsed = parse_function(f_text)
        try:
            n = int(values.get("n", "1"))
        except ValueError:
            raise ValueError("n debe ser un entero positivo.") from None
        method = values.get("method", "trapezoid")
        result = integrate_newton_cotes(parsed.high_precision, _constant(a_text),
                                        _constant(b_text), n, method)
        return ConsoleResult(command, parsed.expression, result, {})

    if command == "derivative":
        f_text, x0_text, h_text = _required(values, "f", "x0", "h")
        parsed = parse_function(f_text)
        result = differentiate_function(parsed.expression, parse_constant(x0_text).expression,
                                        parse_constant(h_text).expression,
                                        values.get("method", "centered"))
        return ConsoleResult(command, parsed.expression, result, {})

    xs_text, ys_text = _required(values, "xs", "ys")
    result = build_lagrange(_list(xs_text), _list(ys_text))
    details = {}
    if "at" in values:
        details["at"] = values["at"]
        details["evaluation"] = result.evaluate(values["at"], 30)
    return ConsoleResult(command, result.polynomial, result, details)


def format_console_result(solved: ConsoleResult) -> str:
    """Build a readable plain-text result for terminals and copy/paste."""
    lines = [f"Método: {solved.command}"]
    if solved.expression is not None:
        lines.append(f"Expresión: {solved.expression}")
    result = solved.value
    if solved.command in {"newton", "bisection", "fixed"}:
        lines.extend((
            f"Estado: {'convergió' if result.converged else 'no convergió'}",
            f"Resultado: {mp.nstr(result.root, 16) if result.root is not None else '—'}",
            f"Iteraciones: {result.iterations}",
            f"Motivo: {result.stop_reason}",
        ))
    elif solved.command == "integrate":
        lines.extend((f"Regla: {result.method}",
                      f"Resultado: {mp.nstr(result.approximation, 16)}",
                      f"Subintervalos: {result.n}", f"Paso h: {mp.nstr(result.h, 16)}"))
    elif solved.command == "derivative":
        lines.extend((f"Esquema: {result.scheme}", f"Resultado: {result.approximation}",
                      f"Derivada exacta: {result.exact_derivative}",
                      f"Error absoluto: {result.absolute_error}"))
    else:
        lines.extend((f"Polinomio: {result.polynomial}", f"Grado: {result.degree}"))
        if "evaluation" in solved.details:
            lines.append(f"P({solved.details['at']}) = {solved.details['evaluation']}")
    return "\n".join(lines)


def main() -> None:
    """Run one command from argv or an interactive prompt when no argv is given."""
    if len(sys.argv) > 1:
        prompts = [" ".join(sys.argv[1:])]
    else:
        print("Consola de ejercicios. Pegá un comando o diccionario; `help` muestra ejemplos; `salir` termina.")
        prompts = iter(lambda: input("ejercicio> ").strip(), "salir")
    for prompt in prompts:
        if not prompt:
            continue
        if prompt.lower() in {"help", "ayuda", "?"}:
            for _, example, description in COMMAND_HELP:
                print(f"{example}\n  {description}")
            continue
        try:
            print(format_console_result(solve_command(prompt)))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
