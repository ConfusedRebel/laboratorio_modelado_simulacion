import pytest

from core.command_console import format_console_result, solve_command


def test_newton_command_solves_function():
    solved = solve_command("newton f=x^3-x-2 x0=1.5 tol=8 max=100")
    assert solved.value.converged
    assert float(solved.value.root) == pytest.approx(1.5213797068)


def test_console_supports_symbolic_constants_in_derivative():
    solved = solve_command("derivative f=sin(x) x0=pi/4 h=1/100 method=centered")
    assert float(solved.value.approximation) == pytest.approx(2 ** -0.5, rel=1e-4)


def test_lagrange_command_can_evaluate_polynomial():
    solved = solve_command('lagrange xs="0,1,2" ys="1,3,0" at=1.5')
    assert float(solved.details["evaluation"]) == pytest.approx(2.125)


def test_console_reports_missing_parameters():
    with pytest.raises(ValueError, match="Faltan parámetros: x0"):
        solve_command("newton f=x^2-2")


def test_console_accepts_a_pasted_dictionary():
    solved = solve_command("{'command': 'newton', 'f': 'x^2-2', 'x0': 1, 'tol': 8}")
    assert solved.value.converged
    assert float(solved.value.root) == pytest.approx(2 ** 0.5)


def test_console_dictionary_accepts_lists_for_lagrange():
    solved = solve_command("{'comando': 'lagrange', 'xs': [0, 1, 2], 'ys': [1, 3, 0], 'at': 1.5}")
    assert float(solved.details["evaluation"]) == pytest.approx(2.125)


def test_terminal_formatter_includes_the_answer():
    output = format_console_result(solve_command("derivative f=x^2 x0=1 h=1/2 method=centered"))
    assert "Resultado: 2" in output
