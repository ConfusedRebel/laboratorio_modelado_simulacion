import mpmath as mp
import pytest

from core.parser import parse_function
from methods.numerical_integration import integrate_gauss_legendre, integrate_newton_cotes
from visualizations.plots import integration_plot


@pytest.mark.parametrize("method,n,expected", [
    ("midpoint", 1, mp.mpf("0.25")),
    ("left_rectangle", 1, mp.mpf("0")),
    ("right_rectangle", 1, mp.mpf("1")),
    ("trapezoid", 1, mp.mpf("0.5")),
    ("simpson_13", 2, mp.mpf(1) / 3),
    ("simpson_38", 3, mp.mpf(1) / 3),
])
def test_simple_rules_for_x_squared(method, n, expected):
    parsed = parse_function("x^2")
    result = integrate_newton_cotes(parsed.high_precision, 0, 1, n, method)
    assert mp.almosteq(result.approximation, expected)
    assert result.n == n
    assert mp.almosteq(result.h, mp.mpf(1) / n)


@pytest.mark.parametrize("method,n", [
    ("midpoint", 12), ("left_rectangle", 1200), ("right_rectangle", 1200),
    ("trapezoid", 12), ("simpson_13", 12), ("simpson_38", 12),
])
def test_composite_rules_approximate_known_integral(method, n):
    result = integrate_newton_cotes(parse_function("x^2").high_precision, 0, 1, n, method)
    tolerance = mp.mpf("0.002") if method in {"midpoint", "left_rectangle", "right_rectangle", "trapezoid"} else mp.mpf("1e-14")
    assert abs(result.approximation - mp.mpf(1) / 3) < tolerance


@pytest.mark.parametrize("method,n,message", [
    ("simpson_13", 3, "par"),
    ("simpson_38", 4, "múltiplo de 3"),
    ("trapezoid", 0, "entero positivo"),
    ("midpoint", 1.5, "entero positivo"),
])
def test_invalid_subinterval_count(method, n, message):
    with pytest.raises(ValueError, match=message):
        integrate_newton_cotes(parse_function("x").high_precision, 0, 1, n, method)


def test_reverse_limits_and_domain_errors():
    reversed_result = integrate_newton_cotes(parse_function("x^2").high_precision, 1, 0, 2, "simpson_13")
    assert mp.almosteq(reversed_result.approximation, -mp.mpf(1) / 3)
    with pytest.raises(ValueError, match="no está definida"):
        integrate_newton_cotes(parse_function("log(x)").high_precision, -1, 1, 2, "trapezoid")


def test_trapezoid_accepts_removable_singularity_at_a_node():
    result = integrate_newton_cotes(
        parse_function("sin(x)/x").high_precision, -1, 1, 4, "trapezoid"
    )

    assert result.points[2].x == 0
    assert result.points[2].fx == 1
    expected = mp.mpf("0.25") * (
        mp.sin(1) + 2 * mp.sin(mp.mpf("0.5")) / mp.mpf("0.5") + 2
        + 2 * mp.sin(mp.mpf("0.5")) / mp.mpf("0.5") + mp.sin(1)
    )
    assert mp.almosteq(result.approximation, expected)


@pytest.mark.parametrize("method,n,shape_name", [
    ("midpoint", 4, "Rectángulos por punto medio"),
    ("left_rectangle", 4, "Rectángulos izquierdos"),
    ("right_rectangle", 4, "Rectángulos derechos"),
    ("trapezoid", 4, "Trapecios"),
    ("simpson_13", 4, "Parábolas de Simpson 1/3"),
    ("simpson_38", 6, "Cúbicas de Simpson 3/8"),
])
def test_integration_plot_shows_the_geometric_approximation(method, n, shape_name):
    parsed = parse_function("exp(x)")
    result = integrate_newton_cotes(parsed.high_precision, 0, 1, n, method)

    figure = integration_plot(parsed.numeric, result)

    assert shape_name in [trace.name for trace in figure.data]
    assert "f(x)" in [trace.name for trace in figure.data]


def test_gauss_legendre_reference_uses_requested_nodes_and_handles_reverse_limits():
    function = parse_function("exp(x)").high_precision
    forward = integrate_gauss_legendre(function, 0, 1, 32)
    reverse = integrate_gauss_legendre(function, 1, 0, 32)

    assert mp.almosteq(forward, mp.e - 1)
    assert mp.almosteq(reverse, -forward)
