import pytest
import sympy as sp

from core.parser import parse_function
from methods.numerical_differentiation import differentiate_function, differentiate_nodes


def test_centered_function_is_exact_for_quadratic():
    result = differentiate_function(parse_function("x^2").expression, "2", "1/10", "centered")
    assert result.approximation == 4
    assert result.exact_derivative == 4
    assert result.absolute_error == 0
    assert result.order == 2
    assert [value.x for value in result.table_values] == [sp.Rational(19, 10), 2, sp.Rational(21, 10)]
    assert [value.role for value in result.table_values] == ["Un paso atrás", "Punto elegido", "Un paso adelante"]


def test_forward_preserves_fractions_and_reports_error():
    result = differentiate_function(parse_function("x^2").expression, "1/2", "1/4", "forward")
    assert result.approximation == sp.Rational(5, 4)
    assert result.signed_error == sp.Rational(1, 4)
    assert result.order == 1


def test_nodes_centered():
    result = differentiate_nodes(["0", "1/2", "1"], ["0", "1/4", "1"], "1/2", "centered")
    assert result.approximation == 1
    assert result.h == sp.Rational(1, 2)
    assert result.exact_derivative is None
    assert [value.y for value in result.table_values] == [0, sp.Rational(1, 4), 1]


def test_table_marks_unavailable_neighbour_without_breaking_one_sided_scheme():
    result = differentiate_nodes([0, 1], [0, 1], 0, "forward")
    assert result.approximation == 1
    assert result.table_values[0].y is None


@pytest.mark.parametrize("xs,ys,message", [
    ([0, 1, 1], [0, 1, 1], "repetidos"),
    ([0, 1, 3], [0, 1, 9], "igualmente espaciados"),
])
def test_invalid_nodes(xs, ys, message):
    with pytest.raises(ValueError, match=message):
        differentiate_nodes(xs, ys, xs[1], "centered")


def test_missing_node_and_undefined_function_are_explained():
    with pytest.raises(ValueError, match="Faltan los nodos"):
        differentiate_nodes([0, 1], [0, 1], 0, "backward")
    with pytest.raises(ValueError, match="no está definida"):
        differentiate_function(parse_function("log(x)").expression, 0, 1, "backward")
