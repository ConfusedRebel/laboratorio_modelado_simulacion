import sympy as sp
import pytest

from methods.lagrange import X, basis_factor, build_lagrange


def test_whiteboard_example():
    result = build_lagrange([0, 1, 2], [1, 3, 0])
    assert sp.expand(result.polynomial) == -sp.Rational(5, 2)*X**2 + sp.Rational(9, 2)*X + 1
    assert [float(result.evaluate(value)) for value in [0, 1, 2]] == [1.0, 3.0, 0.0]


def test_bases_have_delta_property():
    result = build_lagrange([0, 1, 2], [1, 3, 0])
    for basis in result.bases:
        values = [sp.simplify(basis.expression.subs(X, node)) for node in result.x_values]
        assert values == [1 if index == basis.i else 0 for index in range(3)]


def test_factor_and_repeated_nodes():
    result = build_lagrange([0, 1, 2], [1, 3, 0])
    assert sp.simplify(basis_factor(result, 0, 1) - (X-1)/(0-1)) == 0
    with pytest.raises(ValueError):
        build_lagrange([0, 0], [1, 2])


def test_twenty_nodes_are_supported_but_not_twenty_one():
    nodes=list(range(20))
    result=build_lagrange(nodes,[value**2 for value in nodes])
    assert result.degree==2
    with pytest.raises(ValueError):
        build_lagrange(list(range(21)),list(range(21)))
