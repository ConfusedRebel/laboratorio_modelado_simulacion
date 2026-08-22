import pytest
import sympy as sp

from core.calculator import calculate


def test_calculator_keeps_exact_constants_and_roots():
    result = calculate("sin(pi/2)+root(8,3)+2^3")
    assert result.exact == 11


def test_calculator_supports_fraction_and_arbitrary_root():
    assert calculate("1/3+sqrt(4)").exact == sp.Rational(7, 3)


@pytest.mark.parametrize("text,message", [("x+1", "no ingreses x"), ("1/0", "real finito")])
def test_calculator_rejects_variables_and_invalid_results(text, message):
    with pytest.raises(ValueError, match=message):
        calculate(text)
