import sympy as sp

from methods.numerical_differentiation import (differentiate_second_function,
                                                differentiate_table_all)


def test_centered_second_derivative_for_pdf_polynomial_exercise():
    x = sp.Symbol("x", real=True)
    result = differentiate_second_function(x**3 - x, "1", "0.1")
    assert result.approximation == 6
    assert result.exact_derivative == 6
    assert result.absolute_error == 0


def test_complete_position_table_uses_endpoint_and_centered_differences():
    rows = differentiate_table_all([0, 1, 2, 3], [0, 1, 4, 9])
    assert [row.first_derivative for row in rows] == [1, 2, 4, 5]
    assert [row.second_derivative for row in rows] == [2, 2, 2, 2]
