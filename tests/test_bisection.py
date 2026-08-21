import pytest
from core.parser import parse_function
from methods import bisection

def test_bisection_cubic():
    f=parse_function("x^3-x-2").numeric;r=bisection(f,1,2,1e-10)
    assert r.converged and abs(r.root-1.5213797068)<1e-8 and r.iterations<50
def test_invalid_interval():
    with pytest.raises(ValueError):bisection(parse_function("x^2+1").numeric,-1,1)
