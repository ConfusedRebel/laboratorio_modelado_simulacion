from core.parser import parse_function
from methods import newton
def test_newton_cubic():
    f=parse_function("x^3-x-4");r=newton(f.numeric,f.derivative,1,1e-10)
    assert r.converged and abs(float(f.numeric(r.root)))<1e-9 and r.iterations<20
def test_zero_derivative_controlled():
    f=parse_function("x^2+1");r=newton(f.numeric,f.derivative,0)
    assert not r.converged and "Derivada" in r.stop_reason
