import mpmath as mp
from core.parser import parse_function
from methods import fixed_point
from core.convergence import observed_order

def test_arbitrary_root_syntax():
    parsed=parse_function("root(x,3)-2")
    assert abs(float(parsed.numeric(8)))<1e-12

def test_iteration_stop_criterion():
    f=parse_function("x-cos(x)")
    g=parse_function("cos(x)")
    result=fixed_point(f.high_precision,g.high_precision,mp.mpf("0.5"),mp.mpf("1e-50"),3,{"Iteraciones"})
    assert result.iterations==3
    assert not result.converged

def test_observed_order_accepts_mpf():
    errors=[mp.mpf("0.1"),mp.mpf("0.01"),mp.mpf("0.0001"),mp.mpf("0.00000001")]
    order=observed_order(errors)
    assert order is not None and abs(order-2)<1e-12
