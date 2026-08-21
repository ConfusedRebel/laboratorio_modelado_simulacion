from core.parser import parse_function
from methods import fixed_point
def test_contracting_cosine():
    f=parse_function("x-cos(x)").numeric;g=parse_function("cos(x)").numeric;r=fixed_point(f,g,.5,1e-9,200)
    assert r.converged and abs(r.root-.7390851332)<1e-7
