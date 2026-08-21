from core.parser import parse_function
from methods import aitken_accelerate
def test_linear_sequence():
    seq=[1-0.5**n for n in range(10)];f=parse_function("x-1").numeric;r=aitken_accelerate(f,seq,1e-12)
    assert r.history and abs(r.root-1)<1e-12
