"""Aceleración delta-cuadrado de Aitken."""
from core.models import IterationResult, MethodResult
from core.parser import safe_float
import mpmath as mp

def aitken_accelerate(f, sequence: list[float], tolerance: float = 1e-8, denominator_delta: float | None = None) -> MethodResult:
    if len(sequence)<3: raise ValueError("Aitken necesita al menos tres términos de una sucesión.")
    sequence=[mp.mpf(value) for value in sequence];tolerance=mp.mpf(tolerance)
    denominator_delta=mp.power(10,-(mp.mp.dps-5)) if denominator_delta is None else mp.mpf(denominator_delta)
    history=[];warnings=[];previous=None
    for n,(x0,x1,x2) in enumerate(zip(sequence,sequence[1:],sequence[2:])):
        delta=x1-x0; delta2=x2-2*x1+x0
        if abs(delta2)<denominator_delta:
            warnings.append(f"n={n}: Δ²x es aproximadamente cero; se omitió la división.");continue
        accelerated=x0-delta*delta/delta2; residual=abs(safe_float(f,accelerated))
        absolute=float("inf") if previous is None else abs(accelerated-previous)
        relative=float("inf") if previous is None or accelerated==0 else absolute/abs(accelerated)
        history.append(IterationResult(n,previous,x0,accelerated,safe_float(f,accelerated),absolute,relative,residual,
            {"x_n":x0,"x_n+1":x1,"x_n+2":x2,"delta_x":delta,"delta2_x":delta2,"x_Aitken":accelerated}))
        previous=accelerated
    if not history:return MethodResult("Aitken Δ²",False,None,0,"No hubo denominadores numéricamente seguros.",[],warnings)
    last=history[-1]; converged=last.residual is not None and last.residual<=tolerance
    return MethodResult("Aitken Δ²",converged,last.x_next,len(history),"Residuo dentro de tolerancia." if converged else "Secuencia acelerada calculada.",history,warnings)
