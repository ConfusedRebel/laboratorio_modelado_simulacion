"""Método de Newton-Raphson con protecciones numéricas."""
from core.models import IterationResult, MethodResult
from core.errors import metrics, should_stop
from core.parser import safe_float
import mpmath as mp

def newton(f, df, x0: float, tolerance: float = 1e-8, max_iterations: int = 100,
           criteria: set[str] | None = None, mode: str = "alguno", derivative_delta: float = 1e-12) -> MethodResult:
    criteria=criteria or {"Error absoluto", "Residuo"}; history=[]; x=mp.mpf(x0); tolerance=mp.mpf(tolerance); seen=[]; rising=0
    for n in range(1, max_iterations+1):
        fx, dfx = safe_float(f,x), safe_float(df,x)
        if abs(dfx) < derivative_delta:
            return MethodResult("Newton-Raphson", False, x, n-1, f"Derivada aproximadamente cero en x={mp.nstr(x,10)}.", history,
                                ["Newton no puede continuar de forma estable: se evitó dividir por un valor casi nulo."])
        x_next=x-fx/dfx; f_next=safe_float(f,x_next); m=metrics(x,x_next,f_next)
        history.append(IterationResult(n,x,x,x_next,fx,m.absolute,m.relative,abs(f_next),{"f'(x_n)":dfx,"f(x_n+1)":f_next}))
        stop,reason=should_stop(m,tolerance,criteria,mode,n,max_iterations)
        if stop:
            converged=any(label in reason for label in ("error absoluto","error relativo","residuo"))
            return MethodResult("Newton-Raphson",converged,x_next,n,reason,history)
        if abs(x_next)>1e12:return MethodResult("Newton-Raphson",False,x_next,n,"Divergencia aparente.",history,["Las aproximaciones crecieron excesivamente."])
        if any(abs(x_next-v)<=tolerance/10 for v in seen[-6:-1]):return MethodResult("Newton-Raphson",False,x_next,n,"Se detectó un ciclo corto.",history)
        if len(history)>2 and history[-1].absolute_error>history[-2].absolute_error:rising+=1
        else:rising=0
        if rising>=8:return MethodResult("Newton-Raphson",False,x_next,n,"Divergencia aparente.",history,["El error aumentó durante varias iteraciones."])
        seen.append(x);x=x_next
    return MethodResult("Newton-Raphson",False,x,max_iterations,"Se alcanzó el máximo de iteraciones.",history)
