"""Visualizaciones matemáticas interactivas."""
import numpy as np
import plotly.graph_objects as go
import sympy as sp

def _range(lo,hi):
    pad=max((hi-lo)*.25,1);return np.linspace(lo-pad,hi+pad,600)

def bisection_plot(f,row):
    a,b,c=map(float,(row.metadata["a"],row.metadata["b"],row.metadata["c"]));xs=_range(a,b);ys=f(xs)
    fig=go.Figure(go.Scatter(x=xs,y=ys,name="f(x)"));fig.add_hline(y=0,line_color="gray")
    fig.add_vrect(x0=a,x1=b,fillcolor="#5b8ff9",opacity=.12,line_width=0,annotation_text="intervalo actual")
    fig.add_vrect(x0=float(row.metadata["new_a"]),x1=float(row.metadata["new_b"]),fillcolor="#20a39e",opacity=.22,line_width=0,annotation_text="nuevo")
    fig.add_trace(go.Scatter(x=[a,c,b],y=[float(row.metadata["f(a)"]),float(row.metadata["f(c)"]),float(row.metadata["f(b)"])],mode="markers+text",text=["a","c","b"],textposition="top center",name="puntos"))
    fig.update_layout(title=f"Bisección — iteración {row.iteration}",xaxis_title="x",yaxis_title="f(x)");return fig

def fixed_point_plot(g, history, index, a,b):
    xs=_range(a,b);fig=go.Figure();fig.add_trace(go.Scatter(x=xs,y=xs,name="y=x"));fig.add_trace(go.Scatter(x=xs,y=g(xs),name="y=g(x)"))
    rows=history[:index+1];cx=[];cy=[]
    if rows:
        cx=[float(rows[0].x_previous)];cy=[float(rows[0].x_previous)]
        for r in rows:cx += [float(r.x_previous),float(r.x_next)];cy += [float(r.x_next),float(r.x_next)]
    fig.add_trace(go.Scatter(x=cx,y=cy,mode="lines+markers",name="telaraña",line=dict(color="#ef8354")))
    fig.update_layout(title=f"Cobweb — hasta iteración {index+1}",xaxis_title="xₙ",yaxis_title="g(xₙ)");return fig

def newton_plot(f,row):
    x0,x1,fx,dfx=map(float,(row.x_current,row.x_next,row.fx,row.metadata["f'(x_n)"]));xs=_range(min(x0,x1),max(x0,x1));ys=f(xs);tangent=fx+dfx*(xs-x0)
    fig=go.Figure(go.Scatter(x=xs,y=ys,name="f(x)"));fig.add_hline(y=0,line_color="gray");fig.add_trace(go.Scatter(x=xs,y=tangent,name="tangente"));fig.add_trace(go.Scatter(x=[x0,x1],y=[fx,0],mode="markers+text",text=["(xₙ,f(xₙ))","xₙ₊₁"],textposition="top center",name="paso"));fig.update_layout(title=f"Newton — iteración {row.iteration}",xaxis_title="x",yaxis_title="y");return fig

def newton_all_tangents_plot(f, history):
    """Dibuja como máximo las seis tangentes más recientes de Newton."""
    visible_history = history[-6:]
    points = [float(value) for row in visible_history for value in (row.x_current, row.x_next)]
    lo, hi = min(points), max(points)
    xs = _range(lo, hi)
    fig = go.Figure(go.Scatter(x=xs, y=f(xs), name="f(x)", line=dict(width=4)))
    fig.add_hline(y=0, line_color="gray")
    count = len(visible_history)
    levels = [1.0] if count == 1 else [i / (count - 1) for i in range(count)]
    colors = [
        f"hsl(215,{int(30 + 70 * level)}%,{int(72 - 34 * level)}%)"
        for level in levels
    ]
    for index, row in enumerate(visible_history):
        row_x, row_fx, row_dfx = float(row.x_current), float(row.fx), float(row.metadata["f'(x_n)"])
        tangent = row_fx + row_dfx * (xs - row_x)
        fig.add_trace(go.Scatter(
            x=xs, y=tangent, mode="lines", name=f"Tangente n={row.iteration}",
            line=dict(color=colors[index], width=1.2 + 1.75 * levels[index]), opacity=.35 + .65 * levels[index],
        ))
        fig.add_trace(go.Scatter(
            x=[row_x], y=[row_fx], mode="markers", showlegend=False,
            marker=dict(color=colors[index], size=7),
            hovertemplate=f"n={row.iteration}<br>Xn=%{{x:.12f}}<br>f(Xn)=%{{y:.12f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Newton — últimas 6 tangentes",
        xaxis_title="x", yaxis_title="y",
        legend=dict(traceorder="reversed", title="Más reciente primero"),
    )
    return fig

def convergence_plot(results, metric="absolute_error"):
    fig=go.Figure()
    for result in results:
        values=[float(getattr(r,metric)) for r in result.history];iterations=[r.iteration for r in result.history]
        fig.add_trace(go.Scatter(x=iterations,y=values,mode="lines+markers",name=result.method))
    fig.update_layout(title="Convergencia",xaxis_title="Iteración",yaxis_title=metric,yaxis_type="log");return fig

def approximation_plot(results):
    fig=go.Figure()
    for result in results:
        fig.add_trace(go.Scatter(x=[r.iteration for r in result.history],y=[float(r.x_next if r.x_next is not None else r.x_current) for r in result.history],mode="lines+markers",name=result.method))
    fig.update_layout(title="Aproximación por iteración",xaxis_title="Iteración",yaxis_title="xₙ");return fig

def lagrange_plot(result, evaluation_values=None, show_components=False):
    """Grafica el polinomio, sus nodos, evaluaciones y términos ponderados."""
    node_x = np.asarray([float(value) for value in result.x_values], dtype=float)
    node_y = np.asarray([float(value) for value in result.y_values], dtype=float)
    span = max(float(np.max(node_x) - np.min(node_x)), 1.0)
    xs = np.linspace(float(np.min(node_x) - .12 * span), float(np.max(node_x) + .12 * span), 800)
    x_symbol = sp.Symbol("x", real=True)
    polynomial_fn = sp.lambdify(x_symbol, result.polynomial, "numpy")
    with np.errstate(all="ignore"):
        ys = np.asarray(polynomial_fn(xs), dtype=float)
    if ys.ndim == 0:
        ys = np.full_like(xs, float(ys))
    ys = np.where(np.isfinite(ys), ys, np.nan)
    figure = go.Figure()
    if show_components:
        for basis in result.bases:
            component_fn = sp.lambdify(x_symbol, basis.weighted, "numpy")
            with np.errstate(all="ignore"):
                component_y = np.asarray(component_fn(xs), dtype=float)
            if component_y.ndim == 0:
                component_y = np.full_like(xs, float(component_y))
            component_y = np.where(np.isfinite(component_y), component_y, np.nan)
            figure.add_trace(go.Scatter(
                x=xs, y=component_y, mode="lines", name=f"y{basis.i}·L{basis.i}(x)",
                line=dict(width=1.4, dash="dot"), opacity=.55,
            ))
    figure.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=f"P{result.degree}(x)",
        line=dict(color="#7c3aed", width=4),
    ))
    figure.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", name="Nodos",
        text=[f"({x:g}, {y:g})" for x, y in zip(node_x, node_y)], textposition="top center",
        marker=dict(color="#e63946", size=11, line=dict(color="white", width=1.5)),
    ))
    if evaluation_values:
        evaluation_x = [float(value) for value in evaluation_values]
        evaluation_y = [float(result.evaluate(value, 18)) for value in evaluation_values]
        figure.add_trace(go.Scatter(
            x=evaluation_x, y=evaluation_y, mode="markers+text", name="Valores evaluados",
            text=[f"P({x:g})={y:.8g}" for x, y in zip(evaluation_x, evaluation_y)],
            textposition="bottom center", marker=dict(color="#16a085", size=10, symbol="diamond"),
        ))
    figure.update_layout(
        title="Función construida por interpolación de Lagrange",
        xaxis_title="x", yaxis_title="P(x)", hovermode="x unified",
    )
    return figure


def lagrange_comparison_plot(result, reference_expression):
    """Compara en un gráfico independiente el interpolante y la función original."""
    node_x = np.asarray([float(value) for value in result.x_values], dtype=float)
    span = max(float(np.ptp(node_x)), 1.0)
    xs = np.linspace(float(np.min(node_x) - .12 * span), float(np.max(node_x) + .12 * span), 800)
    symbol = sp.Symbol("x", real=True)
    polynomial_fn = sp.lambdify(symbol, result.polynomial, "numpy")
    reference_fn = sp.lambdify(symbol, reference_expression, "numpy")
    with np.errstate(all="ignore"):
        polynomial_y = np.asarray(polynomial_fn(xs), dtype=float)
        reference_y = np.asarray(reference_fn(xs), dtype=float)
    if polynomial_y.ndim == 0:
        polynomial_y = np.full_like(xs, float(polynomial_y))
    if reference_y.ndim == 0:
        reference_y = np.full_like(xs, float(reference_y))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=xs, y=reference_y, name="Función original f(x)", line=dict(color="#0891b2", width=3)))
    figure.add_trace(go.Scatter(x=xs, y=polynomial_y, name=f"Polinomio P{result.degree}(x)", line=dict(color="#7c3aed", width=4, dash="dash")))
    figure.add_trace(go.Scatter(x=node_x, y=[float(y) for y in result.y_values], mode="markers", name="Nodos", marker=dict(color="#e63946", size=10)))
    figure.update_layout(title="Comparación independiente: función original y polinomio", xaxis_title="x", yaxis_title="y", hovermode="x unified")
    return figure


def differentiation_plot(result):
    """Grafica los datos usados y la recta con la pendiente aproximada."""
    used_x = np.asarray([float(value.x) for value in result.values])
    used_y = np.asarray([float(value.y) for value in result.values])
    center, slope = float(result.point), float(result.approximation)
    span = max(float(np.ptp(used_x)), abs(float(result.h)), 1e-3)
    xs = np.linspace(center - 2 * span, center + 2 * span, 500)
    if result.expression is not None:
        function = sp.lambdify(sp.Symbol("x", real=True), result.expression, "numpy")
        with np.errstate(all="ignore"):
            ys = np.asarray(function(xs), dtype=float)
        if ys.ndim == 0:
            ys = np.full_like(xs, float(ys))
    else:
        xs, ys = used_x, used_y
    center_y = float(result.expression.subs(sp.Symbol("x", real=True), result.point)) if result.expression is not None else float(np.interp(center, used_x, used_y))
    line_x = np.linspace(center - span, center + span, 100)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers" if result.expression is None else "lines", name="f(x) / datos"))
    figure.add_trace(go.Scatter(x=used_x, y=used_y, mode="markers", name="Valores utilizados", marker=dict(color="#e63946", size=11)))
    figure.add_trace(go.Scatter(x=line_x, y=center_y + slope * (line_x - center), mode="lines", name="Pendiente aproximada", line=dict(color="#f59e0b", width=4)))
    figure.add_trace(go.Scatter(x=[center], y=[center_y], mode="markers", name="Punto x₀", marker=dict(color="#111827", size=12, symbol="diamond")))
    figure.update_layout(title="Pendiente aproximada en el punto elegido", xaxis_title="x", yaxis_title="y")
    return figure
