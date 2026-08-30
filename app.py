"""Interfaz educativa de métodos numéricos."""
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sympy as sp

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
(BASE_DIR / "exports").mkdir(parents=True, exist_ok=True)

from core.convergence import contraction_analysis, observed_order
from core.calculator import calculate
from core.command_console import COMMAND_HELP, parse_command, solve_command
from core.parser import X, parse_constant, parse_function, safe_float
from examples.course_exercises import PDF_EXERCISES
from examples.root_examples import EXAMPLES
from methods import (aitken_accelerate, basis_factor, bisection, build_lagrange,
                     differentiate_function, differentiate_nodes,
                     differentiate_second_function, differentiate_table_all, fixed_point,
                     integrate_gauss_legendre, integrate_newton_cotes, newton)
from methods.bisection import theoretical_iterations
from visualizations.plots import (
    approximation_plot, bisection_plot, convergence_plot, fixed_point_plot,
    differentiation_plot, lagrange_comparison_plot, lagrange_plot,
    integration_plot, newton_all_tangents_plot, newton_plot,
    differentiation_plot, lagrange_plot, newton_all_tangents_plot, newton_plot,
)

st.set_page_config(page_title="Modelado y Simulación", page_icon="◑", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:4rem}
.method-header{display:flex;align-items:center;gap:1rem;margin-bottom:.8rem}
.method-logo{width:64px;height:64px;border-radius:18px;display:flex;align-items:center;
justify-content:center;background:linear-gradient(145deg,#3156a3,#20396f);color:white;
font-size:1.7rem;font-weight:800;box-shadow:0 8px 22px rgba(49,86,163,.22)}
.method-name{font-size:2.15rem;font-weight:760;line-height:1.05}
.method-subtitle{opacity:.72;margin-top:.35rem}
.section-label{font-size:1.15rem;font-weight:700;margin:1.8rem 0 .6rem;padding-bottom:.4rem;
border-bottom:1px solid rgba(128,128,128,.25)}
.step-number{display:inline-flex;width:28px;height:28px;border-radius:9px;align-items:center;
justify-content:center;margin-right:.5rem;background:#3156a3;color:white;font-size:.88rem}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);padding:.75rem;
border-radius:12px;background:rgba(128,128,128,.035)}
/* Streamlit solo ofrece una barra lateral; la ubicamos a la derecha como cajón desplegable. */
section[data-testid="stSidebar"]{left:auto;right:0;border-left:1px solid rgba(128,128,128,.22)}
[data-testid="stSidebarCollapsedControl"]{left:auto;right:1rem}
</style>""", unsafe_allow_html=True)

METHODS = {
    "Inicio": ("⌂", "Laboratorio de métodos numéricos", "Visualizar, calcular e interpretar."),
    "Consola": (">_", "Consola de ejercicios", "Escribir una función y resolver el ejercicio con un comando."),
    "Bisección": ("½", "Bisección", "Dividir el intervalo en dos y conservar el cambio de signo."),
    "Punto Fijo": ("●", "Punto Fijo", "Buscar un valor que no cambie al aplicar g(x)."),
    "Newton-Raphson": ("╱╲", "Newton–Raphson", "Usar tangentes para acercarse a la raíz."),
    "Aitken": ("↗", "Aitken Δ²", "Tomar un atajo para acelerar una sucesión."),
    "Comparar métodos": ("VS", "Comparar métodos", "Contrastar aproximaciones, errores y velocidad."),
    "Construir función": ("Σ", "Construir una función", "Interpolar datos discretos con bases de Lagrange."),
    "Interpolar desde función": ("f→Σ", "Interpolar desde una función", "Elegir f(x), solicitar nodos y construir su interpolante."),
    "Derivación numérica": ("f′", "Derivación numérica", "Estimar una pendiente a partir de valores cercanos."),
    "Integración numérica": ("∫", "Integración Numérica - Newton-Cotes", "Aproximar el área firmada bajo una curva."),
    "Ejercicios del PDF": ("PDF", "Ejercicios del material auxiliar", "Cargar los datos del libro directamente en cada método."),
    "Laboratorio": ("⚗", "Laboratorio", "Modificar parámetros y observar consecuencias."),
    "Teoría": ("📖", "Teoría", "Fundamentos matemáticos de los métodos."),
}

GAUSS_REFERENCE_NODES = 32


def method_header(page: str) -> None:
    icon, name, subtitle = METHODS[page]
    st.markdown(
        f'<div class="method-header"><div class="method-logo">{icon}</div><div>'
        f'<div class="method-name">{name}</div><div class="method-subtitle">{subtitle}</div>'
        f'</div></div>', unsafe_allow_html=True,
    )


def section(number: int, title: str) -> None:
    st.markdown(
        f'<div class="section-label"><span class="step-number">{number}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def explain(what: str, when: str, action: str, formula: str) -> None:
    with st.expander("¿Cómo funciona este método?", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Qué busca**  \n{what}")
        col2.markdown(f"**Cuándo conviene**  \n{when}")
        col3.markdown(f"**Qué hace en cada paso**  \n{action}")
        st.latex(formula)
        if not st.session_state.get("advanced_mode", False):
            st.markdown("**Datos necesarios:** una función o valores, un punto inicial y los parámetros que aparecen debajo.  \n"
                        "**Antes de ejecutar:** revisá el indicador de condiciones.  \n"
                        "**Resultado:** es una aproximación; la tabla muestra cómo se obtuvo y el gráfico ayuda a interpretarla.  \n"
                        "**Errores frecuentes:** datos fuera del dominio, condiciones del método incumplidas o muy pocas iteraciones.")


def _append_symbol(target: str, symbol: str) -> None:
    st.session_state[target] = st.session_state.get(target, "") + symbol


def floating_math_keyboard() -> None:
    """Add one global keyboard that writes into the last focused editable field."""
    components.html(
        """
<script>
(() => {
  const doc = window.parent.document;
  const win = window.parent;
  const old = doc.getElementById("global-math-keyboard");
  if (old) old.remove();
  const styleId = "global-math-keyboard-style";
  if (!doc.getElementById(styleId)) {
    const style = doc.createElement("style");
    style.id = styleId;
    style.textContent = `
      #global-math-keyboard{position:fixed;right:18px;top:50%;transform:translateY(-50%);z-index:100000;font-family:Arial,sans-serif;color:#172033}
      #global-math-keyboard .mk-toggle{width:52px;height:52px;border:0;border-radius:16px;background:#3156a3;color:white;font-size:23px;font-weight:700;cursor:pointer;box-shadow:0 8px 24px rgba(28,45,86,.3)}
      #global-math-keyboard .mk-toggle:hover{background:#274889}
      #global-math-keyboard .mk-panel{display:none;position:absolute;right:62px;top:50%;transform:translateY(-50%);width:310px;padding:14px;border:1px solid rgba(90,100,120,.28);border-radius:16px;background:white;box-shadow:0 14px 40px rgba(20,30,55,.24)}
      #global-math-keyboard.open .mk-panel{display:block}
      #global-math-keyboard .mk-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;font-size:16px;font-weight:700}
      #global-math-keyboard .mk-close{border:0;background:transparent;font-size:22px;cursor:pointer;color:#667085}
      #global-math-keyboard .mk-help{font-size:12px;color:#667085;line-height:1.35;margin:0 0 10px}
      #global-math-keyboard .mk-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}
      #global-math-keyboard .mk-key{min-height:38px;border:1px solid #d8deea;border-radius:9px;background:#f7f9fc;color:#20396f;font-size:15px;font-weight:650;cursor:pointer}
      #global-math-keyboard .mk-key:hover{background:#eaf0fb;border-color:#9eb1da}
      @media(max-width:600px){#global-math-keyboard{right:10px;top:auto;bottom:18px;transform:none}#global-math-keyboard .mk-panel{position:fixed;left:10px;right:10px;top:auto;bottom:80px;transform:none;width:auto}}
    `;
    doc.head.appendChild(style);
  }
  const root = doc.createElement("div");
  root.id = "global-math-keyboard";
  root.innerHTML = `<button class="mk-toggle" type="button" title="Abrir teclado matemático" aria-label="Abrir teclado matemático" aria-expanded="false">∑</button>
    <div class="mk-panel" role="dialog" aria-label="Teclado matemático"><div class="mk-head"><span>Teclado matemático</span><button class="mk-close" type="button" aria-label="Cerrar">×</button></div><p class="mk-help">Primero ubicá el cursor en un campo y después elegí un símbolo.</p><div class="mk-grid"></div></div>`;
  doc.body.appendChild(root);
  const editable = el => el && (el.matches("textarea, input:not([type]), input[type='text'], input[type='search']") || el.isContentEditable);
  if (!win.__mathKeyboardFocusListener) {
    win.__mathKeyboardFocusListener = true;
    doc.addEventListener("focusin", event => { if (editable(event.target)) win.__mathKeyboardTarget = event.target; }, true);
  }
  const symbols = [["√","sqrt("],["ⁿ√","root("],["xʸ","^"],["x²","^2"],["(","("],[")",")"],["sin","sin("],["cos","cos("],["tan","tan("],["eˣ","exp("],["ln","ln("],["log₁₀","log10("],["|x|","abs("],["π","pi"],["e","e"],["÷","/"],["×","*"],["−","-"],["+","+"],[",",","],["x","x"]];
  const insert = text => {
    const target = win.__mathKeyboardTarget;
    if (!target || !doc.contains(target)) return;
    target.focus();
    if (target.isContentEditable) {
      doc.execCommand("insertText", false, text);
      target.dispatchEvent(new InputEvent("input", {bubbles:true, inputType:"insertText", data:text}));
      return;
    }
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? start;
    const value = target.value.slice(0, start) + text + target.value.slice(end);
    const prototype = target.tagName === "TEXTAREA" ? win.HTMLTextAreaElement.prototype : win.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, "value").set.call(target, value);
    target.dispatchEvent(new InputEvent("input", {bubbles:true, inputType:"insertText", data:text}));
    target.dispatchEvent(new Event("change", {bubbles:true}));
    const cursor = start + text.length;
    try { target.setSelectionRange(cursor, cursor); } catch (_) {}
  };
  const grid = root.querySelector(".mk-grid");
  symbols.forEach(([label, value]) => {
    const button = doc.createElement("button");
    button.type = "button"; button.className = "mk-key"; button.textContent = label;
    button.addEventListener("mousedown", event => event.preventDefault());
    button.addEventListener("click", () => insert(value));
    grid.appendChild(button);
  });
  const toggle = root.querySelector(".mk-toggle");
  const setOpen = open => { root.classList.toggle("open", open); toggle.setAttribute("aria-expanded", String(open)); };
  toggle.addEventListener("click", () => setOpen(!root.classList.contains("open")));
  root.querySelector(".mk-close").addEventListener("click", () => setOpen(false));
  doc.addEventListener("keydown", event => { if (event.key === "Escape") setOpen(false); });
})();
</script>
        """,
        height=0,
        width=0,
    )


def calculator_drawer() -> None:
    """Calculadora persistente dentro del cajón lateral derecho."""
    with st.sidebar.expander("🧮 Calculadora numérica", expanded=False):
        st.caption("Abrila cuando necesites preparar un dato. No usa `eval`.")
        st.text_input("Cálculo", "", key="calculator_input",
                      help="Ejemplo: sin(pi/2) + root(8,3) + 2^3")
        keys = [("7", "7"), ("8", "8"), ("9", "9"), ("÷", "/"),
                ("4", "4"), ("5", "5"), ("6", "6"), ("×", "*"),
                ("1", "1"), ("2", "2"), ("3", "3"), ("−", "-"),
                ("0", "0"), (".", "."), ("(", "("), (")", ")"),
                ("+", "+"), ("xʸ", "^"), ("√", "sqrt("), ("ⁿ√", "root("),
                ("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("ln", "log("),
                ("π", "pi"), ("e", "e"), ("|x|", "abs("), (",", ",")]
        columns = st.columns(4)
        for index, (label, value) in enumerate(keys):
            columns[index % 4].button(label, key=f"calc_key_{index}",
                                      on_click=_append_symbol, args=("calculator_input", value),
                                      use_container_width=True)
        actions = st.columns(2)
        actions[0].button("Borrar", key="calc_clear", on_click=lambda: st.session_state.update(calculator_input=""), use_container_width=True)
        if actions[1].button("Calcular", key="calc_run", type="primary", use_container_width=True):
            try:
                result = calculate(st.session_state.calculator_input)
                st.session_state.calculator_result = result
            except ValueError as exc:
                st.session_state.pop("calculator_result", None)
                st.error(str(exc))
        if result := st.session_state.get("calculator_result"):
            st.latex(rf"{sp.latex(result.expression)}={sp.latex(result.exact)}")
            st.code(str(result.exact), language=None)
            st.caption(f"Decimal: {result.decimal}. Copiá el valor exacto o decimal al campo que necesites.")


def render_formula(text: str, name: str):
    try:
        parsed = parse_function(text)
    except ValueError as exc:
        st.error(str(exc))
        return None
    st.latex(rf"{name}(x)={sp.latex(parsed.expression)}")
    st.latex(rf"{name}'(x)={sp.latex(parsed.derivative_expression)}")
    return parsed


def _automatic_g(f_key: str, g_key: str, x0_key: str,
                 a_key: str | None = None, b_key: str | None = None) -> None:
    try:
        parsed = parse_function(st.session_state.get(f_key, ""))
        x0 = float(st.session_state.get(x0_key, 0.0))
        lam = None
        if a_key and b_key:
            a, b = float(st.session_state[a_key]), float(st.session_state[b_key])
            xs = np.linspace(a, b, 800)
            with np.errstate(all="ignore"):
                derivatives = np.asarray(parsed.derivative(xs), dtype=float)
            derivatives = derivatives[np.isfinite(derivatives)]
            if len(derivatives):
                d_min, d_max = float(np.min(derivatives)), float(np.max(derivatives))
                if d_min * d_max > 0 and abs(d_min + d_max) > 1e-14:
                    lam = 2.0 / (d_min + d_max)
        if lam is None:
            derivative = float(parsed.derivative(x0))
            if not np.isfinite(derivative) or abs(derivative) < 1e-14:
                raise ValueError("f′(x₀) es aproximadamente cero; no se pudo construir una g estable.")
            lam = 1.0 / derivative
        st.session_state[g_key] = str(sp.simplify(X - sp.Float(lam, 16) * parsed.expression))
        st.session_state[g_key + "_message"] = "g(x) generada como x − λf(x). Verificá su contracción."
        st.session_state.pop(g_key + "_error", None)
    except Exception as exc:
        st.session_state[g_key + "_error"] = str(exc)


def automatic_g_button(f_key: str, g_key: str, x0_key: str,
                       a_key: str | None = None, b_key: str | None = None) -> None:
    st.button("Calcular g(x) automáticamente", key=g_key + "_auto", on_click=_automatic_g,
              args=(f_key, g_key, x0_key, a_key, b_key), use_container_width=True)
    if st.session_state.get(g_key + "_message"):
        st.success(st.session_state[g_key + "_message"])
    if st.session_state.get(g_key + "_error"):
        st.error(st.session_state[g_key + "_error"])


def tolerance_input(prefix: str, default: int = 8) -> mp.mpf:
    digits = int(st.number_input("Decimales de tolerancia", 1, 100, default,
                                 key=prefix + "tol_digits"))
    mp.mp.dps = max(mp.mp.dps, digits + 20)
    tolerance = mp.power(10, -digits)
    st.caption(f"ε = 0.{('0' * (digits - 1))}1")
    if digits > 8:
        st.warning("Más de 8 decimales supera lo pedido en la mayoría de los ejercicios.")
    return tolerance


def controls(prefix: str):
    col1, col2, col3 = st.columns(3)
    with col1:
        tolerance = tolerance_input(prefix)
    with col2:
        maximum = int(st.number_input("Máximo de iteraciones", 1, 2000, 100, key=prefix + "n"))
    with col3:
        decimals = int(st.number_input("Decimales mostrados", 1, 100, 12, key=prefix + "dec"))
        mp.mp.dps = max(mp.mp.dps, decimals + 20)
    st.markdown("**Criterios de detención**")
    criteria = set()
    for column, label, checked in zip(st.columns(4),
            ["Error absoluto", "Error relativo", "Residuo", "Iteraciones"],
            [True, False, True, False]):
        if column.checkbox(label, checked, key=f"{prefix}cr_{label}"):
            criteria.add(label)
    st.caption("Residuo = |f(xₙ)|: indica qué tan cerca está la aproximación de cumplir f(x)=0.")
    mode = st.radio("Detener cuando se cumpla", ["alguno", "todos"], horizontal=True,
                    key=prefix + "mode")
    if not criteria:
        st.warning("Seleccioná al menos un criterio.")
    return tolerance, maximum, criteria, mode, decimals


def criteria_label(criteria: set[str], mode: str) -> str:
    return (" y " if mode == "todos" else " o ").join(sorted(criteria)) if criteria else "Ninguno"


def fixed_decimal(value, decimals: int) -> str:
    number = mp.mpf(value)
    if mp.isnan(number):
        return "—"
    if not mp.isfinite(number):
        return "∞" if number > 0 else "−∞"
    sign = "−" if number < 0 else ""
    scaled = mp.floor(abs(number) * mp.power(10, decimals) + mp.mpf("0.5"))
    digits = str(int(scaled)).rjust(decimals + 1, "0")
    return sign + digits if decimals == 0 else f"{sign}{digits[:-decimals]}.{digits[-decimals:]}"


def function_evaluation_rows(parsed, values, decimals: int) -> list[dict]:
    """Evalúa filas independientes y conserva los errores de dominio en la tabla."""
    rows = []
    for raw_value in values:
        text_value = "" if raw_value is None else str(raw_value).strip()
        if not text_value:
            continue
        try:
            constant = parse_constant(text_value)
            evaluated = safe_float(parsed.high_precision, constant.value)
            rows.append({"x ingresado": text_value,
                         "x decimal": fixed_decimal(constant.value, decimals),
                         "f(x)": fixed_decimal(evaluated, decimals), "Estado": "Correcto"})
        except ValueError as exc:
            rows.append({"x ingresado": text_value, "x decimal": "—", "f(x)": "—",
                         "Estado": str(exc)})
    if not rows:
        raise ValueError("Agregá al menos un valor de x en la tabla.")
    return rows


def integration_development(expression, a_expr, b_expr, n: int, method: str,
                            result, decimals: int) -> tuple[sp.Expr, str, str]:
    """Construye la fórmula aplicada con nodos exactos y su sustitución decimal."""
    # nsimplify preserves constants such as pi and turns decimal limits into
    # rational values, so h is presented as an exact fraction when possible.
    h_expr = sp.nsimplify((b_expr - a_expr) / n)
    if method in {"midpoint", "left_rectangle", "right_rectangle"}:
        if method == "midpoint":
            offsets = [sp.Rational(2 * index + 1, 2) for index in range(n)]
        elif method == "left_rectangle":
            offsets = list(range(n))
        else:
            offsets = list(range(1, n + 1))
        exact_nodes = [sp.simplify(a_expr + offset * h_expr) for offset in offsets]
        factor_expr = h_expr
    else:
        exact_nodes = [sp.simplify(a_expr + index * h_expr) for index in range(n + 1)]
        factor_expr = {"trapezoid": h_expr / 2,
                       "simpson_13": h_expr / 3,
                       "simpson_38": 3 * h_expr / 8}[method]

    symbolic_terms = []
    decimal_terms = []
    for node, point in zip(exact_nodes, result.points):
        coefficient = "" if point.weight == 1 else f"{point.weight}\\,"
        evaluated_expr = sp.simplify(expression.subs(X, node))
        symbolic_terms.append(rf"{coefficient}\left({sp.latex(evaluated_expr)}\right)")
        decimal_terms.append(rf"{coefficient}\left({fixed_decimal(point.fx, decimals)}\right)")

    symbolic = (rf"I_{{{n}}}\approx {sp.latex(sp.simplify(factor_expr))}"
                rf"\left[{'+'.join(symbolic_terms)}\right]")
    numeric_factor = {"midpoint": result.h, "left_rectangle": result.h,
                      "right_rectangle": result.h, "trapezoid": result.h / 2,
                      "simpson_13": result.h / 3,
                      "simpson_38": 3 * result.h / 8}[method]
    decimal = (rf"I_{{{n}}}\approx \left({fixed_decimal(numeric_factor, decimals)}\right)"
               rf"\left[{'+'.join(decimal_terms)}\right]="
               rf"{fixed_decimal(result.approximation, decimals)}")
    return h_expr, symbolic, decimal


def result_frame(result) -> pd.DataFrame:
    if result.method == "Newton-Raphson":
        return pd.DataFrame([{
            "n": row.iteration, "Xn": row.x_current, "f(Xn)": row.fx,
            "f'(Xn)": row.metadata.get("f'(x_n)"), "Xn+1": row.x_next,
            "Error Absoluto": row.absolute_error, "Error Relativo": row.relative_error,
        } for row in result.history])
    return pd.DataFrame([row.as_dict() for row in result.history])


def show_result(result, decimals: int) -> None:
    by_iterations = "iteraciones" in result.stop_reason.lower() and not result.converged
    status = "Finalizado por iteraciones" if by_iterations else ("Convergió" if result.converged else "No convergió")
    cols = st.columns(4)
    cols[0].metric("Estado", status)
    cols[1].metric("Raíz / aproximación", "—" if result.root is None else fixed_decimal(result.root, decimals))
    cols[2].metric("Iteraciones", result.iterations)
    cols[3].metric("Residuo final", "—" if not result.history else fixed_decimal(result.history[-1].residual, decimals))
    st.info(result.stop_reason)
    for warning in result.warnings:
        st.warning(warning)


def show_table(result, decimals: int) -> None:
    if not result.history:
        st.info("No se generaron iteraciones.")
        return
    frame = result_frame(result)
    display = frame.copy()
    for column in display.columns:
        if column not in {"n", "iteration"}:
            display[column] = display[column].map(
                lambda value: fixed_decimal(value, decimals)
                if value is not None and not isinstance(value, (str, bool)) else value)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Descargar iteraciones CSV", frame.to_csv(index=False).encode("utf-8"),
                       f"{result.method.lower().replace(' ', '_')}_iteraciones.csv", "text/csv")
    order = observed_order([row.absolute_error for row in result.history if row.absolute_error is not None])
    if order is not None:
        st.caption(f"Orden experimental observado: p ≈ {order:.3f}.")


def bisection_page() -> None:
    method_header("Bisección")
    explain("Encerrar una raíz entre dos extremos.",
            "Cuando f es continua y f(a), f(b) tienen signos opuestos.",
            "Divide el intervalo y conserva la mitad con cambio de signo.",
            r"c_n=\frac{a_n+b_n}{2}")
    section(1, "Ingresar datos")
    with st.container(border=True):
        example = st.selectbox("Ejercicio precargado", [k for k in EXAMPLES if k.startswith("Bisección")])
        ex = EXAMPLES[example]
        ftext = st.text_input("f(x)", ex["f"], key="bi_f")
        c1, c2 = st.columns(2)
        a = c1.number_input("Extremo a", value=ex["a"], key="bi_a")
        b = c2.number_input("Extremo b", value=ex["b"], key="bi_b")
        tolerance, maximum, criteria, mode, decimals = controls("bi")
    section(2, "Resumen de lo ingresado")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            parsed = render_formula(ftext, "f")
            st.write(f"**Intervalo:** [{a:g}, {b:g}] · **Criterio:** {criteria_label(criteria, mode)}")
    with action:
        run = st.button("Ejecutar", type="primary", disabled=not criteria or parsed is None,
                        key="run_bi", use_container_width=True)
        run = run or st.session_state.pop("console_run_bi", False)
    with st.expander("Buscar intervalos con cambio de signo"):
        st.caption("Útil para los ejercicios que primero piden hallar [a,b]. La búsqueda es numérica y debe confirmarse con continuidad.")
        scan_cols = st.columns(3)
        scan_a = scan_cols[0].number_input("Buscar desde", value=-10.0, key="bi_scan_a")
        scan_b = scan_cols[1].number_input("Buscar hasta", value=10.0, key="bi_scan_b")
        scan_parts = int(scan_cols[2].number_input("Divisiones", 10, 10000, 200, key="bi_scan_parts"))
        if st.button("Detectar cambios de signo", disabled=parsed is None, key="bi_scan"):
            points = np.linspace(scan_a, scan_b, scan_parts + 1)
            intervals = []
            previous_x = previous_y = None
            for current_x in points:
                try:
                    current_y = float(parsed.numeric(current_x))
                except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                    previous_x = previous_y = None
                    continue
                if not np.isfinite(current_y):
                    previous_x = previous_y = None
                    continue
                if current_y == 0:
                    intervals.append({"a": current_x, "b": current_x, "observación": "raíz muestreada"})
                elif previous_y is not None and previous_y * current_y < 0:
                    intervals.append({"a": previous_x, "b": current_x, "observación": "cambio de signo"})
                previous_x, previous_y = current_x, current_y
            if intervals:
                st.dataframe(pd.DataFrame(intervals), hide_index=True, use_container_width=True)
            else:
                st.warning("No se detectaron cambios con esta malla. Ampliá el rango o aumentá las divisiones.")
    if run:
        try:
            result = bisection(parsed.high_precision, a, b, tolerance, maximum, criteria, mode)
            st.session_state.bi_result = (parsed, result, decimals, a, b, tolerance)
        except ValueError as exc:
            st.error(str(exc))
    if "bi_result" not in st.session_state:
        return
    parsed, result, decimals, a, b, tolerance = st.session_state.bi_result
    section(3, "Gráfico del método")
    if result.history:
        index = st.slider("Iteración", 1, len(result.history), len(result.history), key="bi_it")
        st.plotly_chart(bisection_plot(parsed.numeric, result.history[index - 1]), use_container_width=True)
        st.caption(f"Estimación teórica: {theoretical_iterations(a, b, tolerance)} iteraciones.")
    section(4, "Resultado")
    show_result(result, decimals)
    section(5, "Tabla de iteraciones")
    show_table(result, decimals)
    section(6, "Gráfico de convergencia")
    st.plotly_chart(convergence_plot([result]), use_container_width=True)


def fixed_page() -> None:
    method_header("Punto Fijo")
    explain("Encontrar un valor que satisfaga x=g(x).",
            "Cuando g conserva el intervalo y se comporta como contracción.",
            "Sustituye el valor actual dentro de g para obtener el siguiente.",
            r"x_{n+1}=g(x_n)")
    section(1, "Ingresar datos")
    with st.container(border=True):
        name = st.selectbox("Ejercicio precargado", [k for k in EXAMPLES if k.startswith("Punto fijo")])
        ex = EXAMPLES[name]
        ftext = st.text_input("f(x)", ex["f"], key="pf_f")
        c1, c2, c3 = st.columns(3)
        x0 = c1.number_input("x₀", value=ex["x0"], key="pf_x0")
        a = c2.number_input("K: extremo a", value=ex["a"], key="pf_a")
        b = c3.number_input("K: extremo b", value=ex["b"], key="pf_b")
        automatic_g_button("pf_f", "pf_g", "pf_x0", "pf_a", "pf_b")
        gtext = st.text_input("g(x)", ex["g"], key="pf_g")
        tolerance, maximum, criteria, mode, decimals = controls("pf")
    section(2, "Resumen de lo ingresado")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            f_parsed, g_parsed = render_formula(ftext, "f"), render_formula(gtext, "g")
            st.write(f"**x₀:** {x0:g} · **K:** [{a:g}, {b:g}] · **Criterio:** {criteria_label(criteria, mode)}")
    with action:
        run = st.button("Ejecutar", type="primary", disabled=not criteria or not f_parsed or not g_parsed,
                        key="run_pf", use_container_width=True)
        run = run or st.session_state.pop("console_run_pf", False)
    if run:
        try:
            result = fixed_point(f_parsed.high_precision, g_parsed.high_precision, x0,
                                 tolerance, maximum, criteria, mode)
            analysis = contraction_analysis(g_parsed.numeric, g_parsed.derivative, a, b)
            st.session_state.pf_result = (g_parsed, result, analysis, decimals, a, b)
        except ValueError as exc:
            st.error(str(exc))
    if "pf_result" not in st.session_state:
        return
    g_parsed, result, analysis, decimals, a, b = st.session_state.pf_result
    section(3, "Gráfico del método")
    if result.history:
        index = st.slider("Iteración del cobweb", 1, len(result.history), len(result.history), key="pf_it")
        st.plotly_chart(fixed_point_plot(g_parsed.numeric, result.history, index - 1, a, b), use_container_width=True)
    section(4, "Resultado")
    show_result(result, decimals)
    st.write(f"**K=[{a:g},{b:g}]** es cerrado y acotado; por lo tanto, compacto en ℝ.")
    if analysis["L"] is None:
        st.warning("No se pudo estimar L.")
    else:
        L = analysis["L"]
        st.write(f"**L ≈ max |g′(x)| ≈ {L:.6g}** · g(K)⊆K: {'sí' if analysis['maps_into_itself'] else 'no'}")
        if L < 1 and analysis["maps_into_itself"]:
            st.success("Existe evidencia numérica de contracción y convergencia en K.")
        else:
            st.warning("El análisis por muestreo no garantiza convergencia en K.")
    section(5, "Tabla de iteraciones")
    show_table(result, decimals)
    section(6, "Gráfico de convergencia")
    st.plotly_chart(convergence_plot([result]), use_container_width=True)


def newton_page() -> None:
    method_header("Newton-Raphson")
    explain("Encontrar una raíz utilizando la pendiente local.",
            "Cuando f es derivable, f′ no es casi cero y x₀ es razonable.",
            "Traza la tangente en xₙ y usa su corte con el eje x.",
            r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}")
    section(1, "Ingresar datos")
    with st.container(border=True):
        ex = EXAMPLES["Newton — cúbica"]
        ftext = st.text_input("f(x)", ex["f"], key="nw_f")
        x0 = st.number_input("x₀", value=ex["x0"], key="nw_x0")
        tolerance, maximum, criteria, mode, decimals = controls("nw")
    section(2, "Resumen de lo ingresado")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            parsed = render_formula(ftext, "f")
            st.write(f"**x₀:** {x0:g} · **Criterio:** {criteria_label(criteria, mode)}")
    with action:
        run = st.button("Ejecutar", type="primary", disabled=not criteria or parsed is None,
                        key="run_nw", use_container_width=True)
        run = run or st.session_state.pop("console_run_nw", False)
    if run:
        try:
            result = newton(parsed.high_precision, parsed.derivative_high_precision, x0,
                            tolerance, maximum, criteria, mode)
            st.session_state.nw_result = (parsed, result, decimals, x0, ftext)
        except ValueError as exc:
            st.error(str(exc))
    if "nw_result" not in st.session_state:
        return
    parsed, result, decimals, executed_x0, executed_text = st.session_state.nw_result
    if ftext != executed_text or x0 != executed_x0:
        st.info("Cambiaste f(x) o x₀. Presioná «Ejecutar» para actualizar el resultado.")
        return
    section(3, "Gráfico del método")
    if result.history:
        view = st.radio("Tangentes visibles", ["Una iteración", "Últimas 6 iteraciones"], horizontal=True)
        if view == "Una iteración":
            index = st.slider("Iteración", 1, len(result.history), len(result.history), key="nw_it")
            st.plotly_chart(newton_plot(parsed.numeric, result.history[index - 1]), use_container_width=True)
        else:
            st.plotly_chart(newton_all_tangents_plot(parsed.numeric, result.history), use_container_width=True)
            st.caption("La tangente más reciente es la más oscura y saturada.")
    section(4, "Resultado")
    show_result(result, decimals)
    st.write("Cada aproximación surge donde la tangente del punto actual corta el eje x.")
    section(5, "Tabla de iteraciones")
    show_table(result, decimals)
    section(6, "Gráfico de convergencia")
    st.plotly_chart(convergence_plot([result]), use_container_width=True)


def aitken_page() -> None:
    method_header("Aitken")
    explain("Acelerar una sucesión que ya se aproxima a un límite.",
            "Cuando Punto Fijo converge aproximadamente de forma lineal.",
            "Combina tres términos consecutivos para estimar un atajo.",
            r"x_n^*=x_n-\frac{(x_{n+1}-x_n)^2}{x_{n+2}-2x_{n+1}+x_n}")
    section(1, "Ingresar datos")
    with st.container(border=True):
        ftext = st.text_input("f(x)", "x-cos(x)", key="ai_f")
        x0 = st.number_input("x₀", value=.5, key="ai_x0")
        automatic_g_button("ai_f", "ai_g", "ai_x0")
        gtext = st.text_input("g(x)", "cos(x)", key="ai_g")
        c1, c2, c3 = st.columns(3)
        accelerate = c1.checkbox("Aplicar aceleración Δ²", True)
        iterations = c2.slider("Cantidad de iteraciones", 3, 500, 20)
        decimals = int(c3.number_input("Decimales mostrados", 1, 100, 12, key="ai_dec"))
        tolerance = tolerance_input("ai")
        mp.mp.dps = max(mp.mp.dps, decimals + 20)
    section(2, "Resumen de lo ingresado")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            f_parsed, g_parsed = render_formula(ftext, "f"), render_formula(gtext, "g")
            st.write(f"**x₀:** {x0:g} · **Iteraciones:** {iterations} · **Aceleración:** {'sí' if accelerate else 'no'}")
    with action:
        run = st.button("Ejecutar", type="primary", disabled=not f_parsed or not g_parsed,
                        key="run_ai", use_container_width=True)
    if run:
        try:
            original = fixed_point(f_parsed.high_precision, g_parsed.high_precision, x0,
                                   tolerance, iterations, {"Iteraciones"}, "alguno")
            sequence = [mp.mpf(x0)] + [row.x_next for row in original.history]
            results, selected = [original], original
            if accelerate:
                selected = aitken_accelerate(f_parsed.high_precision, sequence, tolerance)
                results.append(selected)
            st.session_state.ai_result = (selected, results, sequence, decimals, accelerate)
        except (ValueError, OverflowError) as exc:
            st.error(str(exc))
    if "ai_result" not in st.session_state:
        return
    result, results, sequence, decimals, accelerate = st.session_state.ai_result
    section(3, "Gráfico del método")
    st.plotly_chart(approximation_plot(results), use_container_width=True)
    st.caption("Sucesión original y atajo Δ²." if accelerate else "Sucesión original sin aceleración.")
    section(4, "Resultado")
    show_result(result, decimals)
    section(5, "Tabla de iteraciones")
    show_table(result, decimals)
    original = pd.DataFrame({"n": range(len(sequence)), "Secuencia original": sequence})
    st.download_button("Descargar sucesión original", original.to_csv(index=False).encode(),
                       "sucesion_original.csv", "text/csv")
    section(6, "Gráfico de convergencia")
    st.plotly_chart(convergence_plot(results), use_container_width=True)


def comparison_page() -> None:
    method_header("Comparar métodos")
    st.info("Bisección requiere cambio de signo; Punto Fijo y Aitken también requieren g(x).")
    section(1, "Ingresar datos")
    with st.container(border=True):
        ftext = st.text_input("f(x)", "x^3-x-2", key="cf")
        c1, c2, c3 = st.columns(3)
        a = c1.number_input("a", value=1.0, key="ca")
        b = c2.number_input("b", value=2.0, key="cb")
        x0 = c3.number_input("x₀", value=1.5, key="cx")
        automatic_g_button("cf", "cg", "cx", "ca", "cb")
        gtext = st.text_input("g(x) (opcional)", "(x+2)^(1/3)", key="cg")
        tolerance = tolerance_input("compare")
    section(2, "Resumen de lo ingresado")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            f_parsed = render_formula(ftext, "f")
            g_parsed = render_formula(gtext, "g") if gtext.strip() else None
            st.write(f"**Intervalo:** [{a:g}, {b:g}] · **x₀:** {x0:g}")
    with action:
        run = st.button("Comparar", type="primary", disabled=f_parsed is None,
                        key="run_cmp", use_container_width=True)
    if run:
        try:
            results = [bisection(f_parsed.high_precision, a, b, tolerance),
                       newton(f_parsed.high_precision, f_parsed.derivative_high_precision, x0, tolerance)]
            if g_parsed:
                point = fixed_point(f_parsed.high_precision, g_parsed.high_precision, x0, tolerance)
                results.append(point)
                sequence = [mp.mpf(x0)] + [row.x_next for row in point.history]
                if len(sequence) >= 3:
                    results.append(aitken_accelerate(f_parsed.high_precision, sequence, tolerance))
            st.session_state.cmp_result = results
        except ValueError as exc:
            st.error(str(exc))
    if "cmp_result" not in st.session_state:
        return
    results = st.session_state.cmp_result
    section(3, "Gráfico de aproximaciones")
    st.plotly_chart(approximation_plot(results), use_container_width=True)
    section(4, "Resultado")
    comparison = pd.DataFrame([{
        "Método": result.method, "Raíz": result.root, "Iteraciones": result.iterations,
        "Error final": result.history[-1].absolute_error if result.history else None,
        "Residuo": result.history[-1].residual if result.history else None,
        "Estado": "Convergió" if result.converged else "No convergió",
    } for result in results])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    section(5, "Lectura de la comparación")
    st.write("Compará cantidad de pasos, residuo y las hipótesis necesarias; no solamente la velocidad.")
    section(6, "Gráfico de convergencia")
    st.plotly_chart(convergence_plot(results), use_container_width=True)


def parse_evaluation_values(text: str) -> list[float]:
    """Convierte una lista escrita con comas o punto y coma en valores reales."""
    pieces = [piece.strip() for piece in text.replace(";", ",").split(",") if piece.strip()]
    if len(pieces) > 50:
        raise ValueError("Podés evaluar como máximo 50 valores por ejecución.")
    try:
        return [float(piece) for piece in pieces]
    except ValueError:
        raise ValueError("Los valores a evaluar deben ser números separados por comas.") from None


def lagrange_product_latex(result, i: int) -> str:
    """Representa L_i como producto, sin ocultar los valores de j."""
    xi = sp.latex(result.x_values[i])
    factors = []
    for j in result.bases[i].j_values:
        xj = sp.latex(result.x_values[j])
        factors.append(rf"\frac{{x-({xj})}}{{({xi})-({xj})}}")
    return r"\cdot".join(factors)


def lagrange_page() -> None:
    method_header("Construir función")
    explain(
        "Construir una función continua que pase exactamente por todos los nodos.",
        "Cuando se conocen valores discretos (xᵢ,yᵢ) y se necesitan valores intermedios.",
        "Construye una base Lᵢ para cada nodo y suma las contribuciones yᵢLᵢ(x).",
        r"P_n(x)=\sum_{i=0}^{n}y_iL_i(x),\qquad L_i(x)=\prod_{\substack{j=0\\j\ne i}}^n\frac{x-x_j}{x_i-x_j}",
    )
    section(1, "Ingresar nodos y valores a evaluar")
    with st.container(border=True):
        node_count = int(st.number_input("Cantidad de nodos", 2, 20, 3, key="lag_node_count"))
        routed = st.session_state.get("lag_route_data")
        default_x = ([float(parse_constant(value).value) for value in routed[0]] if routed
                     else [float(index) for index in range(node_count)])
        default_y = ([float(parse_constant(value).value) for value in routed[1]] if routed
                     else [1.0, 3.0, 0.0] + [0.0] * max(0, node_count - 3))
        default_nodes = pd.DataFrame({"i": range(node_count), "x_i": default_x, "y_i": default_y[:node_count]})
        nodes = st.data_editor(
            default_nodes, key=f"lag_nodes_{node_count}_{st.session_state.get('lag_route_version', 0)}",
            hide_index=True, use_container_width=True,
            disabled=["i"], num_rows="fixed",
            column_config={
                "i": st.column_config.NumberColumn("i", format="%d"),
                "x_i": st.column_config.NumberColumn("xᵢ", format="%.10g"),
                "y_i": st.column_config.NumberColumn("yᵢ=f(xᵢ)", format="%.10g"),
            },
        )
        evaluation_text = st.text_input(
            "Valores de x a evaluar en la función final",
            routed[2] if routed else "0, 0.5, 1, 1.5, 2",
            key=f"lag_evaluations_{st.session_state.get('lag_route_version', 0)}",
            help="Separalos con comas. Se permiten hasta 50 valores.",
        )
        reference_text = st.text_input(
            "Función original f(x) (opcional)", "", key="lag_reference",
            help="Si la conocés, se verifican los nodos y se crea un gráfico de comparación separado.",
        )
        if reference_text.strip():
            st.caption("La función es solo una referencia: los nodos siempre se interpolan, aunque no coincidan con ella.")
        show_components = st.checkbox("Mostrar también los términos yᵢ·Lᵢ(x) en el gráfico", value=True)
        result_format = st.radio("Formato de resultados", ["Fracciones / exacto", "Decimales"],
                                 horizontal=True, key="lag_format")
        display_digits = int(st.number_input("Decimales mostrados", 2, 100, 10, key="lag_digits",
                                             disabled=result_format != "Decimales"))
        if node_count > 10:
            st.warning("Un polinomio de grado alto puede oscilar entre nodos (fenómeno de Runge). Interpolar no garantiza una buena extrapolación.")

    section(2, "Resumen de lo ingresado")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            st.write(f"**Nodos:** {node_count} · **Grado máximo:** {node_count-1} · **Índices i,j:** 0 a {node_count-1}, con j≠i")
            st.dataframe(nodes, use_container_width=True, hide_index=True)
            reference_parsed = render_formula(reference_text, "f") if reference_text.strip() else None
    with action:
        run = st.button("Construir función", type="primary", key="run_lagrange", use_container_width=True)
        run = run or st.session_state.pop("console_run_lagrange", False)
    if run:
        try:
            if nodes[["x_i", "y_i"]].isnull().any().any():
                raise ValueError("Todos los nodos deben tener valores x_i e y_i.")
            evaluations = parse_evaluation_values(evaluation_text)
            result = build_lagrange(nodes["x_i"].tolist(), nodes["y_i"].tolist())
            if reference_text.strip() and reference_parsed is None:
                raise ValueError("Corregí la función original antes de construir el polinomio.")
            st.session_state.lagrange_result = (result, evaluations, display_digits, show_components,
                                                reference_parsed)
        except ValueError as exc:
            st.error(str(exc))
    if "lagrange_result" not in st.session_state:
        return
    result, evaluations, display_digits, show_components, reference_parsed = st.session_state.lagrange_result

    section(3, "Bases de Lagrange y valores de i, j")
    col_i, col_j = st.columns(2)
    selected_i = col_i.selectbox("Valor de i", list(range(len(result.x_values))), key=f"lag_i_{len(result.x_values)}")
    valid_j = list(result.bases[selected_i].j_values)
    selected_j = col_j.selectbox("Valor de j (j≠i)", valid_j, key=f"lag_j_{selected_i}")
    factor = basis_factor(result, selected_i, selected_j)
    st.markdown(f"**Factor seleccionado: i={selected_i}, j={selected_j}**")
    st.latex(rf"\frac{{x-x_{{{selected_j}}}}}{{x_{{{selected_i}}}-x_{{{selected_j}}}}}={sp.latex(factor)}")
    st.markdown(f"**Base completa L{selected_i}(x): utiliza j = {', '.join(map(str, valid_j))}**")
    st.latex(rf"L_{{{selected_i}}}(x)={lagrange_product_latex(result, selected_i)}")
    st.latex(rf"L_{{{selected_i}}}(x)={sp.latex(result.bases[selected_i].expanded)}")
    factor_rows = []
    for j in valid_j:
        factor_rows.append({
            "i": selected_i, "j": j,
            "x_i": float(result.x_values[selected_i]), "x_j": float(result.x_values[j]),
            "Factor": sp.sstr(basis_factor(result, selected_i, j)),
        })
    st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
    with st.expander("Mostrar todas las bases y todos los términos"):
        for basis in result.bases:
            st.markdown(f"#### Base L{basis.i}(x) - j = {list(basis.j_values)}")
            st.latex(rf"L_{{{basis.i}}}(x)={lagrange_product_latex(result, basis.i)}={sp.latex(basis.expanded)}")
            st.latex(rf"y_{{{basis.i}}}L_{{{basis.i}}}(x)={sp.latex(basis.weighted)}")

    section(4, "Función final construida")
    displayed_polynomial = sp.N(result.polynomial, display_digits) if result_format == "Decimales" else result.polynomial
    st.latex(rf"P_{{{result.degree}}}(x)={sp.latex(displayed_polynomial)}")
    st.success(f"Se construyó el único polinomio interpolante de grado ≤ {node_count-1} que pasa por todos los nodos.")

    section(5, "Gráfico aproximado y puntos evaluados")
    st.plotly_chart(lagrange_plot(result, evaluations, show_components), use_container_width=True)
    st.caption("La curva violeta interpola exactamente los nodos rojos. Los rombos verdes son valores evaluados en la función final.")

    if reference_parsed is not None:
        st.markdown("#### Verificación de la función original en cada nodo")
        verification_rows = []
        for x_value, y_value in zip(result.x_values, result.y_values):
            original = sp.simplify(reference_parsed.expression.subs(X, x_value))
            signed = sp.simplify(result.polynomial.subs(X, x_value) - original)
            absolute = sp.Abs(signed)
            relative = None if original == 0 else sp.simplify(absolute / sp.Abs(original))
            verification_rows.append({"x": str(x_value), "y ingresado": str(y_value),
                                      "f(x) original": str(original), "Coincide": sp.simplify(y_value-original) == 0,
                                      "Error firmado": str(signed), "Error absoluto": str(absolute),
                                      "Error relativo": "no definido" if relative is None else str(relative)})
        st.dataframe(pd.DataFrame(verification_rows), hide_index=True, use_container_width=True)
        st.markdown("#### Gráfico independiente de comparación")
        st.plotly_chart(lagrange_comparison_plot(result, reference_parsed.expression), use_container_width=True)
        st.caption("La línea celeste es la función original y la violeta discontinua es el polinomio. Que coincidan en los nodos no implica que coincidan entre ellos.")

    section(6, "Valores evaluados en la función final")
    evaluation_rows = []
    for value in evaluations:
        exact = sp.simplify(result.polynomial.subs(sp.Symbol("x", real=True), sp.Rational(str(value))))
        evaluation_rows.append({"x": value, "P(x)": (fixed_decimal(str(sp.N(exact, display_digits+8)), display_digits)
                                                        if result_format == "Decimales" else sp.sstr(exact))})
    evaluation_frame = pd.DataFrame(evaluation_rows)
    st.dataframe(evaluation_frame, use_container_width=True, hide_index=True)
    st.download_button("Descargar valores evaluados CSV", evaluation_frame.to_csv(index=False).encode("utf-8"),
                       "lagrange_valores_evaluados.csv", "text/csv")


def _set_derivative_example() -> None:
    st.session_state.update(der_source="Función", der_function="x^2", der_point="1", der_h="1/2",
                            der_xs="0, 1/2, 1, 3/2, 2", der_ys="0, 1/4, 1, 9/4, 4")
    st.session_state.pop("der_result", None)


def _clear_derivative() -> None:
    st.session_state.update(der_function="", der_point="", der_h="", der_xs="", der_ys="")
    st.session_state.pop("der_result", None)


def _exact_list(text: str) -> list[str]:
    values = [value.strip() for value in text.replace(";", ",").split(",") if value.strip()]
    if not values:
        raise ValueError("Ingresá valores separados por comas; podés usar fracciones como 1/2.")
    return values


def differentiation_page() -> None:
    method_header("Derivación numérica")
    explain("Estimar la pendiente f′(x₀) usando valores cercanos.",
            "Cuando hay una tabla o evaluar la derivada exacta es difícil.",
            "Calcula un cociente de cambios; h es la distancia horizontal entre nodos.",
            r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0-h)}{2h}")
    st.info("**h** es el tamaño de paso: una distancia menor suele reducir el error de truncamiento, pero datos redondeados pueden perder precisión.")
    section(1, "Ingresar datos")
    with st.container(border=True):
        buttons = st.columns(2)
        buttons[0].button("Cargar ejemplo", on_click=_set_derivative_example, use_container_width=True)
        buttons[1].button("Limpiar datos", on_click=_clear_derivative, use_container_width=True)
        source = st.radio("Datos disponibles", ["Función", "Tabla de nodos"], horizontal=True,
                          key="der_source", help="Elegí función si conocés f(x); tabla si solo tenés mediciones.")
        ftext = ""
        if source == "Función":
            ftext = st.text_input("f(x)", "x^2", key="der_function", help="Usá x como variable; se admiten potencias, raíces y funciones usuales.")
            columns = st.columns(2)
            point = columns[0].text_input("Punto x₀", "1", key="der_point", help="Entero, decimal o fracción.")
            h = columns[1].text_input("Paso h", "1/2", key="der_h", help="Debe ser positivo y ubicar los puntos dentro del dominio.")
            calculate_second = st.checkbox("Calcular también f''(x₀) con diferencia centrada", key="der_second")
            complete_table = False
        else:
            xs_text = st.text_input("Nodos x", "0, 1/2, 1, 3/2, 2", key="der_xs", help="Separados por comas y con el mismo paso.")
            ys_text = st.text_input("Valores y=f(x)", "0, 1/4, 1, 9/4, 4", key="der_ys", help="Uno por cada nodo x.")
            point = st.text_input("Punto x₀ (debe ser un nodo)", "1", key="der_point")
            h = None
            calculate_second = False
            complete_table = st.checkbox("Completar velocidad y aceleración en toda la tabla", key="der_complete_table",
                                         help="Usa diferencias centradas en el interior y progresiva/regresiva en los extremos.")
        labels = {"Diferencia hacia adelante": "forward", "Diferencia hacia atrás": "backward",
                  "Diferencia centrada": "centered"}
        label = st.selectbox("Esquema", list(labels), index=2, key="der_scheme",
                             help="Centrada es de orden 2; adelante y atrás son de orden 1.")
        show_decimal = st.checkbox("Mostrar aproximaciones decimales", False)
        digits = int(st.number_input("Decimales", 1, 15, 8, disabled=not show_decimal))
    section(2, "Resumen de lo ingresado")
    parsed = render_formula(ftext, "f") if source == "Función" and ftext.strip() else None
    st.write(f"**Fuente:** {source} · **x₀:** {point or '—'} · **Esquema:** {label}")
    if point and (parsed is not None or source == "Tabla de nodos"):
        st.success("🟢 Datos básicos presentes. Al ejecutar se verificarán dominio, nodos y separación.")
    else:
        st.error("🔴 Faltan datos. Completá la función o tabla y el punto x₀.")
    if label != "Diferencia centrada":
        st.warning("🟡 Esquema de primer orden: puede ser útil en un extremo, pero suele ser menos preciso.")
    run = st.button("Ejecutar derivación", type="primary", disabled=not point, use_container_width=True)
    run = run or st.session_state.pop("console_run_derivative", False)
    if run:
        try:
            if source == "Función":
                if parsed is None:
                    raise ValueError("Corregí la función antes de ejecutar.")
                result = differentiate_function(parsed.expression, point, h, labels[label])
                extra_result = differentiate_second_function(parsed.expression, point, h) if calculate_second else None
            else:
                result = differentiate_nodes(_exact_list(xs_text), _exact_list(ys_text), point, labels[label])
                extra_result = differentiate_table_all(_exact_list(xs_text), _exact_list(ys_text)) if complete_table else None
            st.session_state.der_result = (result, show_decimal, digits, extra_result)
        except ValueError as exc:
            st.error(f"No se pudo calcular: {exc}")
    if "der_result" not in st.session_state:
        return
    result, show_decimal, digits, extra_result = st.session_state.der_result
    section(3, "Gráfico principal")
    st.plotly_chart(differentiation_plot(result), use_container_width=True)
    st.caption("La recta naranja pasa por el punto seleccionado con la pendiente aproximada. Los puntos rojos son los valores usados en el cociente.")
    section(4, "Resultado explicado")
    formulas = {
        "Diferencia hacia adelante": r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0)}h",
        "Diferencia hacia atrás": r"f'(x_0)\approx\frac{f(x_0)-f(x_0-h)}h",
        "Diferencia centrada": r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0-h)}{2h}",
    }
    st.latex(formulas[result.scheme])
    numerator = (result.values[-1].y - result.values[0].y)
    denominator = result.h * (2 if result.scheme == "Diferencia centrada" else 1)
    st.latex(rf"f'({sp.latex(result.point)})\approx\frac{{{sp.latex(result.values[-1].y)}-({sp.latex(result.values[0].y)})}}{{{sp.latex(denominator)}}}={sp.latex(result.approximation)}")
    decimal = f" ≈ {sp.N(result.approximation, digits)}" if show_decimal else ""
    st.success(f"La pendiente aproximada es **{result.approximation}**{decimal}. El esquema es de orden O(h^{result.order}).")
    if extra_result is not None and source == "Función":
        st.markdown("#### Segunda derivada")
        st.latex(r"f''(x_0)\approx\frac{f(x_0+h)-2f(x_0)+f(x_0-h)}{h^2}")
        st.success(f"f''({extra_result.point}) ≈ **{extra_result.approximation}** · exacta: **{extra_result.exact_derivative}** · error absoluto: **{extra_result.absolute_error}**")
    elif extra_result is not None:
        st.markdown("#### Tabla completa de velocidad y aceleración")
        full_rows = [{"x/t": str(row.x), "posición": str(row.position),
                      "velocidad / f′": str(row.first_derivative),
                      "aceleración / f″": str(row.second_derivative)} for row in extra_result]
        st.dataframe(pd.DataFrame(full_rows), hide_index=True, use_container_width=True)
    if result.exact_derivative is not None:
        st.write(f"Derivada exacta en x₀: **{result.exact_derivative}**. Error firmado (aproximación − exacto): **{result.signed_error}**; absoluto: **{result.absolute_error}**; relativo: **{result.relative_error if result.relative_error is not None else 'no definido porque la derivada exacta es 0'}**.")
    else:
        st.info("Como se ingresó una tabla, no hay una derivada exacta de referencia ni errores respecto de ella.")
    section(5, "Tabla de valores utilizados")
    rows = [{"Rol": value.role, "x exacto": str(value.x), "f(x) exacto": str(value.y)} for value in result.values]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("Ver desarrollo y significado del orden"):
        st.write("Orden 1 significa que el error de truncamiento decrece aproximadamente como h; orden 2, como h², bajo condiciones de suavidad.")
    st.markdown(f"**Resumen:** con {result.scheme.lower()} y h={result.h}, la pendiente estimada en x₀={result.point} es {result.approximation}.")


def interpolate_function_page() -> None:
    method_header("Interpolar desde función")
    explain("Construir un polinomio que copie una función en nodos elegidos.",
            "Cuando conocés f(x) y querés estudiar una aproximación polinómica.",
            "Evalúa primero todos los nodos y luego aplica Lagrange sin reemplazarlos por muestras aproximadas.",
            r"y_i=f(x_i),\qquad P_n(x)=\sum_{i=0}^{n}y_iL_i(x)")
    section(1, "Ingresar primero la función y después los nodos")
    with st.container(border=True):
        ftext = st.text_input("Función original f(x)", "sin(x)", key="if_function",
                              help="La función se interpreta con el parser seguro y se evalúa exactamente cuando es posible.")
        nodes_text = st.text_input("Nodos x", "0, pi/4, pi/2", key="if_nodes",
                                   help="Entre 2 y 20 valores distintos separados por comas. Se admiten fracciones y pi.")
        show_decimal = st.checkbox("Mostrar también valores decimales", False, key="if_decimals")
        digits = int(st.number_input("Cantidad de decimales", 1, 15, 8, disabled=not show_decimal, key="if_digits"))
    section(2, "Resumen de lo ingresado")
    parsed = render_formula(ftext, "f") if ftext.strip() else None
    st.write(f"**Nodos solicitados:** {nodes_text or '—'}")
    if parsed is not None and nodes_text.strip():
        st.success("🟢 La función y la lista están presentes; al ejecutar se validará cada nodo.")
    else:
        st.error("🔴 Completá una función válida y al menos dos nodos.")
    if st.button("Evaluar nodos y construir", type="primary", disabled=parsed is None or not nodes_text.strip(), use_container_width=True):
        try:
            raw_nodes = _exact_list(nodes_text)
            if not 2 <= len(raw_nodes) <= 20:
                raise ValueError("Ingresá entre 2 y 20 nodos.")
            x_values = [calculate(value).exact for value in raw_nodes]
            y_values = []
            for value in x_values:
                evaluated = sp.simplify(parsed.expression.subs(X, value))
                if evaluated.is_real is False or evaluated.is_finite is False or evaluated.has(sp.nan, sp.zoo):
                    raise ValueError(f"La función no está definida en el nodo x={value}. Eliminá o corregí ese nodo.")
                y_values.append(evaluated)
            result = build_lagrange(x_values, y_values)
            st.session_state.if_result = (result, parsed, show_decimal, digits)
        except (ValueError, TypeError, SyntaxError) as exc:
            st.error(f"No se pudo construir: {exc}")
    if "if_result" not in st.session_state:
        return
    result, parsed, show_decimal, digits = st.session_state.if_result
    section(3, "Nodos evaluados")
    rows = []
    for x_value, y_value in zip(result.x_values, result.y_values):
        row = {"x exacto": str(x_value), "f(x) exacto": str(y_value)}
        if show_decimal:
            row.update({"x decimal": str(sp.N(x_value, digits)), "f(x) decimal": str(sp.N(y_value, digits))})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    section(4, "Polinomio calculado")
    st.latex(rf"P_{{{result.degree}}}(x)={sp.latex(result.polynomial)}")
    st.success("Todos los nodos fueron evaluados en la función original antes de interpolar.")
    section(5, "Gráfico del polinomio")
    st.plotly_chart(lagrange_plot(result), use_container_width=True)
    st.caption("La curva violeta es el polinomio calculado y los puntos rojos son los valores exactos f(xᵢ).")
    section(6, "Comparación con la función original")
    st.plotly_chart(lagrange_comparison_plot(result, parsed.expression), use_container_width=True)
    st.caption("Este gráfico está separado para distinguir claramente la función original del interpolante.")
    with st.expander("¿Por qué pueden diferir entre nodos?"):
            st.write("Lagrange obliga al polinomio a coincidir en cada nodo, no en todos los puntos. La elección y distribución de nodos determina la calidad de la aproximación entre ellos.")


def integration_page() -> None:
    method_header("Integración numérica")
    st.write("Las reglas de Newton–Cotes reemplazan localmente la función por polinomios fáciles de integrar y suman sus aportes en cada subintervalo.")
    section(1, "Ingresar datos")
    method_labels = {
        "Rectángulo por punto medio": "midpoint",
        "Rectángulo izquierdo": "left_rectangle",
        "Rectángulo derecho": "right_rectangle",
        "Regla del trapecio": "trapezoid",
        "Regla de Simpson 1/3": "simpson_13",
        "Regla de Simpson 3/8": "simpson_38",
    }
    with st.container(border=True):
        ftext = st.text_input("Función f(x)", "x^2", key="int_function",
                              help="Ejemplos: x^2, sin(x), exp(x).")
        limits = st.columns(3)
        a_text = limits[0].text_input("Límite inferior a", value="0", key="int_a",
                                      help="Acepta expresiones como -pi, -√(2) o 1/3.")
        b_text = limits[1].text_input("Límite superior b", value="1", key="int_b",
                                      help="Acepta expresiones como pi, π/2, sqrt(2) o e.")
        n = int(limits[2].number_input("Subintervalos n", min_value=1, value=2, step=1, key="int_n"))
        selected_label = st.selectbox("Método de integración", list(method_labels), key="int_method")
        decimals = int(st.number_input("Decimales mostrados", 6, 15, 6, key="int_decimals",
                                       help="En integración se muestran como mínimo 6 cifras decimales."))

    section(2, "Resumen y validación")
    summary, action = st.columns([4, 1])
    with summary:
        with st.container(border=True):
            parsed = render_formula(ftext, "f")
            try:
                parsed_a = parse_constant(a_text)
                parsed_b = parse_constant(b_text)
                a, b = parsed_a.value, parsed_b.value
                st.latex(rf"a={sp.latex(parsed_a.expression)},\qquad b={sp.latex(parsed_b.expression)}")
            except ValueError as exc:
                parsed_a = parsed_b = None
                a = b = None
                st.error(str(exc))
            restriction = "Sin restricción adicional sobre n."
            if method_labels[selected_label] == "simpson_13":
                restriction = "n debe ser par; n=2 es la fórmula simple."
            elif method_labels[selected_label] == "simpson_38":
                restriction = "n debe ser múltiplo de 3; n=3 es la fórmula simple."
            elif method_labels[selected_label] == "trapezoid":
                restriction = "n=1 es la fórmula simple."
            if parsed_a is not None and parsed_b is not None:
                st.write(f"**Intervalo decimal:** [{fixed_decimal(a, decimals)}, {fixed_decimal(b, decimals)}] · "
                         f"**n:** {n} · **h:** {fixed_decimal((b-a)/n, decimals)}")
            st.caption(restriction)
    with action:
        run = st.button("Calcular", type="primary", disabled=parsed is None or parsed_a is None or parsed_b is None,
                        key="run_integration", use_container_width=True)
        run = run or st.session_state.pop("console_run_integration", False)
    if run:
        try:
            result = integrate_newton_cotes(parsed.high_precision, a, b, n,
                                            method_labels[selected_label])
            st.session_state.integration_result = (parsed, result, decimals, ftext, a_text, b_text, n, selected_label,
                                                   parsed_a.expression, parsed_b.expression)
        except ValueError as exc:
            st.session_state.pop("integration_result", None)
            st.error(str(exc))

    with st.expander("Abrir tabla para evaluar f(x)"):
        st.caption("Editá la columna x y agregá tantas filas como necesites. Se aceptan decimales, fracciones, π y raíces.")
        if "int_eval_input" not in st.session_state:
            default_values = ["0", "1/4", "1/2", "3/4", "1"]
            st.session_state.int_eval_input = pd.DataFrame({"x": default_values})
        evaluation_input = st.data_editor(
            st.session_state.int_eval_input,
            num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={"x": st.column_config.TextColumn("x", help="Ejemplos: 0.25, 1/2, pi/2")},
            key="int_eval_editor",
        )
        evaluate = st.button("Evaluar función", key="int_eval_button", disabled=parsed is None,
                             use_container_width=True)
        if evaluate:
            try:
                values = evaluation_input["x"].tolist() if "x" in evaluation_input else []
                rows = function_evaluation_rows(parsed, values, decimals)
                st.session_state.int_eval_result = (ftext, decimals, rows)
            except ValueError as exc:
                st.session_state.pop("int_eval_result", None)
                st.error(str(exc))
        if "int_eval_result" in st.session_state:
            evaluated_function, evaluated_decimals, rows = st.session_state.int_eval_result
            if (ftext, decimals) == (evaluated_function, evaluated_decimals):
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("Cambiaste la función o los decimales. Presioná «Evaluar función» para actualizar la tabla.")

    if "integration_result" not in st.session_state:
        return
    parsed, result, decimals, executed_f, executed_a, executed_b, executed_n, executed_method, a_expr, b_expr = st.session_state.integration_result
    if (ftext, a_text, b_text, n, selected_label) != (executed_f, executed_a, executed_b, executed_n, executed_method):
        st.info("Cambiaste los datos. Presioná «Calcular» para actualizar el resultado.")
        return

    section(3, "Gráfico del método")
    st.plotly_chart(integration_plot(parsed.numeric, result), use_container_width=True)
    st.caption("Las líneas verticales delimitan los subintervalos y los puntos rojos son las evaluaciones utilizadas.")
    section(4, "Resultado")
    method_key = method_labels[executed_method]
    h_expr, applied_formula, decimal_formula = integration_development(
        parsed.expression, a_expr, b_expr, result.n, method_key, result, decimals,
    )
    metrics = st.columns(4)
    metrics[0].metric("Integral aproximada", fixed_decimal(result.approximation, decimals))
    metrics[1].metric("Método", result.method)
    metrics[2].metric("Subintervalos", result.n)
    h_exact = sp.sstr(h_expr)
    metrics[3].metric("h = (b−a)/n", f"{h_exact} = {fixed_decimal(result.h, decimals)}")
    st.success(f"La aproximación obtenida es {fixed_decimal(result.approximation, decimals)}.")
    with st.expander("Ver fórmula completa aplicada", expanded=True):
        st.latex(rf"h=\frac{{{sp.latex(b_expr)}-({sp.latex(a_expr)})}}{{{result.n}}}="
                 rf"{sp.latex(h_expr)}={fixed_decimal(result.h, decimals)}")
        st.markdown("**Sustitución exacta de todos los nodos y pesos:**")
        st.latex(applied_formula)
        st.markdown("**Sustitución decimal:**")
        st.latex(decimal_formula)
    try:
        gauss_reference = integrate_gauss_legendre(parsed.high_precision, a, b, GAUSS_REFERENCE_NODES)
        absolute_error = abs(result.approximation - gauss_reference)
        relative_error = None if gauss_reference == 0 else absolute_error / abs(gauss_reference)
        estimated_error = gauss_reference - result.approximation
        comparison = st.columns(5)
        comparison[0].metric("Cuadratura de Gauss", fixed_decimal(gauss_reference, decimals))
        comparison[1].metric("Nodos de Gauss", GAUSS_REFERENCE_NODES)
        comparison[2].metric("Error estimado G−Iₙ", fixed_decimal(estimated_error, decimals))
        comparison[3].metric("|Error estimado|", fixed_decimal(absolute_error, decimals))
        comparison[4].metric("Error relativo %", "—" if relative_error is None else fixed_decimal(100 * relative_error, decimals))
        with st.expander("¿Cómo se calcula la referencia de Gauss?"):
            st.write(f"Se usa cuadratura de Gauss–Legendre con **{GAUSS_REFERENCE_NODES} nodos** interiores. "
                     "Los nodos no están igualmente espaciados: son las raíces del polinomio de Legendre correspondiente y se transforman al intervalo [a,b].")
            st.latex(r"G_m=\frac{b-a}{2}\sum_{i=1}^{m}w_i\,f\!\left(\frac{a+b}{2}+\frac{b-a}{2}t_i\right)")
            st.write(f"Con m={GAUSS_REFERENCE_NODES}, la regla es exacta para polinomios de grado hasta "
                     f"{2 * GAUSS_REFERENCE_NODES - 1}. Para otras funciones es una referencia numérica de alta precisión, no una integral exacta.")
        st.caption("El error mostrado es estimado: E ≈ G₃₂ − Iₙ. Su signo indica si Newton–Cotes queda por encima o por debajo de la referencia de Gauss.")
    except (TypeError, ValueError, OverflowError):
        gauss_reference = None
        st.info("No se pudo calcular la referencia con Gauss–Legendre; la aproximación de Newton–Cotes sigue siendo válida.")
    with st.expander("Comparar métodos y valores de n"):
        st.caption("Genera la tabla comparativa solicitada por varios ejercicios de la guía.")
        compare_methods = st.multiselect("Métodos", list(method_labels), default=list(method_labels), key="int_compare_methods")
        compare_ns = st.text_input("Valores de n separados por comas", str(n), key="int_compare_ns")
        if st.button("Construir comparación", key="int_compare"):
            try:
                counts = sorted({int(value.strip()) for value in compare_ns.split(",") if value.strip()})
                if not counts or any(value <= 0 for value in counts):
                    raise ValueError("Ingresá enteros positivos para n.")
                reference_value = integrate_gauss_legendre(parsed.high_precision, a, b, GAUSS_REFERENCE_NODES)
                rows = []
                for method_name in compare_methods:
                    for count in counts:
                        try:
                            candidate = integrate_newton_cotes(parsed.high_precision, a, b, count,
                                                               method_labels[method_name])
                        except ValueError:
                            continue
                        error = abs(candidate.approximation - reference_value)
                        relative = None if reference_value == 0 else 100 * error / abs(reference_value)
                        rows.append({"Método": method_name, "n": count,
                                     "Aproximación": fixed_decimal(candidate.approximation, decimals),
                                     "Error estimado vs. Gauss": fixed_decimal(error, decimals),
                                     "Error relativo %": "—" if relative is None else fixed_decimal(relative, decimals)})
                if not rows:
                    raise ValueError("Ninguna combinación cumple las restricciones de los métodos elegidos.")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            except (TypeError, ValueError) as exc:
                st.error(str(exc))
    section(5, "Puntos y pesos utilizados")
    point_rows = [{"i": index, "xᵢ": fixed_decimal(point.x, decimals),
                   "f(xᵢ)": fixed_decimal(point.fx, decimals), "peso": point.weight}
                  for index, point in enumerate(result.points)]
    st.dataframe(pd.DataFrame(point_rows), hide_index=True, use_container_width=True)
    section(6, "Fundamento teórico")
    st.dataframe(pd.DataFrame([
        {"Método": "Punto medio", "Grado": "0", "Error": "O(h²)", "Restricción en n": "Entero positivo"},
        {"Método": "Rectángulo izquierdo", "Grado": "0", "Error": "O(h)", "Restricción en n": "Entero positivo"},
        {"Método": "Rectángulo derecho", "Grado": "0", "Error": "O(h)", "Restricción en n": "Entero positivo"},
        {"Método": "Trapecio", "Grado": "1", "Error": "O(h²)", "Restricción en n": "Entero positivo"},
        {"Método": "Simpson 1/3", "Grado": "2", "Error": "O(h⁴)", "Restricción en n": "Par"},
        {"Método": "Simpson 3/8", "Grado": "3", "Error": "O(h⁴)", "Restricción en n": "Múltiplo de 3"},
    ]), hide_index=True, use_container_width=True)
    theory = [
        ("Rectángulo por punto medio", "Aproxima cada tramo por una constante: la altura en su punto medio.",
         r"I\approx h\sum_{i=0}^{n-1}f\!\left(a+\left(i+\frac12\right)h\right)", "0 (constante)", "O(h^2)", "n entero positivo."),
        ("Rectángulo izquierdo", "Usa como altura el valor de la función en el extremo izquierdo de cada tramo.",
         r"I\approx h\sum_{i=0}^{n-1}f(x_i)", "0 (constante)", "O(h)", "n entero positivo."),
        ("Rectángulo derecho", "Usa como altura el valor de la función en el extremo derecho de cada tramo.",
         r"I\approx h\sum_{i=1}^{n}f(x_i)", "0 (constante)", "O(h)", "n entero positivo."),
        ("Regla del trapecio", "Une los extremos de cada tramo con una recta y suma las áreas de los trapecios.",
         r"I\approx\frac h2\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]", "1 (lineal)", "O(h^2)", "n entero positivo; n=1 es la regla simple."),
        ("Regla de Simpson 1/3", "Ajusta parábolas sobre pares de subintervalos.",
         r"I\approx\frac h3\left[f(x_0)+4\sum_{i\,impar}f(x_i)+2\sum_{i\,par}f(x_i)+f(x_n)\right]", "2 (cuadrático)", "O(h^4)", "n par; n=2 es la regla simple."),
        ("Regla de Simpson 3/8", "Ajusta polinomios cúbicos sobre grupos de tres subintervalos.",
         r"I\approx\frac{3h}{8}\left[f(x_0)+3\sum_{3\nmid i}f(x_i)+2\sum_{3\mid i}f(x_i)+f(x_n)\right]", "3 (cúbico)", "O(h^4)", "n múltiplo de 3; n=3 es la regla simple."),
    ]
    tabs = st.tabs([item[0] for item in theory])
    for tab, (name, description, formula, degree, order, rule) in zip(tabs, theory):
        with tab:
            st.write(description)
            st.latex(formula)
            cols = st.columns(3)
            cols[0].markdown(f"**Grado:** {degree}")
            cols[1].markdown(f"**Error compuesto:** {order}")
            cols[2].markdown(f"**Restricción:** {rule}")


def home_page() -> None:
    method_header("Inicio")
    st.write("Todas las pantallas siguen el mismo recorrido: datos → resumen → gráfico → resultado → tabla → convergencia.")
    st.info("¿Estás resolviendo la guía? Abrí **PDF · Ejercicios** en la navegación para cargar sus datos automáticamente.")
    st.subheader("¿Qué método elijo?")
    st.markdown("""
- **Tengo un intervalo con cambio de signo → Bisección.**
- **Tengo una función g(x) contractiva → Punto Fijo.**
- **Tengo un valor inicial y puedo derivar → Newton–Raphson.**
- **Tengo una sucesión que converge lentamente → Aitken.**
- **Tengo nodos discretos → Interpolación de Lagrange o Derivación numérica.**
- **Quiero aproximar el área bajo una curva → Integración Newton–Cotes.**
""")
    cards = [
        ("±", "1. Conceptos básicos y errores", "Interpretá error absoluto, relativo, residuo y convergencia."),
        ("½", "Bisección", "Confiable; requiere un intervalo con cambio de signo."),
        ("●", "Punto Fijo", "Muestra la convergencia con un diagrama de telaraña."),
        ("╱╲", "Newton–Raphson", "Rápido, pero depende de x₀ y de la derivada."),
        ("↗", "Aitken Δ²", "Acelera una sucesión de Punto Fijo que ya converge."),
        ("VS", "Comparación", "Contrasta errores, raíces y cantidad de pasos."),
        ("Σ", "Interpolación de Lagrange", "Interpola hasta 20 nodos con bases de Lagrange."),
        ("f′", "Derivación numérica", "Estima pendientes con diferencias adelante, atrás o centrada."),
        ("∫", "Integración Newton–Cotes", "Aproxima integrales con punto medio, trapecio y Simpson."),
        ("🗓", "Próximamente", "Monte Carlo y ecuaciones diferenciales; todavía no implementados."),
        ("📖", "Teoría", "Explica las hipótesis detrás de cada algoritmo."),
    ]
    details = {
        "1. Conceptos básicos y errores": (r"E_a=|x_n-x_{n-1}|,\quad r_n=|f(x_n)|", "El error compara aproximaciones; el residuo mide cuánto falta para satisfacer la ecuación. Son preguntas distintas."),
        "Bisección": (r"c_n=\frac{a_n+b_n}{2}", "Se apoya en continuidad y cambio de signo. Es robusta porque nunca abandona el intervalo que encierra la raíz."),
        "Punto Fijo": (r"x_{n+1}=g(x_n)", "Busca una función contractiva que acerque puntos. La condición |g′|<1 explica por qué las distancias tienden a reducirse."),
        "Newton–Raphson": (r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}", "Reemplaza localmente la curva por su tangente. Puede ser muy rápido, pero una derivada casi nula o un x₀ inadecuado lo desestabilizan."),
        "Aitken Δ²": (r"x_n^*=x_n-\frac{(\Delta x_n)^2}{\Delta^2x_n}", "Usa tres términos consecutivos para estimar el límite de una sucesión que ya converge; no convierte una sucesión divergente en convergente."),
        "Interpolación de Lagrange": (r"P_n(x)=\sum_{i=0}^ny_iL_i(x)", "Cada base vale uno en su propio nodo y cero en los demás. Por eso la suma reproduce exactamente todos los datos."),
        "Derivación numérica": (r"f'(x_0)\approx\frac{f(x_0+h)-f(x_0-h)}{2h}", "Aproxima una pendiente con cambios cercanos. h controla la escala de observación y el esquema centrado tiene orden dos."),
        "Integración Newton–Cotes": (r"\int_a^b f(x)\,dx\approx\sum_i w_i f(x_i)", "Punto medio, trapecio y Simpson sustituyen la curva por polinomios locales y suman áreas más simples."),
        "Próximamente": (r"\text{Monte Carlo y EDO}", "Estos contenidos permanecen como hoja de ruta y no están implementados en esta etapa."),
        "Comparación": (r"\text{hipótesis + error + costo}", "No hay un método universalmente mejor: conviene comparar requisitos, robustez, residuo y cantidad de pasos."),
        "Teoría": (r"\text{modelo}\to\text{método}\to\text{interpretación}", "Las fórmulas son herramientas: comprender sus hipótesis permite reconocer cuándo un resultado numérico es confiable."),
    }
    for start in range(0, len(cards), 3):
        for column, (icon, title, text) in zip(st.columns(3), cards[start:start + 3]):
            with column:
                with st.container(border=True):
                    st.markdown(f"### {icon} {title}")
                    st.write(text)
                    formula, reason = details[title]
                    with st.expander("Ver función, razón y teoría"):
                        st.latex(formula)
                        st.write(reason)


def laboratory_page() -> None:
    method_header("Laboratorio")
    experiment = st.selectbox("Experimento", ["A — distintos x₀ en Newton", "B — distintas g(x)",
                                                   "C — tolerancia en Bisección", "D — Aitken sobre Punto Fijo"])
    texts = {"A — distintos x₀ en Newton": "Observá cómo cambia la trayectoria de las tangentes.",
             "B — distintas g(x)": "Compará cobwebs y max |g′(x)|.",
             "C — tolerancia en Bisección": "Relacioná decimales y cantidad de iteraciones.",
             "D — Aitken sobre Punto Fijo": "Compará la sucesión original y el atajo Δ²."}
    st.info(texts[experiment])


def theory_page() -> None:
    method_header("Teoría")
    st.markdown("""
### El recorrido numérico
**Modelo → método → iteraciones → visualización → error → interpretación**

### Existencia y convergencia son conceptos diferentes
Bisección utiliza continuidad y cambio de signo. Punto Fijo estudia si g conserva el intervalo y contrae distancias. Newton depende de la derivada y del valor inicial. Aitken acelera una sucesión que ya existe.

### Construcción de funciones con Lagrange
Cada base Lᵢ(x) vale 1 en xᵢ y 0 en los demás nodos. La suma ponderada Pₙ(x)=ΣyᵢLᵢ(x) produce el único polinomio de grado ≤n que interpola los n+1 nodos distintos.

### Integración numérica con Newton–Cotes
Punto medio y trapecio tienen error compuesto O(h²). Simpson 1/3 y 3/8 alcanzan O(h⁴), pero requieren respectivamente una cantidad par de subintervalos y un múltiplo de tres.

### Criterios de detención
- **Error absoluto:** cambio entre aproximaciones consecutivas.
- **Error relativo:** cambio comparado con el tamaño de la aproximación.
- **Residuo:** cuánto falta para cumplir f(x)=0.
""")


def _route_console_command(text: str) -> None:
    """Fill the guided page selected by a console prompt and request one automatic run."""
    solve_command(text)  # Validate all required parameters before changing page state.
    command, values = parse_command(text)
    digits = int(values.get("tol", "8"))
    maximum = int(values.get("max", "100"))
    if not 1 <= digits <= 100 or not 1 <= maximum <= 2000:
        raise ValueError("Usá tol entre 1 y 100 decimales y max entre 1 y 2000 iteraciones.")

    routes = {
        "newton": ("╱╲  Newton–Raphson", "nw"),
        "bisection": ("½  Bisección", "bi"),
        "fixed": ("●  Punto Fijo", "pf"),
        "integrate": ("∫  Integración Newton–Cotes", "integration"),
        "derivative": ("f′  Derivación numérica", "derivative"),
        "lagrange": ("Σ  Construir función", "lagrange"),
    }
    target, pending = routes[command]
    if command == "newton":
        st.session_state.update(nw_f=values["f"], nw_x0=float(parse_constant(values["x0"]).value),
                                nwtol_digits=digits, nwn=maximum)
    elif command == "bisection":
        st.session_state.update(bi_f=values["f"], bi_a=float(parse_constant(values["a"]).value),
                                bi_b=float(parse_constant(values["b"]).value),
                                bitol_digits=digits, bin=maximum)
    elif command == "fixed":
        x0 = float(parse_constant(values["x0"]).value)
        st.session_state.update(pf_f=values["f"], pf_g=values["g"], pf_x0=x0,
                                pf_a=float(parse_constant(values.get("a", str(x0 - 1))).value),
                                pf_b=float(parse_constant(values.get("b", str(x0 + 1))).value),
                                pftol_digits=digits, pfn=maximum)
    elif command == "integrate":
        labels = {"midpoint": "Rectángulo por punto medio", "left_rectangle": "Rectángulo izquierdo",
                  "right_rectangle": "Rectángulo derecho", "trapezoid": "Regla del trapecio",
                  "simpson_13": "Regla de Simpson 1/3", "simpson_38": "Regla de Simpson 3/8"}
        method = values.get("method", "trapezoid")
        if method not in labels:
            raise ValueError("Elegí un método de integración válido.")
        st.session_state.update(int_function=values["f"], int_a=values["a"], int_b=values["b"],
                                int_n=int(values.get("n", "1")), int_method=labels[method])
    elif command == "derivative":
        labels = {"forward": "Diferencia hacia adelante", "backward": "Diferencia hacia atrás",
                  "centered": "Diferencia centrada"}
        method = values.get("method", "centered")
        if method not in labels:
            raise ValueError("Elegí diferencia hacia adelante, hacia atrás o centrada.")
        st.session_state.update(der_source="Función", der_function=values["f"],
                                der_point=values["x0"], der_h=values["h"], der_scheme=labels[method])
    else:
        xs = [item.strip() for item in values["xs"].split(",") if item.strip()]
        ys = [item.strip() for item in values["ys"].split(",") if item.strip()]
        if len(xs) != len(ys) or not 2 <= len(xs) <= 20:
            raise ValueError("xs e ys deben contener entre 2 y 20 valores y tener igual longitud.")
        st.session_state["lag_route_data"] = (xs, ys, values.get("at", ", ".join(xs)))
        st.session_state["lag_route_version"] = st.session_state.get("lag_route_version", 0) + 1
        st.session_state["lag_node_count"] = len(xs)

    st.session_state["navigation"] = target
    st.session_state[f"console_run_{pending}"] = True
    st.session_state.pop("console_error", None)
    st.session_state["console_show_help"] = False


def _submit_console_command() -> None:
    """Route during Streamlit's callback phase, before navigation is instantiated."""
    command = st.session_state.get("console_command", "").strip()
    if command.lower() in {"help", "ayuda", "?"}:
        st.session_state["console_show_help"] = True
        st.session_state.pop("console_error", None)
        return
    try:
        _route_console_command(command)
    except ValueError as exc:
        st.session_state["console_error"] = str(exc)


def console_page() -> None:
    method_header("Consola")
    st.write("Ingresá el ejercicio en una línea. La consola reconoce la función, los datos y el método, y usa los mismos motores seguros que las pantallas guiadas.")
    console_tab, dictionary_tab = st.tabs([">_ Consola", "📘 Diccionario de prompts"])

    with console_tab:
        st.code("newton f=x^3-x-2 x0=1.5 tol=8 max=100", language=None)
        with st.form("exercise_console_form"):
            command = st.text_input(
                "Comando",
                key="console_command",
                placeholder="newton f=x^3-x-2 x0=1.5 tol=8 max=100",
                help="Si una expresión contiene espacios, encerrala entre comillas.",
            )
            st.form_submit_button("Ejecutar", type="primary", use_container_width=True,
                                  on_click=_submit_console_command)
        if error := st.session_state.get("console_error"):
            st.error(error)
        if st.session_state.get("console_show_help"):
            st.info("Abrí **Diccionario de prompts** para consultar todos los comandos y parámetros.")

    with dictionary_tab:
        st.subheader("Cómo se escribe un prompt")
        st.markdown("El formato es `comando nombre=valor`. Separá cada dato con un espacio. Las potencias aceptan `^`; las listas usan comas; si un valor contiene espacios, escribilo entre comillas.")
        st.caption("Usá el botón de copiar del bloque y pegá una línea completa en la consola.")
        st.code("\n".join(example for _, example, _ in COMMAND_HELP), language=None)
        st.dataframe(pd.DataFrame([
            {"Comando": name, "Prompt para copiar": example, "Qué resuelve": description}
            for name, example, description in COMMAND_HELP
        ]), hide_index=True, use_container_width=True)
        st.markdown("También podés pegar un diccionario completo:")
        st.code("{'command': 'newton', 'f': 'x^3-x-2', 'x0': 1.5, 'tol': 8, 'max': 100}", language=None)
        st.markdown("""
**Parámetros comunes**

- `f`: función de `x`. Ejemplos: `x^3-x-2`, `sin(x)`, `exp(-x)-x`.
- `x0`: valor inicial; `a` y `b`: extremos del intervalo.
- `tol`: cantidad de decimales de tolerancia, no el valor decimal. `tol=8` significa ε = 10⁻⁸.
- `max`: máximo de iteraciones.
- `g`: función de iteración para punto fijo.
- `n`: cantidad de subintervalos para integración.
- `xs`, `ys`: listas de nodos separadas por comas; `at` evalúa el interpolante y es opcional.

**Valores de `method`**

- Integración: `midpoint`, `left_rectangle`, `right_rectangle`, `trapezoid`, `simpson_13`, `simpson_38`.
- Derivación: `forward`, `backward`, `centered`.

**Funciones y constantes admitidas**

`sin`, `cos`, `tan`, `sqrt`, `root`, `exp`, `log`/`ln`, `log10`, `abs`, `pi`, `e`. También podés escribir `help` en la consola para volver a esta guía.
""")


def _open_pdf_exercise(exercise: dict) -> None:
    """Load a course exercise into its matching widgets and navigate there."""
    st.session_state.update(exercise["values"])
    st.session_state["navigation"] = exercise["target"]
    for result_key in ("bi_result", "pf_result", "nw_result", "ai_result", "der_result",
                       "if_result", "integration_result"):
        st.session_state.pop(result_key, None)


def pdf_exercises_page() -> None:
    method_header("Ejercicios del PDF")
    st.write("Estos ejercicios fueron transcritos del capítulo I del libro auxiliar. Elegí uno para abrir el método con la función, intervalo, paso y valor inicial ya cargados.")
    st.caption("Fuente: “Modelado y Simulación”, segunda edición 2026. La página indicada corresponde a la numeración impresa del libro.")
    sections = list(dict.fromkeys(exercise["section"] for exercise in PDF_EXERCISES))
    selected_section = st.selectbox("Tema", sections, key="pdf_section")
    choices = [exercise for exercise in PDF_EXERCISES if exercise["section"] == selected_section]
    selected_title = st.selectbox("Ejercicio", [exercise["title"] for exercise in choices], key="pdf_exercise")
    exercise = next(item for item in choices if item["title"] == selected_title)
    with st.container(border=True):
        st.markdown(f"### {exercise['title']}")
        st.write(f"**Página:** {exercise['page']} · **Herramienta:** {NAVIGATION[exercise['target']]}")
        st.dataframe(pd.DataFrame([{"Dato": key, "Valor cargado": str(value)}
                                   for key, value in exercise["values"].items()]),
                     hide_index=True, use_container_width=True)
        st.button("Abrir y resolver", type="primary", use_container_width=True,
                  on_click=_open_pdf_exercise, args=(exercise,))
    st.info("La aplicación guía el procedimiento y muestra iteraciones, gráficos y errores. El resultado debe acompañarse con la interpretación y las hipótesis del método.")


NAVIGATION = {"⌂  Inicio": "Inicio", ">_  Consola": "Consola", "½  Bisección": "Bisección", "●  Punto Fijo": "Punto Fijo",
              "╱╲  Newton–Raphson": "Newton-Raphson", "↗  Aitken Δ²": "Aitken",
              "VS  Comparar métodos": "Comparar métodos", "⚗  Laboratorio": "Laboratorio",
              "Σ  Construir función": "Construir función",
              "f→Σ  Interpolar desde función": "Interpolar desde función",
              "f′  Derivación numérica": "Derivación numérica",
              "∫  Integración Newton–Cotes": "Integración numérica",
              "PDF  Ejercicios": "Ejercicios del PDF",
              "📖  Teoría": "Teoría"}
st.sidebar.title("MODELADO Y SIMULACIÓN")
calculator_drawer()
floating_math_keyboard()
st.sidebar.toggle("Modo avanzado", value=False, key="advanced_mode",
                  help="Muestra controles secundarios; el modo guiado es el predeterminado.")
st.sidebar.caption("🟢 cumplida · 🟡 dudosa · 🔴 no cumplida")
page = NAVIGATION[st.sidebar.radio("Navegación", list(NAVIGATION), key="navigation")]

if page == "Inicio": home_page()
elif page == "Consola": console_page()
elif page == "Bisección": bisection_page()
elif page == "Punto Fijo": fixed_page()
elif page == "Newton-Raphson": newton_page()
elif page == "Aitken": aitken_page()
elif page == "Comparar métodos": comparison_page()
elif page == "Construir función": lagrange_page()
elif page == "Interpolar desde función": interpolate_function_page()
elif page == "Derivación numérica": differentiation_page()
elif page == "Integración numérica": integration_page()
elif page == "Ejercicios del PDF": pdf_exercises_page()
elif page == "Laboratorio": laboratory_page()
else: theory_page()
