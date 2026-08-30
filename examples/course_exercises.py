"""Ejercicios precargados del libro auxiliar (capítulo I)."""

PDF_EXERCISES = [
    # Búsqueda de raíces, páginas 17–18.
    {"section": "Bisección", "title": "3.a · √x − cos(x)", "page": 17,
     "target": "½  Bisección", "values": {"bi_f": "sqrt(x)-cos(x)", "bi_a": 0.0, "bi_b": 1.0}},
    {"section": "Bisección", "title": "3.b · x − 2⁻ˣ", "page": 17,
     "target": "½  Bisección", "values": {"bi_f": "x-2^(-x)", "bi_a": 0.0, "bi_b": 1.0}},
    {"section": "Bisección", "title": "3.c · eˣ − x² + 3x − 2", "page": 17,
     "target": "½  Bisección", "values": {"bi_f": "exp(x)-x^2+3*x-2", "bi_a": 0.0, "bi_b": 1.0}},
    {"section": "Bisección", "title": "3.d · intervalo [−3,−2]", "page": 17,
     "target": "½  Bisección", "values": {"bi_f": "2*x*cos(x)-(x+1)^2", "bi_a": -3.0, "bi_b": -2.0}},
    {"section": "Bisección", "title": "4.a · polinomio en [−2,−1]", "page": 17,
     "target": "½  Bisección", "values": {"bi_f": "x^4-2*x^3-4*x^2+4*x+4", "bi_a": -2.0, "bi_b": -1.0}},
    # Punto fijo, página 18. Se expresa f(x)=g(x)-x para verificar el residuo.
    {"section": "Punto Fijo", "title": "3 · g(x)=e⁻ˣ", "page": 18,
     "target": "●  Punto Fijo", "values": {"pf_f": "exp(-x)-x", "pf_g": "exp(-x)", "pf_x0": 0.0, "pf_a": 0.0, "pf_b": 1.0}},
    {"section": "Punto Fijo", "title": "4 · x³−x−1", "page": 18,
     "target": "●  Punto Fijo", "values": {"pf_f": "x^3-x-1", "pf_g": "root(x+1,3)", "pf_x0": 1.0, "pf_a": 1.0, "pf_b": 2.0}},
    # Aitken, página 18.
    {"section": "Aitken", "title": "2 · cos(x)−x", "page": 18,
     "target": "↗  Aitken Δ²", "values": {"ai_f": "cos(x)-x", "ai_g": "cos(x)", "ai_x0": 0.5}},
    {"section": "Aitken", "title": "4 · g(x)=e⁻ˣ", "page": 18,
     "target": "↗  Aitken Δ²", "values": {"ai_f": "exp(-x)-x", "ai_g": "exp(-x)", "ai_x0": 1.0}},
    {"section": "Aitken", "title": "6 · g(x)=ln(x+1)", "page": 18,
     "target": "↗  Aitken Δ²", "values": {"ai_f": "log(x+1)-x", "ai_g": "log(x+1)", "ai_x0": 0.5}},
    # Newton–Raphson, página 19.
    {"section": "Newton–Raphson", "title": "2 · x³−2x−5", "page": 19,
     "target": "╱╲  Newton–Raphson", "values": {"nw_f": "x^3-2*x-5", "nw_x0": 1.5}},
    {"section": "Newton–Raphson", "title": "4 · raíz sexta de 2", "page": 19,
     "target": "╱╲  Newton–Raphson", "values": {"nw_f": "x^6-2", "nw_x0": 1.0}},
    {"section": "Newton–Raphson", "title": "7 · ln(x)−1", "page": 19,
     "target": "╱╲  Newton–Raphson", "values": {"nw_f": "log(x)-1", "nw_x0": 2.0}},
    # Interpolación desde función, páginas 25–26.
    {"section": "Lagrange", "title": "9 · sin(x) en [0,π], grado 2", "page": 25,
     "target": "f→Σ  Interpolar desde función", "values": {"if_function": "sin(x)", "if_nodes": "0, pi/2, pi"}},
    {"section": "Lagrange", "title": "11 · 1/x con nodos 2, 2.5, 4.5", "page": 25,
     "target": "f→Σ  Interpolar desde función", "values": {"if_function": "1/x", "if_nodes": "2, 2.5, 4.5"}},
    {"section": "Lagrange", "title": "13.c · ln(x+1)", "page": 26,
     "target": "f→Σ  Interpolar desde función", "values": {"if_function": "log(x+1)", "if_nodes": "0, 0.6, 0.9"}},
    # Diferencias finitas, página 26.
    {"section": "Diferencias finitas", "title": "3 · x³−x en x=1, h=0.1", "page": 26,
     "target": "f′  Derivación numérica", "values": {"der_source": "Función", "der_function": "x^3-x", "der_point": "1", "der_h": "0.1", "der_second": True}},
    {"section": "Diferencias finitas", "title": "4 · eˣ sin(x) en x=1", "page": 26,
     "target": "f′  Derivación numérica", "values": {"der_source": "Función", "der_function": "exp(x)*sin(x)", "der_point": "1", "der_h": "0.01", "der_second": True}},
    {"section": "Diferencias finitas", "title": "6 · posición, velocidad y aceleración", "page": 26,
     "target": "f′  Derivación numérica", "values": {"der_source": "Tabla de nodos", "der_xs": "0,1,2,3,4,5,6,7,8", "der_ys": "0,1.9,4.2,7.8,12,17,25,32,42", "der_point": "4", "der_complete_table": True}},
    # Newton–Cotes, páginas 34–35.
    {"section": "Newton–Cotes", "title": "1.a · (6+3cos x), trapecio n=2", "page": 34,
     "target": "∫  Integración Newton–Cotes", "values": {"int_function": "6+3*cos(x)", "int_a": "0", "int_b": "π/2", "int_n": 2, "int_method": "Regla del trapecio"}},
    {"section": "Newton–Cotes", "title": "1.b · (6+3cos x), Simpson 1/3", "page": 34,
     "target": "∫  Integración Newton–Cotes", "values": {"int_function": "6+3*cos(x)", "int_a": "0", "int_b": "π/2", "int_n": 4, "int_method": "Regla de Simpson 1/3"}},
    {"section": "Newton–Cotes", "title": "4.a · x²eˣ, trapecio n=4", "page": 34,
     "target": "∫  Integración Newton–Cotes", "values": {"int_function": "x^2*exp(x)", "int_a": "0", "int_b": "3", "int_n": 4, "int_method": "Regla del trapecio"}},
    {"section": "Newton–Cotes", "title": "6.b · sin(x), Simpson 1/3", "page": 34,
     "target": "∫  Integración Newton–Cotes", "values": {"int_function": "sin(x)", "int_a": "0", "int_b": "π", "int_n": 4, "int_method": "Regla de Simpson 1/3"}},
    {"section": "Newton–Cotes", "title": "7.b · e^(x²), punto medio n=5", "page": 34,
     "target": "∫  Integración Newton–Cotes", "values": {"int_function": "exp(x^2)", "int_a": "0", "int_b": "1", "int_n": 5, "int_method": "Rectángulo por punto medio"}},
]
