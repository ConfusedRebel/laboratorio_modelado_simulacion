"""Regresión: la navegación de Streamlit debe conservar sintaxis válida."""
import ast
from pathlib import Path


def test_app_has_valid_python_syntax():
    source = Path("app.py").read_text(encoding="utf-8")
    ast.parse(source, filename="app.py")


def test_integration_contains_function_evaluation_table():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'with st.expander("Abrir tabla para evaluar f(x)")' in source
    assert "function_evaluation_rows(parsed, values, decimals)" in source


def test_integration_defaults_to_six_decimals_and_shows_complete_formula():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'st.number_input("Decimales mostrados", 6, 15, 6, key="int_decimals"' in source
    assert 'st.expander("Ver fórmula completa aplicada", expanded=True)' in source
