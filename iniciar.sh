#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

cd "${PROJECT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
    printf 'ERROR: no se encontró python3. Instalalo con el gestor de paquetes de tu distribución.\n' >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    printf 'Creando el entorno virtual...\n'
    if ! python3 -m venv "${VENV_DIR}"; then
        printf '\nNo se pudo crear el entorno virtual.\n' >&2
        printf 'En Ubuntu/Debian: sudo apt install python3-venv\n' >&2
        printf 'En Arch Linux:    sudo pacman -S python\n' >&2
        exit 1
    fi
fi

PYTHON="${VENV_DIR}/bin/python"

if ! "${PYTHON}" -c 'import streamlit, numpy, sympy, pandas, plotly, mpmath' >/dev/null 2>&1; then
    printf 'Instalando dependencias (solo se hace la primera vez)...\n'
    "${PYTHON}" -m pip install -r requirements.txt
fi

printf '\nIniciando la aplicación en http://localhost:8501\n'
printf 'Para cerrarla, presioná Ctrl+C.\n\n'
exec "${PYTHON}" -m streamlit run app.py
