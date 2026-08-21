# Laboratorio de Modelado y Simulación

Aplicación educativa local para estudiar métodos de búsqueda de raíces mediante iteraciones, visualizaciones geométricas, errores y análisis de convergencia. Implementa Bisección, Punto Fijo, Newton–Raphson y aceleración Δ² de Aitken.

## Instalación

Requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate       # Windows PowerShell
pip install -r requirements.txt
```

## Ejecución

En Linux, la forma automática es:

```bash
chmod +x iniciar.sh
./iniciar.sh
```

El script crea `.venv`, instala las dependencias si todavía no están disponibles
y ejecuta la aplicación. La instalación solo necesita conexión la primera vez.

La forma manual es:

```bash
streamlit run app.py
```

La aplicación no usa APIs ni necesita Internet una vez instaladas las dependencias.

## Pruebas

```bash
pytest -q
```

## Funcionalidad

- Entrada segura de expresiones con SymPy, sin `eval()` directo.
- Bisección con verificación de Bolzano, intervalo animado y estimación teórica.
- Punto Fijo con cobweb, análisis aproximado de contracción, compacidad y Banach.
- Newton con derivada simbólica y tangente seleccionable por iteración.
- Aitken aplicado a una sucesión de Punto Fijo, con Δx y Δ²x.
- Errores absoluto/relativo, residuo, orden experimental, comparador y CSV.
- Teclado matemático para insertar raíces, potencias, funciones y constantes.
- Raíces de cualquier índice mediante `root(valor, índice)`, por ejemplo `root(x,3)`.
- Fórmulas interpretadas y derivadas simbólicas mostradas con notación matemática.
- Tabla de Newton: `n | Xn | f(Xn) | f'(Xn) | Xn+1 | Error Absoluto | Error Relativo`.
- Vista gráfica de cada tangente de Newton o de todas las tangentes simultáneas.
- Tolerancia ingresada como cantidad de decimales (hasta 100) y criterios mediante casillas, incluida la cantidad de iteraciones.
- Cálculo asistido de `g(x)` mediante una formulación de relajación `g(x)=x−λf(x)`.
- Aitken permite activar o desactivar la aceleración Δ² para comparar con la sucesión original.
- Diseño unificado en seis bloques: datos, resumen/ejecución, gráfico, resultado, tabla y convergencia.
- Identidad visual por método: ½ Bisección, ● Punto Fijo, ╱╲ Newton, ↗ Aitken, VS Comparación y 📖 Teoría.
- Explicaciones breves centradas en qué busca cada método, cuándo conviene y qué hace en cada paso.
- Constructor de funciones por interpolación de Lagrange con hasta 20 nodos.
- Visualización de cada base L_i(x), factores para los índices i y j, términos y_iL_i(x), polinomio final, gráfico y tabla de valores evaluados.
- Precisión de presentación configurable hasta 15 decimales, sin redondeo interno.
- Mensajes controlados para dominios inválidos, derivada o denominador casi nulos, ciclos y divergencia.

## Estructura

`core/` contiene parsing, métricas, modelos y convergencia; `methods/` no depende de Streamlit; `visualizations/` transforma historiales en gráficos; `examples/` reúne ejercicios; `tests/` valida el motor. `exports/` se crea automáticamente con rutas relativas.

## Agregar un método

1. Crear un módulo en `methods/` que devuelva `MethodResult` e `IterationResult`.
2. Crear su gráfico en `visualizations/` consumiendo solo el historial.
3. Añadir una pantalla delgada en `app.py`.
4. Incorporar ejemplos y tests sin acoplar el cálculo a la interfaz.

La separación permite incorporar más adelante interpolación, integración, EDO y sistemas dinámicos sin reescribir el núcleo.
