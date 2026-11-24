"""
Módulo: metodos.py
Descripción: Implementación de métodos numéricos iterativos de un paso para
resolver sistemas de ecuaciones diferenciales ordinarias (EDOs) de primer orden
en forma de problema de Cauchy: y'(t) = f(t, y), y(a) = y0.

Dependencias:
    - Python 3.x
    - numpy (pip install numpy)

Ejecución:
    Se importa desde scripts de simulación (por ejemplo, malseirs.py).
"""

import time
import numpy as np


def euler_sistema(f, a: float, b: float, y0, n_pasos: int):
    """Resuelve y'(t) = f(t, y) en [a, b] con Euler explícito en n_pasos iteraciones."""
    if n_pasos < 1:
        raise ValueError("n_pasos debe ser >= 1.")

    inicio = time.perf_counter()

    y0_arr = np.asarray(y0, dtype=float)
    h = (b - a) / n_pasos  # Parada al completar n_pasos sobre la malla [a, b]
    tv = np.linspace(a, b, n_pasos + 1)
    yv = np.zeros((n_pasos + 1, y0_arr.size), dtype=float)
    yv[0] = y0_arr

    for k in range(n_pasos):
        yv[k + 1] = yv[k] + h * f(tv[k], yv[k])

    tiempo = time.perf_counter() - inicio
    return tv, yv, tiempo


def calcular_error_infinito(y_aprox: np.ndarray, y_ref: np.ndarray) -> float:
    """Error máximo en norma infinito entre dos trayectorias en los mismos instantes."""
    if y_aprox.shape != y_ref.shape:
        raise ValueError("y_aprox y y_ref deben tener la misma forma.")
    return float(np.max(np.abs(y_aprox - y_ref)))