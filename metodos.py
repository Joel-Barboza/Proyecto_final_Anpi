"""
Módulo: metodos.py

Descripción:
Implementación de métodos numéricos de un paso para resolver sistemas
de ecuaciones diferenciales ordinarias de primer orden en forma de
problema de Cauchy:

    y'(t) = f(t, y),    y(a) = y0

Métodos incluidos:
    - Euler explícito
    - Heun
    - Runge-Kutta clásico de cuarto orden (RK4)
"""

import time
import numpy as np


def euler_sistema(f, a: float, b: float, y0, n_pasos: int):
    """Resuelve un sistema de EDO con Euler explícito."""

    if n_pasos < 1:
        raise ValueError("n_pasos debe ser >= 1.")

    inicio = time.perf_counter()

    y0_arr = np.asarray(y0, dtype=float)
    h = (b - a) / n_pasos
    tv = np.linspace(a, b, n_pasos + 1)
    yv = np.zeros((n_pasos + 1, y0_arr.size), dtype=float)
    yv[0] = y0_arr

    for k in range(n_pasos):
        yv[k + 1] = yv[k] + h * f(tv[k], yv[k])

    tiempo = time.perf_counter() - inicio

    return tv, yv, tiempo


def heun_sistema(f, a: float, b: float, y0, n_pasos: int):
    """Resuelve un sistema de EDO con el método de Heun."""

    if n_pasos < 1:
        raise ValueError("n_pasos debe ser >= 1.")

    inicio = time.perf_counter()

    y0_arr = np.asarray(y0, dtype=float)
    h = (b - a) / n_pasos
    tv = np.linspace(a, b, n_pasos + 1)
    yv = np.zeros((n_pasos + 1, y0_arr.size), dtype=float)
    yv[0] = y0_arr

    for k in range(n_pasos):
        k1 = f(tv[k], yv[k])

        k2 = f(
            tv[k] + h,
            yv[k] + h * k1
        )

        yv[k + 1] = yv[k] + (h / 2.0) * (k1 + k2)

    tiempo = time.perf_counter() - inicio

    return tv, yv, tiempo


def rk4_sistema(f, a: float, b: float, y0, n_pasos: int):
    """Resuelve un sistema de EDO con Runge-Kutta clásico de orden 4."""

    if n_pasos < 1:
        raise ValueError("n_pasos debe ser >= 1.")

    inicio = time.perf_counter()

    y0_arr = np.asarray(y0, dtype=float)
    h = (b - a) / n_pasos
    medio_paso = h / 2.0
    tv = np.linspace(a, b, n_pasos + 1)
    yv = np.zeros((n_pasos + 1, y0_arr.size), dtype=float)
    yv[0] = y0_arr

    for k in range(n_pasos):
        t = tv[k]
        y = yv[k]

        k1 = f(t, y)

        k2 = f(
            t + medio_paso,
            y + medio_paso * k1
        )

        k3 = f(
            t + medio_paso,
            y + medio_paso * k2
        )

        k4 = f(
            t + h,
            y + h * k3
        )

        yv[k + 1] = y + (h / 6.0) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )

    tiempo = time.perf_counter() - inicio

    return tv, yv, tiempo


def calcular_error_infinito(
    y_aprox: np.ndarray,
    y_ref: np.ndarray
) -> float:
    """Calcula el error máximo en norma infinito entre dos trayectorias."""

    if y_aprox.shape != y_ref.shape:
        raise ValueError(
            "y_aprox y y_ref deben tener la misma forma."
        )

    return float(
        np.max(
            np.abs(y_aprox - y_ref)
        )
    )