"""
Módulo: malseirs.py

Descripción:
Definición del modelo MalSEIRS aplicado a la propagación de malware
en redes de computadoras y ejecución de prueba para Euler explícito,
Heun y Runge-Kutta de cuarto orden.

Dependencias:
    - Python 3.x
    - numpy
    - metodos.py

Ejecución:
    python malseirs.py
"""

import numpy as np

from metodos import (
    euler_sistema,
    heun_sistema,
    rk4_sistema
)


def parametros_malseirs_base() -> dict:
    """Retorna los parámetros base del modelo MalSEIRS."""

    return {
        "beta_0": 0.8,
        "xi": 1.0,
        "mu": 0.1,
        "alpha": 0.2,
        "c": 5.0,
        "sigma": 0.9,
        "phi": 0.46,
        "p": 0.1,
        "Lambda": 1.0,
        "a": 0.1,
        "m": 1.0,
    }


def condiciones_iniciales_base() -> np.ndarray:
    """Retorna [S(0), E(0), I(0), R(0)]."""

    return np.array(
        [30.0, 2.0, 1.0, 0.0],
        dtype=float
    )


def crear_sistema_malseirs(params: dict = None):
    """
    Retorna la función f(t, y) del sistema MalSEIRS.

    y = [S, E, I, R]
    """

    if params is None:
        params = parametros_malseirs_base()

    beta_0 = params["beta_0"]
    xi = params["xi"]
    mu = params["mu"]
    alpha = params["alpha"]
    c = params["c"]
    sigma = params["sigma"]
    phi = params["phi"]
    p = params["p"]
    lam = params["Lambda"]
    a = params["a"]
    m = params["m"]

    def f(t: float, y: np.ndarray) -> np.ndarray:

        s_val, e_val, i_val, r_val = y

        beta_t = beta_0 / (
            1.0 + xi * i_val
        )

        gamma_t = np.tanh(
            i_val / c
        )

        omega_t = abs(
            np.exp(-a * t)
            * np.cos(
                (2.0 * np.pi * t) / m
            )
        )

        ds_dt = (
            p * lam
            - beta_t * i_val * s_val
            + omega_t * r_val
            - (mu + phi) * s_val
        )

        de_dt = (
            beta_t * i_val * s_val
            - sigma * e_val
            - mu * e_val
        )

        di_dt = (
            sigma * e_val
            - gamma_t * i_val
            - (mu + alpha) * i_val
        )

        dr_dt = (
            (1.0 - p) * lam
            + gamma_t * i_val
            - omega_t * r_val
            - mu * r_val
            + phi * s_val
        )

        return np.array(
            [
                ds_dt,
                de_dt,
                di_dt,
                dr_dt
            ],
            dtype=float
        )

    return f


def mostrar_prueba_metodo(
    nombre,
    metodo,
    sistema,
    t0,
    tf,
    y0,
    n_pasos
):
    """Ejecuta una prueba básica de un método."""

    tv, yv, tiempo = metodo(
        sistema,
        t0,
        tf,
        y0,
        n_pasos
    )

    indice_max = int(
        np.argmax(
            yv[:, 2]
        )
    )

    i_max = float(
        yv[indice_max, 2]
    )

    t_max = float(
        tv[indice_max]
    )

    print(
        f"\nMétodo: {nombre}"
    )

    print(
        f"Valores iniciales "
        f"[S(0), E(0), I(0), R(0)]: "
        f"{y0.tolist()}"
    )

    print(
        f"Número de iteraciones (pasos): "
        f"{n_pasos} "
        f"(h = {(tf - t0) / n_pasos:.6f})"
    )

    print(
        f"Aproximación numérica final "
        f"en t = {tv[-1]:.2f} "
        f"[S, E, I, R]: "
        f"{np.round(yv[-1], 6).tolist()}"
    )

    print(
        f"Tiempo de ejecución: "
        f"{tiempo:.6f} s"
    )

    print(
        f"Máximo de infectados: "
        f"I_max = {i_max:.6f} "
        f"en t_max = {t_max:.4f}"
    )


if __name__ == "__main__":

    sistema = crear_sistema_malseirs()

    y0 = condiciones_iniciales_base()

    t0 = 0.0
    tf = 30.0
    n_pasos = 300

    metodos = [
        (
            "Euler explícito",
            euler_sistema
        ),
        (
            "Heun",
            heun_sistema
        ),
        (
            "RK4",
            rk4_sistema
        ),
    ]

    for nombre, metodo in metodos:

        mostrar_prueba_metodo(
            nombre,
            metodo,
            sistema,
            t0,
            tf,
            y0,
            n_pasos
        )