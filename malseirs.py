"""
Módulo: malseirs.py

Descripción:
Definición del modelo MalSEIRS aplicado a la propagación de malware
en redes de computadoras y ejecución de prueba para los métodos
de Euler explícito y Heun.

Dependencias:
    - Python 3.x
    - numpy
    - metodos.py en el mismo directorio

Ejecución:
    python malseirs.py
"""

import numpy as np

from metodos import euler_sistema, heun_sistema


def parametros_malseirs_base() -> dict:
    """
    Parámetros base del modelo MalSEIRS.
    """

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
    """
    Condiciones iniciales:
    [S(0), E(0), I(0), R(0)].
    """

    return np.array(
        [30.0, 2.0, 1.0, 0.0],
        dtype=float
    )


def crear_sistema_malseirs(params: dict = None):
    """
    Retorna la función f(t, y) correspondiente al sistema
    de cuatro EDO del modelo MalSEIRS.

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

        beta_t = beta_0 / (1.0 + xi * i_val)

        gamma_t = np.tanh(i_val / c)

        omega_t = abs(
            np.exp(-a * t)
            * np.cos((2.0 * np.pi * t) / m)
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
            [ds_dt, de_dt, di_dt, dr_dt],
            dtype=float
        )

    return f


if __name__ == "__main__":

    sistema = crear_sistema_malseirs()

    y0 = condiciones_iniciales_base()

    t0 = 0.0
    tf = 30.0
    n_pasos = 300


    # ========================================================
    # EULER EXPLICITO
    # ========================================================

    tv, yv, tiempo = euler_sistema(
        sistema,
        t0,
        tf,
        y0,
        n_pasos
    )

    i_max = float(
        np.max(yv[:, 2])
    )

    t_max = float(
        tv[np.argmax(yv[:, 2])]
    )

    print("Método: Euler explícito")

    print(
        f"Valores iniciales [S(0), E(0), I(0), R(0)]: "
        f"{y0.tolist()}"
    )

    print(
        f"Número de iteraciones (pasos): {n_pasos} "
        f"(h = {(tf - t0) / n_pasos:.6f})"
    )

    print(
        f"Aproximación numérica final en t = {tv[-1]:.2f} "
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


    # ========================================================
    # HEUN
    # ========================================================

    print("\n" + "=" * 60)

    tv_heun, yv_heun, tiempo_heun = heun_sistema(
        sistema,
        t0,
        tf,
        y0,
        n_pasos
    )

    i_max_heun = float(
        np.max(yv_heun[:, 2])
    )

    t_max_heun = float(
        tv_heun[np.argmax(yv_heun[:, 2])]
    )

    print("Método: Heun")

    print(
        f"Valores iniciales [S(0), E(0), I(0), R(0)]: "
        f"{y0.tolist()}"
    )

    print(
        f"Número de iteraciones (pasos): {n_pasos} "
        f"(h = {(tf - t0) / n_pasos:.6f})"
    )

    print(
        f"Aproximación numérica final en t = "
        f"{tv_heun[-1]:.2f} [S, E, I, R]: "
        f"{np.round(yv_heun[-1], 6).tolist()}"
    )

    print(
        f"Tiempo de ejecución: "
        f"{tiempo_heun:.6f} s"
    )

    print(
        f"Máximo de infectados: "
        f"I_max = {i_max_heun:.6f} "
        f"en t_max = {t_max_heun:.4f}"
    )