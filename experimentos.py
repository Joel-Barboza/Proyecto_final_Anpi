"""
Módulo: experimentos.py

Descripción:
Experimentos numéricos para comparar Euler explícito, Heun y
Runge-Kutta de cuarto orden aplicados al modelo MalSEIRS.

Para cada método se estudian:
    - Tamaño de paso h
    - Número de pasos N
    - Aproximación numérica final
    - Error numérico
    - Orden observado de convergencia
    - Tiempo de ejecución
    - Máximo de infectados I_max
    - Instante t_max del máximo de infección

La solución de referencia se obtiene con RK4 usando una malla fina.
También se verifica la estabilidad de esa referencia al reducir
su tamaño de paso a la mitad.

Dependencias:
    - Python 3.x
    - numpy
    - matplotlib
    - malseirs.py
    - metodos.py

Ejecución:
    python experimentos.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from malseirs import (
    crear_sistema_malseirs,
    condiciones_iniciales_base
)

from metodos import (
    euler_sistema,
    heun_sistema,
    rk4_sistema,
    calcular_error_infinito
)


# Configuración de los experimentos

T0 = 0.0
TF = 30.0

H_VALORES = [
    0.1,
    0.05,
    0.025,
    0.0125
]

H_REF = 0.00125


# Funciones auxiliares

def numero_pasos(t0, tf, h):
    """Calcula el número de pasos para un tamaño de paso h."""

    n = int(
        round(
            (tf - t0) / h
        )
    )

    h_real = (
        (tf - t0) / n
    )

    if not np.isclose(
        h_real,
        h
    ):

        raise ValueError(
            f"El tamaño de paso h = {h} "
            f"no divide exactamente "
            f"el intervalo [{t0}, {tf}]."
        )

    return n


def referencia_en_malla(
    tv,
    tv_ref,
    yv_ref
):
    """
    Extrae de la referencia los valores correspondientes
    a los mismos tiempos de la solución aproximada.
    """

    h_ref = (
        tv_ref[1]
        - tv_ref[0]
    )

    indices = np.rint(
        (tv - tv_ref[0])
        / h_ref
    ).astype(int)

    if np.any(
        indices < 0
    ) or np.any(
        indices >= len(tv_ref)
    ):

        raise ValueError(
            "La malla solicitada está fuera "
            "de la solución de referencia."
        )

    if not np.allclose(
        tv_ref[indices],
        tv,
        atol=1e-10,
        rtol=0.0
    ):

        raise ValueError(
            "Los tiempos de la aproximación "
            "no coinciden con los tiempos "
            "de la solución de referencia."
        )

    return yv_ref[indices]


def calcular_metricas(
    tv,
    yv,
    tiempo,
    h,
    metodo,
    error=None
):
    """Calcula las métricas principales de una ejecución."""

    indice_max = int(
        np.argmax(
            yv[:, 2]
        )
    )

    return {
        "metodo": metodo,
        "h": h,
        "N": len(tv) - 1,
        "error": error,
        "orden_observado": None,
        "tiempo": tiempo,
        "I_max": float(
            yv[indice_max, 2]
        ),
        "t_max": float(
            tv[indice_max]
        ),
        "S_final": float(
            yv[-1, 0]
        ),
        "E_final": float(
            yv[-1, 1]
        ),
        "I_final": float(
            yv[-1, 2]
        ),
        "R_final": float(
            yv[-1, 3]
        ),
    }


def ejecutar_experimento(
    nombre,
    metodo,
    sistema,
    y0,
    h,
    tv_ref,
    yv_ref
):
    """Ejecuta un método para un tamaño de paso h."""

    n_pasos = numero_pasos(
        T0,
        TF,
        h
    )

    tv, yv, tiempo = metodo(
        sistema,
        T0,
        TF,
        y0,
        n_pasos
    )

    y_ref_malla = referencia_en_malla(
        tv,
        tv_ref,
        yv_ref
    )

    error = calcular_error_infinito(
        yv,
        y_ref_malla
    )

    metricas = calcular_metricas(
        tv,
        yv,
        tiempo,
        h,
        nombre,
        error
    )

    return (
        tv,
        yv,
        metricas
    )


def calcular_ordenes_observados(
    resultados
):
    """
    Calcula el orden observado entre dos refinamientos consecutivos.

    p = log(E1/E2) / log(h1/h2)
    """

    metodos = sorted(
        set(
            resultado["metodo"]
            for resultado in resultados
        )
    )

    for metodo in metodos:

        datos = [
            resultado
            for resultado in resultados
            if resultado["metodo"] == metodo
        ]

        datos = sorted(
            datos,
            key=lambda resultado:
            resultado["h"],
            reverse=True
        )

        datos[0]["orden_observado"] = None

        for i in range(
            1,
            len(datos)
        ):

            error_anterior = (
                datos[i - 1]["error"]
            )

            error_actual = (
                datos[i]["error"]
            )

            h_anterior = (
                datos[i - 1]["h"]
            )

            h_actual = (
                datos[i]["h"]
            )

            if (
                error_anterior is None
                or error_actual is None
                or error_anterior <= 0
                or error_actual <= 0
            ):

                datos[i][
                    "orden_observado"
                ] = None

            else:

                datos[i][
                    "orden_observado"
                ] = float(
                    np.log(
                        error_anterior
                        / error_actual
                    )
                    / np.log(
                        h_anterior
                        / h_actual
                    )
                )


# Solución numérica de referencia

def crear_solucion_referencia(
    sistema,
    y0
):
    """
    Genera la referencia con RK4 y verifica su estabilidad
    reduciendo el tamaño de paso a la mitad.
    """

    n_ref = numero_pasos(
        T0,
        TF,
        H_REF
    )

    print(
        "\nSolución numérica de referencia"
    )

    print(
        "Método: RK4"
    )

    print(
        f"h_ref = {H_REF}"
    )

    print(
        f"N_ref = {n_ref}"
    )

    tv_ref, yv_ref, _ = rk4_sistema(
        sistema,
        T0,
        TF,
        y0,
        n_ref
    )

    tv_fina, yv_fina, _ = rk4_sistema(
        sistema,
        T0,
        TF,
        y0,
        2 * n_ref
    )

    y_fina_en_ref = referencia_en_malla(
        tv_ref,
        tv_fina,
        yv_fina
    )

    diferencia = calcular_error_infinito(
        yv_ref,
        y_fina_en_ref
    )

    print(
        f"Diferencia máxima al reducir "
        f"el paso a {H_REF / 2:g}: "
        f"{diferencia:.6e}"
    )

    return (
        tv_ref,
        yv_ref
    )


# Resultados en consola

def imprimir_configuracion(
    y0
):
    """Muestra los datos comunes a todas las ejecuciones."""

    print(
        "\nConfiguración de los experimentos"
    )

    print(
        f"Valores iniciales "
        f"[S(0), E(0), I(0), R(0)]: "
        f"{y0.tolist()}"
    )

    print(
        f"Intervalo de simulación: "
        f"[{T0}, {TF}]"
    )

    print(
        f"Tamaños de paso utilizados: "
        f"{H_VALORES}"
    )


def imprimir_tabla(
    resultados
):
    """Imprime la tabla principal de resultados."""

    print(
        "\nRESULTADOS DE LOS EXPERIMENTOS\n"
    )

    encabezado = (
        f"{'Método':<18}"
        f"{'h':<11}"
        f"{'N':<9}"
        f"{'Error':<17}"
        f"{'p obs.':<11}"
        f"{'Tiempo (s)':<15}"
        f"{'I_max':<14}"
        f"{'t_max':<12}"
    )

    print(
        encabezado
    )

    print(
        "-" * 107
    )

    for resultado in resultados:

        error_texto = (
            f"{resultado['error']:.6e}"
        )

        if (
            resultado["orden_observado"]
            is None
        ):

            orden_texto = "-"

        else:

            orden_texto = (
                f"{resultado['orden_observado']:.4f}"
            )

        print(
            f"{resultado['metodo']:<18}"
            f"{resultado['h']:<11.5f}"
            f"{resultado['N']:<9d}"
            f"{error_texto:<17}"
            f"{orden_texto:<11}"
            f"{resultado['tiempo']:<15.6f}"
            f"{resultado['I_max']:<14.6f}"
            f"{resultado['t_max']:<12.6f}"
        )


def imprimir_soluciones_finales(
    resultados
):
    """Imprime las aproximaciones numéricas obtenidas en t = TF."""

    print(
        f"\nAPROXIMACIONES NUMÉRICAS "
        f"FINALES EN t = {TF}\n"
    )

    encabezado = (
        f"{'Método':<18}"
        f"{'h':<12}"
        f"{'S(tf)':<16}"
        f"{'E(tf)':<16}"
        f"{'I(tf)':<16}"
        f"{'R(tf)':<16}"
    )

    print(
        encabezado
    )

    print(
        "-" * 94
    )

    for resultado in resultados:

        print(
            f"{resultado['metodo']:<18}"
            f"{resultado['h']:<12.5f}"
            f"{resultado['S_final']:<16.6f}"
            f"{resultado['E_final']:<16.6f}"
            f"{resultado['I_final']:<16.6f}"
            f"{resultado['R_final']:<16.6f}"
        )


# Guardar resultados

def guardar_csv(
    resultados
):
    """Guarda los resultados numéricos en un archivo CSV."""

    nombre_archivo = (
        "resultados_experimentos.csv"
    )

    campos = [
        "metodo",
        "h",
        "N",
        "error",
        "orden_observado",
        "tiempo",
        "I_max",
        "t_max",
        "S_final",
        "E_final",
        "I_final",
        "R_final",
    ]

    with open(
        nombre_archivo,
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()

        escritor.writerows(
            resultados
        )

    print(
        f"\nResultados guardados en: "
        f"{nombre_archivo}"
    )


def nombre_seguro(
    nombre
):
    """Convierte el nombre de un método en un nombre de archivo simple."""

    nombre = nombre.lower()

    nombre = nombre.replace(
        " ",
        "_"
    )

    nombre = nombre.replace(
        "í",
        "i"
    )

    nombre = nombre.replace(
        "é",
        "e"
    )

    nombre = nombre.replace(
        "ó",
        "o"
    )

    return nombre


# Gráficas

def graficar_trayectorias(
    nombre_metodo,
    tv,
    yv,
    h
):
    """Grafica S(t), E(t), I(t) y R(t) para un método."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        tv,
        yv[:, 0],
        label="S(t)"
    )

    plt.plot(
        tv,
        yv[:, 1],
        label="E(t)"
    )

    plt.plot(
        tv,
        yv[:, 2],
        label="I(t)"
    )

    plt.plot(
        tv,
        yv[:, 3],
        label="R(t)"
    )

    plt.xlabel(
        "Tiempo"
    )

    plt.ylabel(
        "Número de nodos"
    )

    plt.title(
        f"Modelo MalSEIRS - "
        f"{nombre_metodo} - h = {h}"
    )

    plt.legend()

    plt.grid(
        True
    )

    archivo = (
        f"trayectorias_"
        f"{nombre_seguro(nombre_metodo)}.png"
    )

    plt.tight_layout()

    plt.savefig(
        archivo,
        dpi=300
    )

    plt.close()


def graficar_infectados(
    soluciones
):
    """Compara I(t) entre los tres métodos."""

    plt.figure(
        figsize=(10, 6)
    )

    estilos = (
        "-",
        "--",
        ":"
    )

    for (
        nombre,
        solucion
    ), estilo in zip(
        soluciones.items(),
        estilos
    ):

        tv, yv, h = solucion

        plt.plot(
            tv,
            yv[:, 2],
            label=f"{nombre}, h = {h}",
            linestyle=estilo
        )

    plt.xlabel(
        "Tiempo"
    )

    plt.ylabel(
        "Nodos infectados I(t)"
    )

    plt.title(
        "Comparación de la población "
        "infectada I(t)"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "comparacion_infectados.png",
        dpi=300
    )

    plt.close()


def graficar_tiempos(
    resultados
):
    """Compara el tiempo de ejecución según el tamaño de paso."""

    metodos = sorted(
        set(
            resultado["metodo"]
            for resultado in resultados
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    for metodo in metodos:

        datos = [
            resultado
            for resultado in resultados
            if resultado["metodo"] == metodo
        ]

        datos = sorted(
            datos,
            key=lambda resultado:
            resultado["h"]
        )

        h = [
            resultado["h"]
            for resultado in datos
        ]

        tiempos = [
            resultado["tiempo"]
            for resultado in datos
        ]

        plt.plot(
            h,
            tiempos,
            marker="o",
            label=metodo
        )

    plt.xscale(
        "log"
    )

    plt.xticks(
        H_VALORES,
        [
            f"{paso:g}"
            for paso in H_VALORES
        ]
    )

    plt.tick_params(
        axis="x",
        which="minor",
        labelbottom=False
    )

    plt.xlabel(
        "Tamaño de paso h"
    )

    plt.ylabel(
        "Tiempo de ejecución (s)"
    )

    plt.title(
        "Tiempo de ejecución de los métodos"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "comparacion_tiempos.png",
        dpi=300
    )

    plt.close()


def graficar_errores(
    resultados
):
    """Grafica el error respecto a la referencia para cada método."""

    metodos = sorted(
        set(
            resultado["metodo"]
            for resultado in resultados
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    for metodo in metodos:

        datos = [
            resultado
            for resultado in resultados
            if resultado["metodo"] == metodo
        ]

        datos = sorted(
            datos,
            key=lambda resultado:
            resultado["h"]
        )

        h = [
            resultado["h"]
            for resultado in datos
        ]

        errores = [
            resultado["error"]
            for resultado in datos
        ]

        plt.loglog(
            h,
            errores,
            marker="o",
            label=metodo
        )

    plt.xticks(
        H_VALORES,
        [
            f"{paso:g}"
            for paso in H_VALORES
        ]
    )

    plt.tick_params(
        axis="x",
        which="minor",
        labelbottom=False
    )

    plt.xlabel(
        "Tamaño de paso h"
    )

    plt.ylabel(
        "Error máximo en norma infinito"
    )

    plt.title(
        "Error numérico según "
        "el tamaño de paso"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        "comparacion_errores.png",
        dpi=300
    )

    plt.close()


# Programa principal

if __name__ == "__main__":

    sistema = crear_sistema_malseirs()

    y0 = condiciones_iniciales_base()

    resultados = []

    soluciones_grafica = {}

    tv_ref, yv_ref = crear_solucion_referencia(
        sistema,
        y0
    )

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

        for h in H_VALORES:

            tv, yv, metricas = ejecutar_experimento(
                nombre,
                metodo,
                sistema,
                y0,
                h,
                tv_ref,
                yv_ref
            )

            resultados.append(
                metricas
            )

            if h == H_VALORES[-1]:

                soluciones_grafica[
                    nombre
                ] = (
                    tv,
                    yv,
                    h
                )

    calcular_ordenes_observados(
        resultados
    )

    imprimir_configuracion(
        y0
    )

    imprimir_tabla(
        resultados
    )

    imprimir_soluciones_finales(
        resultados
    )

    guardar_csv(
        resultados
    )

    for nombre, solucion in soluciones_grafica.items():

        tv, yv, h = solucion

        graficar_trayectorias(
            nombre,
            tv,
            yv,
            h
        )

    graficar_infectados(
        soluciones_grafica
    )

    graficar_tiempos(
        resultados
    )

    graficar_errores(
        resultados
    )

    print(
        "\nExperimentos completados."
    )
