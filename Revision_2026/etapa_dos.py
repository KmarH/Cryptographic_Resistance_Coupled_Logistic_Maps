"""Segunda etapa del ataque: recuperar las condiciones iniciales.

El metodo del articulo tiene dos etapas. La primera optimiza los parametros
del mapa con la funcion objetivo evaluada a partir del segundo punto, de forma
que las condiciones iniciales no entran en la busqueda. La segunda fija los
parametros en el valor estimado, libera \\(x_0\\) e \\(y_0\\), y compara
orbitas que si incluyen el punto inicial. Asi lo hace el cuaderno original:

    etapa 1   bounds = [(0,4), (0,4), (0.1,1), (x0,x0), (y0,y0)]
    etapa 2   bounds = [(mu,mu), (mu1,mu1), (alpha,alpha), (0.001,0.1), (0.001,0.1)]

Aqui se reproduce esa segunda etapa sobre los parametros que cada ataque
recupero, para los dos mapas. La pregunta que responde es si la llave queda
recuperada por completo o solo en sus parametros, que es lo que hasta ahora
estaba medido.

Las cotas \\([0.001, 0.1]\\) son las del cuaderno original. Se reporta tambien
el resultado con el dominio completo \\([0,1]\\), que es el que corresponde a
un atacante que no sabe de antemano donde cae la respuesta.

Uso:  python etapa_dos.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from cbpso import CBPSOSettings
from cbpso import minimize as cbpso_minimize
from maps import (
    GLM_TRUE,
    LM_TRUE,
    N_POINTS,
    glm_orbit,
    glm_orbit_batch,
    lm_orbit,
    lm_orbit_batch,
)

EXITO = 1e-6
SALIDA = Path("etapa_dos.json")
COTAS = {
    "del cuaderno original": np.array([(0.001, 0.1), (0.001, 0.1)]),
    "dominio completo": np.array([(0.0, 1.0), (0.0, 1.0)]),
}


def objetivo_iniciales(mapa, parametros, verdad):
    """ECM sobre orbitas que incluyen el punto inicial, con parametros fijos."""
    if mapa == "LM":
        real = lm_orbit(
            verdad["mu"], verdad["mu1"], verdad["alpha"],
            verdad["x0"], verdad["y0"], n=N_POINTS, skip_first=False,
        )
    else:
        real = glm_orbit(
            verdad["mu"], verdad["mu1"], verdad["nu"], verdad["nu1"],
            verdad["alpha"], verdad["x0"], verdad["y0"],
            n=N_POINTS, skip_first=False,
        )

    def evaluar(matriz):
        matriz = np.atleast_2d(np.asarray(matriz, dtype=np.float64))
        x0, y0 = matriz[:, 0], matriz[:, 1]
        unos = np.ones(len(matriz))
        with np.errstate(over="ignore", invalid="ignore"):
            if mapa == "LM":
                estimada = lm_orbit_batch(
                    parametros["mu"] * unos, parametros["mu1"] * unos,
                    parametros["alpha"] * unos, x0, y0,
                    n=N_POINTS, skip_first=False,
                )
            else:
                estimada = glm_orbit_batch(
                    parametros["mu"] * unos, parametros["mu1"] * unos,
                    parametros["nu"] * unos, parametros["nu1"] * unos,
                    parametros["alpha"] * unos, x0, y0,
                    n=N_POINTS, skip_first=False,
                )
            error = np.sum((real[None] - estimada) ** 2, axis=(1, 2)) / N_POINTS
        return np.where(np.isfinite(error), error, np.inf)

    return evaluar


def atacar(evaluar, cotas, semillas=10):
    """CBPSO con la configuracion del articulo, y evolucion diferencial igualada."""
    resultados = {}
    ajustes = CBPSOSettings(n_particles=1000, n_iterations=250)
    valores, puntos = [], []
    for semilla in range(semillas):
        salida = cbpso_minimize(evaluar, cotas, ajustes, seed=semilla)
        valores.append(float(salida.best_error))
        puntos.append(np.asarray(salida.best_position))
    mejor = int(np.argmin(valores))
    resultados["CBPSO"] = (float(valores[mejor]), puntos[mejor],
                           int(np.sum(np.array(valores) <= EXITO)), semillas)

    valores, puntos = [], []
    for semilla in range(semillas):
        salida = differential_evolution(
            lambda m: evaluar(np.asarray(m).T), cotas, seed=semilla,
            init="sobol", polish=True, vectorized=True,
            maxiter=1254, popsize=100, tol=1e-16,
        )
        valores.append(float(salida.fun))
        puntos.append(np.asarray(salida.x))
    mejor = int(np.argmin(valores))
    resultados["evolucion diferencial"] = (
        float(valores[mejor]), puntos[mejor],
        int(np.sum(np.array(valores) <= EXITO)), semillas,
    )
    return resultados


def main():
    # Parametros que cada ataque de la etapa 1 recupero, tomados de lo medido.
    casos = {
        "LM, parametros exactos": (
            "LM", {k: LM_TRUE[k] for k in ("mu", "mu1", "alpha")}, LM_TRUE,
        ),
        "GLM, parametros de la corrida que rompio": (
            "GLM",
            {k: GLM_TRUE[k] for k in ("mu", "mu1", "nu", "nu1", "alpha")},
            GLM_TRUE,
        ),
        "GLM, mejor ciclo de CBPSO": (
            "GLM",
            {"mu": 0.83697532, "mu1": 0.88769771, "nu": 0.76073693,
             "nu1": 0.29870959, "alpha": 0.21762990},
            GLM_TRUE,
        ),
    }

    estado = {}
    for nombre, (mapa, parametros, verdad) in casos.items():
        evaluar = objetivo_iniciales(mapa, parametros, verdad)
        print(f"\n=== {nombre}")
        estado[nombre] = {}
        for etiqueta, cotas in COTAS.items():
            for metodo, (ecm, punto, exitos, n) in atacar(evaluar, cotas).items():
                error = 100.0 * np.abs(
                    punto - np.array([verdad["x0"], verdad["y0"]])
                ) / np.array([verdad["x0"], verdad["y0"]])
                print(f"  [{etiqueta:<22} {metodo:<22}] ECM={ecm:.3e} "
                      f"x0={punto[0]:.8f} y0={punto[1]:.8f} "
                      f"err={error[0]:.3g}% / {error[1]:.3g}%  exitos={exitos}/{n}")
                estado[nombre][f"{etiqueta} | {metodo}"] = {
                    "mse": ecm, "x0": float(punto[0]), "y0": float(punto[1]),
                    "error_pct": error.tolist(), "exitos": exitos, "corridas": n,
                }
        SALIDA.write_text(json.dumps(estado, indent=1), encoding="utf-8")
    print(f"\n[escrito en {SALIDA}]")


if __name__ == "__main__":
    main()
