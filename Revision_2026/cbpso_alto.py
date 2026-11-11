"""Intenta romper el mapa generalizado usando unicamente CBPSO.

La llave del mapa generalizado ya se recupero con evolucion diferencial de
presupuesto alto seguida de reinicios locales. Queda por saber si el mismo
resultado se alcanza sin salir del algoritmo que el trabajo ya emplea, porque
de eso depende que el articulo tenga que introducir un segundo algoritmo
completo o le baste con escalar el suyo.

Se mantiene CBPSO tal cual: peso de inercia exponencial, inicializacion
caotica, busqueda caotica, c1 = c2 = 2. Lo unico que cambia es el tamano del
enjambre y el numero de iteraciones, y opcionalmente un refinamiento local de
Nelder-Mead sobre el mejor resultado, que no es un algoritmo de busqueda rival
sino el pulido final del mismo ataque.

Cada configuracion se corre con las mismas 12 semillas y sobre el espacio de
llaves declarado, para que los numeros sean comparables con los del barrido de
evolucion diferencial. El resultado se guarda despues de cada corrida.

Uso:  python cbpso_alto.py [--seeds 12] [--restarts 40] [--config nombre]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize as minimizar_local

from cbpso import CBPSOSettings
from cbpso import minimize as cbpso_minimize
from maps import SCENARIOS, make_objective

EXITO = 1e-6
SALIDA = Path("cbpso_alto.json")

# Presupuesto del ataque con evolucion diferencial que si recupero la llave:
# 300 individuos por 6001 generaciones.
PRESUPUESTO_DE = 300 * 6001

# El enjambre crece antes que el numero de iteraciones, porque la figura del
# progreso ya mostro que la mediana del mapa generalizado se estanca pasadas
# unas 50 iteraciones: alargar la corrida no es lo que falta.
CONFIGURACIONES = {
    "base": CBPSOSettings(n_particles=1000, n_iterations=250),
    "igualada": CBPSOSettings(n_particles=3000, n_iterations=600),
    "diez": CBPSOSettings(n_particles=10000, n_iterations=1800),
    "cien": CBPSOSettings(n_particles=20000, n_iterations=9000),
}


def parse_seeds(spec):
    """Convierte "0,6-11" en [0, 6, 7, 8, 9, 10, 11], sin repetir y en orden."""
    semillas = set()
    for trozo in spec.split(","):
        trozo = trozo.strip()
        if "-" in trozo.lstrip("-"):
            bajo, alto = trozo.split("-", 1)
            semillas.update(range(int(bajo), int(alto) + 1))
        elif trozo:
            semillas.add(int(trozo))
    return sorted(semillas)


def presupuesto(ajustes):
    return ajustes.n_particles * (ajustes.n_iterations + 1)


def refinar(objetivo, punto, cotas, semilla, reinicios):
    """Pulido local de Nelder-Mead alrededor del mejor punto del enjambre."""
    def escalar(vector):
        return float(objetivo(np.asarray(vector)[None, :])[0])

    mejor_valor = escalar(punto)
    mejor_punto = np.asarray(punto, dtype=float)
    generador = np.random.default_rng(semilla)
    for _ in range(reinicios):
        inicio = np.clip(
            mejor_punto + generador.normal(0, 1e-3, len(cotas)),
            cotas[:, 0], cotas[:, 1],
        )
        local = minimizar_local(
            escalar, inicio, method="Nelder-Mead", bounds=cotas,
            options={"maxiter": 20000, "xatol": 1e-14, "fatol": 1e-18},
        )
        if local.fun < mejor_valor:
            mejor_valor, mejor_punto = float(local.fun), np.asarray(local.x)
    return mejor_valor, mejor_punto


def main():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--seeds", default="0-11")
    analizador.add_argument("--restarts", type=int, default=40)
    analizador.add_argument("--config", choices=tuple(CONFIGURACIONES), action="append")
    analizador.add_argument("--out", default=str(SALIDA))
    argumentos = analizador.parse_args()
    nombres = argumentos.config or ["igualada"]
    semillas = parse_seeds(argumentos.seeds)
    salida = Path(argumentos.out)

    escenario = SCENARIOS["GLM-A"]
    objetivo, _ = make_objective(escenario)
    cotas = escenario.bounds
    verdad = np.array([escenario.truth[k] for k in escenario.free])

    estado = json.loads(salida.read_text(encoding="utf-8")) if salida.exists() else {}

    for nombre in nombres:
        ajustes = CONFIGURACIONES[nombre]
        print(f"\n=== {nombre}: {ajustes.n_particles} particulas x "
              f"{ajustes.n_iterations} iteraciones = {presupuesto(ajustes):,} "
              f"evaluaciones ({presupuesto(ajustes) / PRESUPUESTO_DE:.2f} veces "
              f"el ataque con evolucion diferencial)", flush=True)
        corridas = estado.get(nombre, {}).get("runs", [])
        hechas = {c["seed"] for c in corridas}
        for semilla in semillas:
            if semilla in hechas:
                continue
            arranque = time.perf_counter()
            crudo = cbpso_minimize(objetivo, cotas, ajustes, seed=semilla)
            solo = float(crudo.best_error)
            pulido, punto = refinar(
                objetivo, crudo.best_position, cotas, semilla, argumentos.restarts,
            )
            segundos = time.perf_counter() - arranque
            corridas.append({
                "seed": semilla,
                "mse_cbpso": solo,
                "mse_refinado": pulido,
                "position": punto.tolist(),
                "seconds": segundos,
            })
            marca = "  <-- CRUZA EL UMBRAL" if pulido <= EXITO else ""
            print(f"[{nombre} semilla {semilla}] CBPSO={solo:.6e} "
                  f"refinado={pulido:.6e} ({segundos:.0f} s){marca}", flush=True)

            valores_solo = np.array([c["mse_cbpso"] for c in corridas])
            valores_ref = np.array([c["mse_refinado"] for c in corridas])
            mejor = min(corridas, key=lambda c: c["mse_refinado"])
            vector = np.array(mejor["position"])
            estado[nombre] = {
                "particulas": ajustes.n_particles,
                "iteraciones": ajustes.n_iterations,
                "evaluaciones": presupuesto(ajustes),
                "semillas": len(corridas),
                "exitos_cbpso_solo": int(np.sum(valores_solo <= EXITO)),
                "exitos_con_refinamiento": int(np.sum(valores_ref <= EXITO)),
                "min_cbpso": float(valores_solo.min()),
                "min_refinado": float(valores_ref.min()),
                "mediana_refinado": float(np.median(valores_ref)),
                "error_pct_mejor": (
                    100.0 * np.abs(vector - verdad) / np.abs(verdad)
                ).tolist(),
                "runs": sorted(corridas, key=lambda c: c["seed"]),
            }
            salida.write_text(json.dumps(estado, indent=1), encoding="utf-8")

        fila = estado[nombre]
        print(f"[{nombre}] CBPSO solo {fila['exitos_cbpso_solo']}/{fila['semillas']}, "
              f"con refinamiento {fila['exitos_con_refinamiento']}/{fila['semillas']}, "
              f"minimo {fila['min_refinado']:.6e}", flush=True)

    print(f"\n[escrito en {SALIDA}]")


if __name__ == "__main__":
    main()
