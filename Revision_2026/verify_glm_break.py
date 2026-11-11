"""Verificacion del hallazgo: un ataque de presupuesto alto si rompe el mapa generalizado.

Un barrido largo con evolucion diferencial seguida de reinicios de Nelder-Mead
recupero la llave exacta del mapa generalizado en 1 de 12 corridas. Ese barrido
usaba las cotas estrechas de la version original del codigo, que son mucho mas
favorables al atacante que las cotas simetricas del experimento principal. La
pregunta que decide que puede afirmar el articulo es si el exito depende de esas
cotas o no, de modo que aqui se repite la receta en los dos casos.

La receta se reproduce de manera fiel: `updating="immediate"`, que es el modo en
que corrio el barrido original. El modo vectorizado de scipy fuerza
`updating="deferred"`, que cambia la trayectoria estocastica del algoritmo, de
modo que las semillas no son comparables entre los dos modos y la version
diferida resulto mas debil. Se conserva `--vectorized` unicamente para
exploraciones rapidas.

El resultado de cada semilla se guarda apenas termina, asi que una interrupcion
no pierde el trabajo hecho y una segunda ejecucion retoma donde quedo.

El trabajo se puede repartir entre maquinas: `--solo` elige una de las dos
mitades y `--seeds` acepta una especificacion de semillas, de modo que cada
maquina corre un subconjunto disjunto y escribe su propio archivo. Las semillas
quedan anotadas en cada resultado, asi que los archivos se fusionan sin
ambiguedad; repetir una semilla en dos maquinas sirve de prueba de que ambas
corren el mismo experimento.

Uso:  python verify_glm_break.py [--seeds 0-11] [--solo completas]
                                 [--restarts 40] [--vectorized]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from maps import SCENARIOS, make_objective

SUCCESS = 1e-6
CHECKPOINT = Path("verificacion_glm.json")

# Cotas de la version original del codigo del ataque (notebook CBPSOResults_GLM).
NARROW = np.array([(0.5, 1.0), (0.5, 1.0), (0.5, 1.0), (0.01, 1.0), (0.01, 0.5)])
# Cotas declaradas por la formulacion del mapa, usadas en el experimento principal.
FULL = SCENARIOS["GLM-A"].bounds


def parse_seeds(spec):
    """Convierte "0,6-11" en [0, 6, 7, 8, 9, 10, 11], sin repetir y en orden."""
    seeds = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if "-" in piece.lstrip("-"):
            low, high = piece.split("-", 1)
            seeds.update(range(int(low), int(high) + 1))
        elif piece:
            seeds.add(int(piece))
    return sorted(seeds)


def attack(objective, bounds, seed, restarts, vectorized):
    """Evolucion diferencial seguida de reinicios locales de Nelder-Mead."""
    def scalar(vector):
        return float(objective(np.asarray(vector)[None, :])[0])

    if vectorized:
        def target(matrix):
            return objective(np.asarray(matrix).T)
    else:
        target = scalar

    outcome = differential_evolution(
        target, bounds, seed=seed, init="sobol", polish=True, vectorized=vectorized,
        maxiter=6000, popsize=60, mutation=(0.2, 1.5), recombination=0.9,
        strategy="best1bin", tol=1e-16,
    )
    best_value, best_vector = float(outcome.fun), np.asarray(outcome.x)

    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        start = np.clip(
            best_vector + rng.normal(0, 1e-3, len(bounds)), bounds[:, 0], bounds[:, 1],
        )
        local = minimize(
            scalar, start, method="Nelder-Mead", bounds=bounds,
            options={"maxiter": 20000, "xatol": 1e-14, "fatol": 1e-18},
        )
        if local.fun < best_value:
            best_value, best_vector = float(local.fun), np.asarray(local.x)
    return best_value, best_vector


def summarize(runs, truth):
    values = np.array([r["mse"] for r in runs])
    best = int(np.argmin(values))
    vector = np.array(runs[best]["position"])
    return {
        "seeds": len(runs),
        "mse_min": float(values.min()),
        "mse_median": float(np.median(values)),
        "successes": int(np.sum(values <= SUCCESS)),
        "best_position": vector.tolist(),
        "percent_error_best": (100.0 * np.abs(vector - truth) / np.abs(truth)).tolist(),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-11")
    parser.add_argument("--solo", choices=("estrechas", "completas"))
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--vectorized", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    halves = [("estrechas", NARROW), ("completas", FULL)]
    if args.solo:
        halves = [h for h in halves if h[0] == args.solo]
    checkpoint = (
        CHECKPOINT if not args.solo else CHECKPOINT.with_name(
            f"{CHECKPOINT.stem}_{args.solo}{CHECKPOINT.suffix}",
        )
    )

    scenario = SCENARIOS["GLM-A"]
    objective, _ = make_objective(scenario)
    truth = np.array([scenario.truth[k] for k in scenario.free])

    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    state.setdefault("meta", {})
    state["meta"].update({
        "restarts": args.restarts,
        "updating": "deferred" if args.vectorized else "immediate",
        "seeds_asignadas": seeds,
        "maquina": platform.node(),
    })

    for label, bounds in halves:
        runs = state.get(label, {}).get("runs", [])
        done = {r["seed"] for r in runs}
        started = time.perf_counter()
        for seed in seeds:
            if seed in done:
                continue
            value, vector = attack(objective, bounds, seed, args.restarts, args.vectorized)
            runs.append({"seed": seed, "mse": value, "position": vector.tolist()})
            flag = "  <-- CRUZA EL UMBRAL" if value <= SUCCESS else ""
            print(f"[{label} seed {seed}] ECM={value:.6e}{flag}", flush=True)
            state[label] = summarize(sorted(runs, key=lambda r: r["seed"]), truth)
            checkpoint.write_text(json.dumps(state, indent=1), encoding="utf-8")
        row = state[label]
        print(
            f"[{label}] exitos={row['successes']}/{row['seeds']} "
            f"min={row['mse_min']:.6e} mediana={row['mse_median']:.6e} "
            f"({time.perf_counter() - started:.0f} s)\n",
            flush=True,
        )

    print(f"[escrito en {checkpoint}]")


if __name__ == "__main__":
    main()
