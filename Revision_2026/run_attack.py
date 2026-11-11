"""Ataque caotico corregido sobre los dos mapas, y las figuras del informe.

Corre los cuatro escenarios (dos mapas x dos conjuntos de incognitas), guarda
las estadisticas en JSON y regenera las mismas figuras que el informe, con las
correcciones siguientes:

  - La distancia media entre puntos se mide sobre los N = 30 puntos que define
    la funcion objetivo, no sobre trayectorias de 3000 puntos donde la metrica
    esta saturada.
  - El conteo de exitos es por ciclo y contra el umbral 10^-6 que el informe
    declara, no contra el mejor acumulado ni contra 9x10^-3.
  - Las graficas de la funcion objetivo usan N = 30, el mismo N del ataque, y no
    N = 1000 como los notebooks `GraphsMSE.ipynb` y `Graphs2025.ipynb`.
  - Se guarda el vector de parametros de cada ciclo, de modo que los cuadros del
    informe se pueden reconstruir desde los datos y no desde la salida impresa.
  - Las cotas son las mismas para los dos mapas y nadie recibe un prior estrecho
    alrededor de la respuesta.

Uso:  python run_attack.py [--cycles 100] [--outdir .]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cbpso import CBPSOSettings, minimize
from maps import (
    GLM_TRUE,
    LM_TRUE,
    SCENARIOS,
    glm_orbit,
    glm_orbit_batch,
    lm_orbit,
    lm_orbit_batch,
    make_objective,
    mean_point_distance,
)

SUCCESS = 1e-6          # umbral de indistinguibilidad que declara el informe
N_POINTS = 30
PRETTY = {
    "mu": r"$\mu$", "mu1": r"$\mu_1$", "nu": r"$\nu$", "nu1": r"$\nu_1$",
    "alpha": r"$\alpha$", "x0": r"$x_0$", "y0": r"$y_0$",
}


def run_scenario(name, cycles, settings):
    scenario = SCENARIOS[name]
    objective, true_orbit = make_objective(scenario)
    errors, distances, seconds, positions, histories = [], [], [], [], []

    for cycle in range(cycles):
        result = minimize(objective, scenario.bounds, settings, seed=cycle)
        estimated = scenario.orbit(result.best_position)
        errors.append(result.best_error)
        distances.append(mean_point_distance(true_orbit, estimated))
        seconds.append(result.seconds)
        positions.append(result.best_position.tolist())
        histories.append(result.history.tolist())

    errors = np.array(errors)
    distances = np.array(distances)
    seconds = np.array(seconds)
    matrix = np.array(positions)
    truth = np.array([scenario.truth[k] for k in scenario.free])
    percent = 100.0 * np.abs(matrix - truth) / np.abs(truth)
    best = int(np.argmin(errors))

    return {
        "name": name,
        "label": scenario.label,
        "map": scenario.map_name,
        "free": scenario.free,
        "dim": scenario.dim,
        "bounds": scenario.bounds.tolist(),
        "truth": truth.tolist(),
        "cycles": cycles,
        "mse_mean": float(errors.mean()),
        "mse_std": float(errors.std(ddof=1)),
        "mse_min": float(errors.min()),
        "mse_max": float(errors.max()),
        "dist_mean": float(distances.mean()),
        "dist_std": float(distances.std(ddof=1)),
        "dist_min": float(distances.min()),
        "dist_max": float(distances.max()),
        "seconds_mean": float(seconds.mean()),
        "seconds_std": float(seconds.std(ddof=1)),
        "successes": int(np.sum(errors <= SUCCESS)),
        "percent_error_mean": percent.mean(axis=0).tolist(),
        "percent_error_best": percent[best].tolist(),
        "best_cycle": best + 1,
        "best_mse": float(errors[best]),
        "best_position": matrix[best].tolist(),
        "best_distance": float(distances[best]),
        "errors": errors.tolist(),
        "positions": positions,
        "history_median": np.median(np.array(histories), axis=0).tolist(),
        "history_p10": np.percentile(np.array(histories), 10, axis=0).tolist(),
        "history_p90": np.percentile(np.array(histories), 90, axis=0).tolist(),
    }


def differential_evolution_control(name, runs=10):
    """Control metodologico: evolucion diferencial con el mismo presupuesto.

    Sirve para separar "el mapa resiste" de "este optimizador no lo logro".
    """
    from scipy.optimize import differential_evolution

    scenario = SCENARIOS[name]
    objective, _ = make_objective(scenario)
    budget = CBPSOSettings().n_particles * (CBPSOSettings().n_iterations + 1)

    truth = np.array([scenario.truth[k] for k in scenario.free])
    # La funcion objetivo ya evalua poblaciones completas, de modo que se pasa a
    # scipy en modo vectorizado; scipy entrega (dim, S) y espera (S,).
    population = 40 * scenario.dim
    max_generations = max(1, budget // population - 1)
    values, vectors = [], []
    for seed in range(runs):
        def batch(matrix):
            return objective(np.asarray(matrix).T)

        outcome = differential_evolution(
            batch, scenario.bounds, seed=seed, init="sobol", polish=True,
            maxiter=max_generations, popsize=40, tol=1e-16, vectorized=True,
        )
        values.append(float(outcome.fun))
        vectors.append(np.asarray(outcome.x))
    values = np.array(values)
    best = int(np.argmin(values))
    percent = 100.0 * np.abs(vectors[best] - truth) / np.abs(truth)
    return {
        "runs": runs,
        "budget_per_run": int(budget),
        "mse_min": float(values.min()),
        "mse_median": float(np.median(values)),
        "successes": int(np.sum(values <= SUCCESS)),
        "best_position": vectors[best].tolist(),
        "percent_error_best": percent.tolist(),
    }


def figure_landscape_2d(path, map_name, parameter):
    """ECM en funcion de un parametro, con las condiciones iniciales conocidas."""
    step = 0.0001
    if map_name == "LM":
        truth, lo, hi = LM_TRUE, 0.0, 4.0
        if parameter == "alpha":
            lo, hi = 0.0, 1.0
        reference = lm_orbit(**truth, n=N_POINTS)
        sweep = np.arange(lo, hi, step)
        args = {k: truth[k] * np.ones_like(sweep) for k in
                ("mu", "mu1", "alpha", "x0", "y0")}
        args[parameter] = sweep
        orbits = lm_orbit_batch(**args, n=N_POINTS)
    else:
        truth, lo, hi = GLM_TRUE, 0.01, 0.99
        reference = glm_orbit(**truth, n=N_POINTS)
        sweep = np.arange(lo, hi, step)
        args = {k: truth[k] * np.ones_like(sweep) for k in
                ("mu", "mu1", "nu", "nu1", "alpha", "x0", "y0")}
        args[parameter] = sweep
        orbits = glm_orbit_batch(**args, n=N_POINTS)

    error = np.sum((reference[None] - orbits) ** 2, axis=(1, 2)) / N_POINTS
    plt.figure(figsize=(7, 4.2))
    plt.plot(sweep, error, linewidth=0.7, color="tab:blue")
    plt.xlabel(f"Valor de {PRETTY[parameter]}")
    plt.ylabel("Error cuadrático medio")
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _landscape_grid(map_name, parameter, size=140):
    if map_name == "LM":
        truth = LM_TRUE
        lo, hi = (0.0, 4.0) if parameter in ("mu", "mu1") else (0.0, 1.0)
        reference = lm_orbit(**truth, n=N_POINTS, skip_first=False)
        runner, keys = lm_orbit_batch, ("mu", "mu1", "alpha", "x0", "y0")
    else:
        truth, lo, hi = GLM_TRUE, 0.01, 0.99
        reference = glm_orbit(**truth, n=N_POINTS, skip_first=False)
        runner = glm_orbit_batch
        keys = ("mu", "mu1", "nu", "nu1", "alpha", "x0", "y0")

    axis_param = np.linspace(lo, hi, size)
    axis_x0 = np.linspace(0.0, 1.0, size)
    grid_param, grid_x0 = np.meshgrid(axis_param, axis_x0)
    flat_param, flat_x0 = grid_param.ravel(), grid_x0.ravel()
    args = {k: truth[k] * np.ones_like(flat_param) for k in keys}
    args[parameter] = flat_param
    args["x0"] = flat_x0
    orbits = runner(**args, n=N_POINTS, skip_first=False)
    error = np.sum((reference[None] - orbits) ** 2, axis=(1, 2)) / N_POINTS
    return grid_param, grid_x0, error.reshape(grid_param.shape)


def figure_landscape_contour(path, map_name, parameter):
    grid_param, grid_x0, error = _landscape_grid(map_name, parameter)
    plt.figure(figsize=(6.5, 5))
    contour = plt.contourf(grid_param, grid_x0, error, 20, cmap="viridis")
    plt.colorbar(contour, label="Error cuadrático medio")
    plt.xlabel(PRETTY[parameter])
    plt.ylabel(r"$x_0$")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def figure_landscape_surface(path, map_name, parameter):
    grid_param, grid_x0, error = _landscape_grid(map_name, parameter)
    figure = plt.figure(figsize=(7, 5.4))
    axes = figure.add_subplot(111, projection="3d")
    axes.plot_surface(
        grid_param, grid_x0, error, cmap="viridis", linewidth=0, antialiased=False,
    )
    axes.set_xlabel(PRETTY[parameter], labelpad=10)
    axes.set_ylabel(r"$x_0$", labelpad=10)
    axes.set_zlabel("Error", labelpad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def figure_map_comparison(path, result):
    """Mapa original contra el aproximado con el mejor resultado del ataque."""
    scenario = SCENARIOS[result["name"]]
    truth = scenario.orbit(n=3000)
    estimated = scenario.orbit(np.array(result["best_position"]), n=3000)
    plt.figure(figsize=(6.4, 5.6))
    plt.scatter(truth[:, 0], truth[:, 1], s=1, color="tab:blue", label="Mapa original")
    plt.scatter(
        estimated[:, 0], estimated[:, 1], s=1, color="tab:red", label="Mapa aproximado",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(markerscale=5)
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def figure_iterations(path, result):
    iterations = np.arange(1, len(result["history_median"]) + 1)
    plt.figure(figsize=(7, 4.2))
    plt.fill_between(
        iterations, result["history_p10"], result["history_p90"],
        color="tab:blue", alpha=0.25, label="Percentiles 10-90",
    )
    plt.plot(
        iterations, result["history_median"], color="tab:blue",
        linewidth=1.2, label="Mediana de los ciclos",
    )
    plt.yscale("log")
    plt.xlabel("Iteración")
    plt.ylabel("Error cuadrático medio")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--skip-control", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    images = outdir / "Imagenes"
    images.mkdir(parents=True, exist_ok=True)
    settings = CBPSOSettings()

    results = {}
    for name in ("LM-A", "LM-B", "GLM-A", "GLM-B"):
        started = time.perf_counter()
        results[name] = run_scenario(name, args.cycles, settings)
        elapsed = time.perf_counter() - started
        row = results[name]
        print(
            f"[{name}] {row['label']}: ECM min={row['mse_min']:.3e} "
            f"mediana={np.median(row['errors']):.3e} exitos={row['successes']}"
            f"/{args.cycles}  ({elapsed:.0f} s)",
            flush=True,
        )

    if not args.skip_control:
        for name in results:
            results[name]["de_control"] = differential_evolution_control(name)
            print(f"[{name}] control DE: {results[name]['de_control']}", flush=True)

    figure_landscape_2d(images / "mu.png", "LM", "mu")
    figure_landscape_contour(images / "muContorno.png", "LM", "mu")
    figure_landscape_surface(images / "mu3D.png", "LM", "mu")
    figure_landscape_2d(images / "2d.png", "GLM", "nu")
    figure_landscape_contour(images / "contorno.png", "GLM", "nu")
    figure_landscape_surface(images / "3d.png", "GLM", "nu")
    # Las figuras comparativas corresponden al escenario con condiciones
    # iniciales conocidas, que es el que contrasta los dos mapas en igualdad de
    # condiciones y el que discute el texto.
    figure_map_comparison(images / "mapaAproximadovsOriginal.png", results["LM-A"])
    figure_map_comparison(images / "glmComparativo.png", results["GLM-A"])
    figure_iterations(images / "iters.png", results["LM-A"])
    figure_iterations(images / "itersGLM.png", results["GLM-A"])
    print("[figuras escritas]", flush=True)

    with open(outdir / "resultados_ataque.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    print(f"[escrito en {outdir / 'resultados_ataque.json'}]", flush=True)


if __name__ == "__main__":
    main()
