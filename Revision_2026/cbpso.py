"""CBPSO vectorizado, con el mismo algoritmo de los notebooks y los bugs corregidos.

Se conserva sin cambios todo lo que es decision de diseno del metodo:

  - Peso de inercia w(t) = (w_max - w_min) exp(-p (t/T)^2) + w_min, con
    w_max = 0.9, w_min = 0.2, p = 100.
  - c_1 = c_2 = 2, T = 250, N_p = 1000.
  - Inicializacion caotica con el mapa logistico de mu = 4.
  - Busqueda caotica con factor de rango a = 0.1, disparada por la misma
    condicion que el codigo original.
  - Recorte de posiciones a las cotas.

Se corrige:

  1. Inicializacion degenerada. El original hace
     `positions[i] = bounds[i][0] + (bounds[i][1]-bounds[i][0]) * logistic[particle_id]`,
     con el MISMO escalar `logistic[particle_id]` para todas las dimensiones, de
     modo que el enjambre arranca sobre una recta del espacio de busqueda en
     lugar de cubrirlo. Es justo lo contrario del argumento que justifica
     N_p = 1000. Aqui se consume un iterado distinto del mapa logistico por cada
     par (particula, dimension).
  2. El mismo defecto en `chaotic_search`, que con un unico escalar por
     particula coloca la nueva posicion sobre una recta que pasa por la mejor
     posicion global.
  3. Rango de busqueda caotica proporcional al ancho de la cota y no al valor de
     la mejor posicion global. Con `search_range = a * global_best[i]` el rango
     se anula cuando la coordenada tiende a cero, asi que parametros con valor
     pequeno quedaban practicamente sin busqueda local.
  4. Semilla explicita, para que los ciclos sean reproducibles. El original no
     fija ninguna, y ademas la inicializacion caotica arranca siempre en
     x_0 = 0.6, de modo que las posiciones iniciales eran identicas en los 100
     ciclos mientras el resto era irreproducible.
  5. Conteo de exitos por ciclo. El original evalua
     `if bestmse <= cond_exito: exitos += 1` sobre el MEJOR ACUMULADO, no sobre
     el ciclo en curso, asi que incrementa en todos los ciclos posteriores al
     primer exito; en el notebook del mapa generalizado imprime 95 de 100.

Queda documentado, sin cambiarlo, que la condicion de disparo de la busqueda
caotica tal como esta escrita
(`err > 1`, o bien `10^-(i+2) < err < 10^-i`) solo puede cumplirse en las
primeras iteraciones: para i mayor que unos 20 la ventana es inalcanzable y la
busqueda caotica queda apagada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CBPSOResult", "CBPSOSettings", "minimize"]


@dataclass
class CBPSOSettings:
    n_particles: int = 1000
    n_iterations: int = 250
    w_max: float = 0.9
    w_min: float = 0.2
    p: float = 100.0
    c1: float = 2.0
    c2: float = 2.0
    a: float = 0.1
    chaos_mu: float = 4.0
    chaos_x0: float = 0.6


@dataclass
class CBPSOResult:
    best_position: np.ndarray
    best_error: float
    history: np.ndarray = field(repr=False)
    seconds: float = 0.0


def _logistic_stream(count, mu, x0):
    """`count` iterados del mapa logistico, para la inicializacion caotica."""
    out = np.empty(count, dtype=np.float64)
    x = x0
    for i in range(count):
        x = mu * x * (1.0 - x)
        out[i] = x
    return out


def minimize(objective, bounds, settings=None, seed=0):
    """Minimiza `objective` sobre la caja `bounds` de forma (dim, 2).

    `objective` recibe (P, dim) y devuelve (P,).
    """
    settings = settings or CBPSOSettings()
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=np.float64)
    dim = len(bounds)
    lower, upper = bounds[:, 0], bounds[:, 1]
    width = upper - lower
    n_p = settings.n_particles

    # Inicializacion caotica: un iterado por cada par (particula, dimension).
    # El arranque del mapa logistico se desplaza con la semilla para que los
    # ciclos no compartan las posiciones iniciales.
    x0 = settings.chaos_x0
    while True:
        x0 = (x0 + 0.000137 * seed) % 1.0
        if x0 not in (0.0, 0.25, 0.5, 0.75) and 0.0 < x0 < 1.0:
            break
    stream = _logistic_stream(n_p * dim + 100, settings.chaos_mu, x0)[100:]
    position = lower + width * stream.reshape(n_p, dim)
    velocity = rng.uniform(-1.0, 1.0, size=(n_p, dim))

    error = objective(position)
    best_local_position = position.copy()
    best_local_error = error.copy()
    index = int(np.argmin(error))
    best_global_position = position[index].copy()
    best_global_error = float(error[index])

    history = np.empty(settings.n_iterations, dtype=np.float64)
    started = time.perf_counter()

    for iteration in range(settings.n_iterations):
        weight = (settings.w_max - settings.w_min) * np.exp(
            -settings.p * (iteration / settings.n_iterations) ** 2
        ) + settings.w_min

        r1 = rng.random((n_p, dim))
        r2 = rng.random((n_p, dim))
        velocity = (
            weight * velocity
            + settings.c1 * r1 * (best_local_position - position)
            + settings.c2 * r2 * (best_global_position - position)
        )
        position = np.clip(position + velocity, lower, upper)

        # Busqueda caotica, con la condicion de disparo del codigo original.
        factor = None
        if best_global_error > 1.0:
            factor = settings.a
        elif 10.0 ** -(iteration + 2) < best_global_error < 10.0**-iteration:
            factor = settings.a / 2.0
        if factor is not None:
            chaos = rng.uniform(0.1, 0.9, size=(n_p, dim))
            for _ in range(100):
                chaos = settings.chaos_mu * chaos * (1.0 - chaos)
            span = factor * width
            position = np.clip(
                best_global_position - span + 2.0 * span * chaos, lower, upper,
            )

        error = objective(position)
        improved = error < best_local_error
        best_local_position[improved] = position[improved]
        best_local_error[improved] = error[improved]
        index = int(np.argmin(error))
        if error[index] < best_global_error:
            best_global_error = float(error[index])
            best_global_position = position[index].copy()
        history[iteration] = best_global_error

    return CBPSOResult(
        best_position=best_global_position,
        best_error=best_global_error,
        history=history,
        seconds=time.perf_counter() - started,
    )
