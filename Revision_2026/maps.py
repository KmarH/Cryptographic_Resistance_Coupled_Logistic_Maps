"""Los dos mapas acoplados y la funcion objetivo del ataque, en version escalar y vectorizada.

La version escalar es transcripcion literal de `sim_mapa` de los notebooks
`CBPSOResults_LM.ipynb` y `CBPSOResults_GLM.ipynb`. La vectorizada evalua N_p
juegos de parametros a la vez y es la que consume CBPSO; `test_maps.py`
comprueba que las dos coinciden bit a bit.

Dos cosas que se conservan tal cual estaban:

  - El `% 1` en las dos coordenadas, como en los notebooks del ataque.
  - La normalizacion 1/N del error con N = 30, aunque la suma del escenario con
    condiciones iniciales conocidas tenga 29 terminos. Es la definicion del
    informe y cambiarla solo reescalaria todos los valores por 30/29.

Y una que se corrige: en el mapa generalizado, `log_map(v, r, q)` recibe el
UMBRAL p como segundo argumento y la AMPLITUD q como tercero, y la llamada de
los notebooks es `log_map(x, nu, mu)`. Es decir p -> nu y q -> mu, al contrario
de lo que dice el informe. Aqui los campos se llaman `threshold` y `amplitude`
para que no haya ambiguedad.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GLM_TRUE",
    "LM_TRUE",
    "SCENARIOS",
    "Scenario",
    "glm_orbit",
    "glm_orbit_batch",
    "lm_orbit",
    "lm_orbit_batch",
    "make_objective",
    "mean_point_distance",
]

N_POINTS = 30

# Llaves publicadas en el informe.
LM_TRUE = {"mu": 3.1, "mu1": 2.9, "alpha": 0.3314, "x0": 0.03, "y0": 0.01}
GLM_TRUE = {
    "mu": 0.91, "mu1": 0.85, "nu": 0.9, "nu1": 0.333,
    "alpha": 0.3314, "x0": 0.03, "y0": 0.01,
}


def lm_orbit(mu, mu1, alpha, x0, y0, n=N_POINTS, skip_first=True):
    """Orbita del mapa logistico simple acoplado, version escalar."""
    xs, ys = [x0], [y0]
    for i in range(1, n):
        xn = (mu * xs[i - 1] * (1 - xs[i - 1]) + alpha * (ys[i - 1] - xs[i - 1])) % 1
        yn = (mu1 * ys[i - 1] * (1 - ys[i - 1]) + alpha * (xs[i - 1] - ys[i - 1])) % 1
        xs.append(xn)
        ys.append(yn)
    start = 1 if skip_first else 0
    return np.column_stack((xs[start:], ys[start:]))


def _glm_scalar(v, threshold, amplitude):
    if v <= threshold:
        return -amplitude / (threshold * threshold) * (threshold - v) ** 2 + amplitude
    return -amplitude / ((1 - threshold) ** 2) * (threshold - v) ** 2 + amplitude


def glm_orbit(mu, mu1, nu, nu1, alpha, x0, y0, n=N_POINTS, skip_first=True):
    """Orbita del mapa logistico generalizado acoplado, version escalar.

    `nu` y `nu1` son los umbrales p; `mu` y `mu1` son las amplitudes q.
    """
    xs, ys = [x0], [y0]
    for i in range(1, n):
        xn = (_glm_scalar(xs[i - 1], nu, mu) + alpha * (ys[i - 1] - xs[i - 1])) % 1
        yn = (_glm_scalar(ys[i - 1], nu1, mu1) + alpha * (xs[i - 1] - ys[i - 1])) % 1
        xs.append(xn)
        ys.append(yn)
    start = 1 if skip_first else 0
    return np.column_stack((xs[start:], ys[start:]))


def lm_orbit_batch(mu, mu1, alpha, x0, y0, n=N_POINTS, skip_first=True):
    """Igual que `lm_orbit` pero con arreglos de forma (P,); devuelve (P, puntos, 2)."""
    x = np.array(x0, dtype=np.float64, copy=True)
    y = np.array(y0, dtype=np.float64, copy=True)
    out = np.empty((x.size, n, 2), dtype=np.float64)
    out[:, 0, 0], out[:, 0, 1] = x, y
    for i in range(1, n):
        xn = (mu * x * (1 - x) + alpha * (y - x)) % 1.0
        yn = (mu1 * y * (1 - y) + alpha * (x - y)) % 1.0
        x, y = xn, yn
        out[:, i, 0], out[:, i, 1] = x, y
    return out[:, 1:, :] if skip_first else out


def _glm_batch(v, threshold, amplitude):
    denom = np.where(v <= threshold, threshold * threshold, (1.0 - threshold) ** 2)
    return -amplitude / denom * (threshold - v) ** 2 + amplitude


def glm_orbit_batch(mu, mu1, nu, nu1, alpha, x0, y0, n=N_POINTS, skip_first=True):
    """Igual que `glm_orbit` pero con arreglos de forma (P,); devuelve (P, puntos, 2)."""
    x = np.array(x0, dtype=np.float64, copy=True)
    y = np.array(y0, dtype=np.float64, copy=True)
    out = np.empty((x.size, n, 2), dtype=np.float64)
    out[:, 0, 0], out[:, 0, 1] = x, y
    for i in range(1, n):
        xn = (_glm_batch(x, nu, mu) + alpha * (y - x)) % 1.0
        yn = (_glm_batch(y, nu1, mu1) + alpha * (x - y)) % 1.0
        x, y = xn, yn
        out[:, i, 0], out[:, i, 1] = x, y
    return out[:, 1:, :] if skip_first else out


class Scenario:
    """Un escenario de ataque: que mapa, que incognitas y con que cotas.

    Los dos escenarios son los mismos para ambos mapas, que es lo que hace la
    comparacion simetrica. En el informe el mapa simple se atacaba con 3
    parametros libres y las condiciones iniciales clavadas en su valor exacto,
    mientras el generalizado se atacaba con 5 parametros libres, de modo que la
    diferencia de resultados mezclaba la estructura del mapa con el numero de
    incognitas.
    """

    def __init__(self, map_name, free, bounds, truth):
        self.map_name = map_name
        self.free = list(free)
        self.bounds = np.array([bounds[k] for k in self.free], dtype=np.float64)
        self.truth = dict(truth)
        self.known_initial = "x0" not in self.free
        self.dim = len(self.free)

    @property
    def label(self):
        kind = "cond. iniciales conocidas" if self.known_initial else "todo libre"
        return f"{self.map_name} / {kind} ({self.dim}D)"

    def orbit_batch(self, matrix, n=N_POINTS):
        """`matrix` tiene forma (P, dim); devuelve (P, puntos, 2)."""
        values = {k: self.truth[k] * np.ones(len(matrix)) for k in self.truth}
        values.update({name: matrix[:, j] for j, name in enumerate(self.free)})
        skip = self.known_initial
        if self.map_name == "LM":
            return lm_orbit_batch(
                values["mu"], values["mu1"], values["alpha"],
                values["x0"], values["y0"], n=n, skip_first=skip,
            )
        return glm_orbit_batch(
            values["mu"], values["mu1"], values["nu"], values["nu1"],
            values["alpha"], values["x0"], values["y0"], n=n, skip_first=skip,
        )

    def orbit(self, vector=None, n=N_POINTS):
        """Version escalar, con los valores verdaderos si `vector` es None."""
        values = dict(self.truth)
        if vector is not None:
            values.update(dict(zip(self.free, vector, strict=True)))
        skip = self.known_initial
        if self.map_name == "LM":
            return lm_orbit(
                values["mu"], values["mu1"], values["alpha"],
                values["x0"], values["y0"], n=n, skip_first=skip,
            )
        return glm_orbit(
            values["mu"], values["mu1"], values["nu"], values["nu1"],
            values["alpha"], values["x0"], values["y0"], n=n, skip_first=skip,
        )


# Umbrales p = 0 y p = 1 son singulares en la ecuacion (7), de ahi el recorte.
_EPS = 0.01
LM_BOUNDS = {
    "mu": (0.0, 4.0), "mu1": (0.0, 4.0), "alpha": (0.0, 1.0),
    "x0": (0.0, 1.0), "y0": (0.0, 1.0),
}
GLM_BOUNDS = {
    "mu": (0.0, 1.0), "mu1": (0.0, 1.0),
    "nu": (_EPS, 1.0 - _EPS), "nu1": (_EPS, 1.0 - _EPS),
    "alpha": (0.0, 1.0), "x0": (0.0, 1.0), "y0": (0.0, 1.0),
}

SCENARIOS = {
    "LM-A": Scenario("LM", ["mu", "mu1", "alpha"], LM_BOUNDS, LM_TRUE),
    "LM-B": Scenario("LM", ["mu", "mu1", "alpha", "x0", "y0"], LM_BOUNDS, LM_TRUE),
    "GLM-A": Scenario("GLM", ["mu", "mu1", "nu", "nu1", "alpha"], GLM_BOUNDS, GLM_TRUE),
    "GLM-B": Scenario(
        "GLM", ["mu", "mu1", "nu", "nu1", "alpha", "x0", "y0"], GLM_BOUNDS, GLM_TRUE,
    ),
}


def make_objective(scenario, n=N_POINTS):
    """Error cuadratico medio entre la orbita verdadera y la estimada.

    Devuelve (objetivo_vectorizado, orbita_verdadera). El objetivo acepta (P, dim)
    y devuelve (P,).
    """
    true_orbit = scenario.orbit(n=n)

    def objective(matrix):
        matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
        with np.errstate(over="ignore", invalid="ignore"):
            estimated = scenario.orbit_batch(matrix, n=n)
            error = np.sum((true_orbit[None, :, :] - estimated) ** 2, axis=(1, 2)) / n
        return np.where(np.isfinite(error), error, np.inf)

    return objective, true_orbit


def mean_point_distance(true_orbit, estimated_orbit):
    """Distancia media entre puntos correspondientes, ecuaciones (10) y (11).

    Sobre los MISMOS N puntos que define la funcion objetivo. Los notebooks la
    calculaban sobre trayectorias de 3000 puntos, donde la metrica esta saturada:
    dos orbitas del mismo mapa que difieren 10^-12 en x0 dan 0.390, dos orbitas
    independientes dan 0.394 y pares de puntos al azar del atractor dan 0.377,
    de modo que el 0.360 del mapa simple y el 0.434 del generalizado caen los dos
    dentro de la banda de saturacion y no distinguen un ataque exitoso de uno
    fallido.
    """
    return float(np.mean(np.linalg.norm(true_orbit - estimated_orbit, axis=1)))
