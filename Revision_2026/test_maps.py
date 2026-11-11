"""Tests de los mapas y de la funcion objetivo del ataque caotico.

Lo que hay que fijar antes de creerle cualquier numero al ataque:

  1. Que la version vectorizada que consume CBPSO da exactamente lo mismo que la
     transcripcion escalar de los notebooks.
  2. Que el error cuadratico medio se anula en los parametros verdaderos.
  3. Que la distancia media entre puntos esta saturada si se mide sobre 3000
     puntos y no lo esta sobre los N = 30 que define la funcion objetivo. Es el
     defecto que invalida los valores 0.360 y 0.434 del informe.

Ejecutar:  python -m pytest test_maps.py -q
"""

import numpy as np
import pytest

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


def test_lm_batch_matches_scalar():
    scalar = lm_orbit(**LM_TRUE, n=30)
    batch = lm_orbit_batch(
        np.array([LM_TRUE["mu"]]), np.array([LM_TRUE["mu1"]]),
        np.array([LM_TRUE["alpha"]]), np.array([LM_TRUE["x0"]]),
        np.array([LM_TRUE["y0"]]), n=30,
    )
    np.testing.assert_array_equal(scalar, batch[0])


def test_glm_batch_matches_scalar():
    scalar = glm_orbit(**GLM_TRUE, n=30)
    batch = glm_orbit_batch(
        *(np.array([GLM_TRUE[k]]) for k in ("mu", "mu1", "nu", "nu1", "alpha", "x0", "y0")),
        n=30,
    )
    np.testing.assert_array_equal(scalar, batch[0])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_objective_vanishes_at_the_truth(name):
    scenario = SCENARIOS[name]
    objective, _ = make_objective(scenario)
    truth = np.array([[scenario.truth[k] for k in scenario.free]])
    assert objective(truth)[0] == 0.0


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_objective_batch_matches_one_at_a_time(name):
    scenario = SCENARIOS[name]
    objective, _ = make_objective(scenario)
    rng = np.random.default_rng(0)
    lower, upper = scenario.bounds[:, 0], scenario.bounds[:, 1]
    sample = lower + (upper - lower) * rng.random((7, scenario.dim))
    together = objective(sample)
    apart = np.array([objective(row[None, :])[0] for row in sample])
    np.testing.assert_array_equal(together, apart)


def test_glm_threshold_convention_is_the_one_in_the_code():
    """Con la convencion del informe la orbita converge; con la del codigo es caotica."""
    code = glm_orbit(**GLM_TRUE, n=4000, skip_first=False)
    swapped = dict(GLM_TRUE)
    swapped["mu"], swapped["nu"] = GLM_TRUE["nu"], GLM_TRUE["mu"]
    swapped["mu1"], swapped["nu1"] = GLM_TRUE["nu1"], GLM_TRUE["mu1"]
    paper = glm_orbit(**swapped, n=4000, skip_first=False)
    assert len(np.unique(np.round(code[-2000:], 10), axis=0)) > 1000
    assert len(np.unique(np.round(paper[-2000:], 10), axis=0)) == 1


def test_distance_metric_saturates_over_long_trajectories():
    true_long = lm_orbit(**LM_TRUE, n=3001)
    nudged = dict(LM_TRUE)
    nudged["x0"] += 1e-12
    near_long = lm_orbit(**nudged, n=3001)

    # Sobre 3000 puntos, una orbita que difiere 1e-12 en x0 ya esta saturada.
    assert mean_point_distance(true_long, near_long) > 0.3
    # Sobre los N = 30 de la funcion objetivo, la metrica si resuelve.
    assert mean_point_distance(true_long[:29], near_long[:29]) < 1e-6


def test_scenarios_are_symmetric_between_maps():
    """Los dos mapas se atacan con los mismos escenarios, que es lo que faltaba."""
    assert SCENARIOS["LM-A"].known_initial and SCENARIOS["GLM-A"].known_initial
    assert not SCENARIOS["LM-B"].known_initial
    assert not SCENARIOS["GLM-B"].known_initial
    for name in ("LM-A", "LM-B", "GLM-A", "GLM-B"):
        scenario = SCENARIOS[name]
        assert scenario.bounds[scenario.free.index("alpha")].tolist() == [0.0, 1.0]
