"""Das Kostenmodell LERNEN: die fünf Gewichte des parametrisierten DCOP (`ladcop_model`) werden per Cross-Entropy-
Methode (eine Evolutionsstrategie) aus dem Feedback des ECHTEN Makespans gelernt - Fitness = mittlerer Makespan der
exakten DPOP-Lösung relativ zu Contract Net auf Trainingsinstanzen. Der Löser bleibt der unveränderte exakte DPOP.

Trainingsinstanzen sind klein (n=6, dort ist der exakte DPOP billig) und stammen aus Seeds weit außerhalb von Demo-
und Held-out-Seeds (entkoppelt); das gelernte Modell wird auf größere Instanzen übertragen."""

import time
from dataclasses import dataclass

import numpy as np

import cn_constants as C
from cn_scenario import generate_instance
from ladcop_dpop_np import solve_dpop_np
from ladcop_model import (
    EFFECTIVE_MASK, VEHICLE_WEIGHTS, ModelWeights, cnp_solution, cost_arrays, instance_arrays, makespan_np,
)


@dataclass(frozen=True)
class TrainResult:
    weights: ModelWeights
    history: tuple            # je Iteration: bester Fitness-Wert (Makespan / CNP, kleiner ist besser)
    start_fitness: float      # Fitness der Fahrzeug-Gewichte auf den Trainingsinstanzen
    seconds: float


def train_set(n_agents, duration_variability, travel, n_instances, learn_seed, n_jobs=C.TRAIN_JOBS):
    out = []
    for i in range(n_instances):
        instance = generate_instance(
            n_jobs, n_agents, duration_variability, travel, C.TRAIN_SEED_BASE + learn_seed * 1000 + i,
        )
        arrays = instance_arrays(instance)
        out.append((arrays, cnp_solution(arrays)[0]))
    return out


def fitness(vector, tset):
    """Mittlerer Makespan der exakten DPOP-Lösung relativ zu Contract Net (kleiner ist besser)."""
    ratios = []
    for arrays, cnp_makespan in tset:
        unary, pair = cost_arrays(arrays, vector)
        assignment, _, _ = solve_dpop_np(unary, pair)
        ratios.append(makespan_np(arrays, assignment) / cnp_makespan)
    return float(np.mean(ratios))


def train_weights(
    n_agents, duration_variability, travel, n_instances=C.DEFAULT_TRAIN_INSTANCES, iterations=C.DEFAULT_CEM_ITERATIONS,
    learn_seed=C.DEFAULT_LEARN_SEED,
):
    started = time.perf_counter()
    tset = train_set(n_agents, duration_variability, travel, n_instances, learn_seed)
    mask = np.array(EFFECTIVE_MASK, dtype=float)
    rng = np.random.default_rng([learn_seed, 29])
    mean = VEHICLE_WEIGHTS.as_vector()
    sd = np.full(len(mean), C.CEM_SIGMA0) * mask
    start = fitness(mean, tset)
    best_value, best_vector = start, mean.copy()
    history = []
    for _ in range(iterations):
        candidates = mean + sd * rng.standard_normal((C.CEM_POPULATION, len(mean)))
        candidates[0] = mean
        values = np.array([fitness(c, tset) for c in candidates])
        elite = np.argsort(values)[:C.CEM_ELITE]
        mean = candidates[elite].mean(0)
        sd = candidates[elite].std(0) + 0.02 * mask
        if values[elite[0]] < best_value:
            best_value, best_vector = float(values[elite[0]]), candidates[elite[0]].copy()
        history.append(best_value)
    return TrainResult(ModelWeights.from_vector(best_vector), tuple(history), start, time.perf_counter() - started)
