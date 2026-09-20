"""Das parametrisierte DCOP-Modell: dieselbe Struktur wie `dcop_graph` (eine Variable je Auftrag, Domäne = Agent,
Unärkosten + Paarkosten über einen VOLLSTÄNDIGEN Graphen), aber mit einer kleinen Familie einstellbarer Gewichte.

    unär(j, a)     = w_travel * Anfahrt(Start_a -> Auftrag j) + w_duration * Dauer_j
    paar(i, j, a=b) = w_distance * Anfahrt(Auftrag i -> Auftrag j) + w_load * Dauer_i * Dauer_j / 10 + w_count
    paar(i, j, a!=b) = 0

Mit den Fahrzeug-Gewichten (1, 1, 1, 0, 0) ist das exakt das Modell aus dcop-demo. `w_duration` ist wirkungslos: die
Dauer eines Auftrags ist für jeden Agenten gleich, sie verschiebt alle Zuteilungen um dieselbe Konstante.
`w_load` erzeugt (summiert über alle Paare eines Agenten) die quadrierte Last, `w_count` die Anzahl der Paare
(= n_a * (n_a - 1) / 2 je Agent) - beides sind Lastausgleichs-Terme, die dem Fahrzeug-Modell fehlen."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class ModelWeights:
    w_travel: float = 1.0
    w_duration: float = 1.0
    w_distance: float = 1.0
    w_load: float = 0.0
    w_count: float = 0.0

    def as_vector(self):
        return np.array([self.w_travel, self.w_duration, self.w_distance, self.w_load, self.w_count], dtype=float)

    @staticmethod
    def from_vector(vector):
        return ModelWeights(*[float(v) for v in vector])


VEHICLE_WEIGHTS = ModelWeights(1.0, 1.0, 1.0, 0.0, 0.0)          # das Modell aus dcop-demo
HAND_BALANCE_WEIGHTS = ModelWeights(1.0, 1.0, 1.0, 2.0, 0.0)     # von Hand: + Summe der quadrierten Lasten / 10
WEIGHT_NAMES = ("Anfahrt (unär)", "Dauer (unär, wirkungslos)", "Distanz (Paar)", "Last-Produkt (Paar)", "Anzahl (Paar)")
EFFECTIVE_MASK = (1, 0, 1, 1, 1)  # die vier wirksamen Gewichte; w_duration wird nicht gelernt


class Arrays(NamedTuple):
    pos: np.ndarray
    dur: np.ndarray
    start: np.ndarray
    tr: float


def instance_arrays(instance):
    return Arrays(
        np.array([j.position for j in instance.jobs]), np.array([j.duration for j in instance.jobs]),
        np.array(instance.agent_start_positions, dtype=float), instance.travel_time_per_unit,
    )


def cost_arrays(arrays, weights):
    """(U[n, k], S[n, n]): unäre Kosten je (Auftrag, Agent) und Paarkosten S[i, j] für "beide beim selben Agenten"."""
    pos, dur, start, tr = arrays
    k = len(start)
    w = weights.as_vector() if isinstance(weights, ModelWeights) else np.asarray(weights, dtype=float)
    unary = w[0] * np.abs(start[None, :] - pos[:, None]) * tr + w[1] * dur[:, None] * np.ones((1, k))
    distance = np.abs(pos[:, None] - pos[None, :]) * tr
    pair = w[2] * distance + w[3] * np.outer(dur, dur) / 10.0 + w[4]
    return unary, pair


def objective_np(unary, pair, assignment):
    """DCOP-Ziel einer Zuteilung: Σ unär + Σ_{i<j, gleicher Agent} paar."""
    a = np.asarray(assignment)
    n = len(a)
    same = a[:, None] == a[None, :]
    upper = np.triu_indices(n, 1)
    return float(unary[np.arange(n), a].sum() + (pair * same)[upper].sum())


def makespan_np(arrays, assignment):
    """Echter Makespan: Aufträge je Agent in aufsteigender Index-Reihenfolge (wie `cn_schedule`)."""
    pos, dur, start, tr = arrays
    position = start.copy()
    free = np.zeros(len(start))
    for j, a in enumerate(assignment):
        free[a] += abs(position[a] - pos[j]) * tr + dur[j]
        position[a] = pos[j]
    return float(free.max())


def cnp_solution(arrays):
    """Rohes Contract Net über numpy: (Makespan, Zuteilung)."""
    pos, dur, start, tr = arrays
    position = start.copy()
    free = np.zeros(len(start))
    assignment = []
    for j in range(len(pos)):
        finish = free + np.abs(position - pos[j]) * tr + dur[j]
        a = int(np.argmin(finish))
        free[a] = finish[a]
        position[a] = pos[j]
        assignment.append(a)
    return float(free.max()), assignment
