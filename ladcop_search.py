"""Lokale Suche auf demselben DCOP-Ziel (DSA/MGM-artig, aber zentral simuliert): beste Einzel-Umzuweisung, wiederholt
bis zum lokalen Optimum, mit Neustarts. Die stärkste ehrliche Konkurrenz der gelernten Nachrichten - 1 bis 2 ms."""

import numpy as np

from ladcop_model import objective_np


def local_search(unary, pair, n_agents, restarts=10, seed=0):
    n = unary.shape[0]
    rng = np.random.default_rng([seed, 73])
    upper = np.triu(pair, 1)
    symmetric = upper + upper.T
    best_cost, best_assignment = None, None
    for _ in range(restarts):
        x = rng.integers(0, n_agents, n)
        for _ in range(200):
            one_hot = np.eye(n_agents)[x]
            cost_if = unary + symmetric @ one_hot           # Kosten von Auftrag j bei Agent a, gegeben die übrigen
            current = cost_if[np.arange(n), x]
            delta = cost_if - current[:, None]
            j, a = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[j, a] >= -1e-9:
                break
            x[j] = a
        cost = objective_np(unary, pair, x)
        if best_cost is None or cost < best_cost:
            best_cost, best_assignment = cost, x.copy()
    return [int(v) for v in best_assignment]
