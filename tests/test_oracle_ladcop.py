"""Orakel-Test: Modellkosten, numpy-DPOP, UTIL-Tabellen, lokale Suche und Nachrichten-Merkmale gegen unabhängige Schleifen-
Implementierungen aus der Dokumentation (Aufzählung aller Zuteilungen, Rekursion, Wasserfüllung per Bisektion)."""

import itertools
import math
import random
from functools import lru_cache

import numpy as np

from cn_scenario import Instance, Job, generate_instance

import ladcop_dpop_np as DN
import ladcop_messages as MSG
import ladcop_model as LM
import ladcop_search as LS


def _weights(rng, integral):
    if integral:
        return LM.ModelWeights(float(rng.randint(0, 3)), float(rng.randint(0, 3)), float(rng.randint(0, 3)),
                               float(rng.choice([0, 10, 20])), float(rng.randint(-2, 3)))
    return LM.ModelWeights(*[rng.uniform(-0.5, 2) for _ in range(5)])


def _costs(inst, w):
    n, k, tau = inst.n_jobs, inst.n_agents, inst.travel_time_per_unit
    unary = [[w.w_travel * abs(inst.agent_start_positions[a] - inst.jobs[j].position) * tau + w.w_duration * inst.jobs[j].duration
              for a in range(k)] for j in range(n)]
    pair = [[w.w_distance * abs(inst.jobs[i].position - inst.jobs[j].position) * tau
             + w.w_load * inst.jobs[i].duration * inst.jobs[j].duration / 10.0 + w.w_count for j in range(n)] for i in range(n)]
    return unary, pair


def _objective(unary, pair, x):
    n = len(x)
    return sum(unary[j][x[j]] for j in range(n)) + sum(pair[i][j] for i in range(n) for j in range(i + 1, n) if x[i] == x[j])


def _int_instance(rng, n, k):
    jobs = tuple(Job(i, float(rng.randint(0, 6)), float(rng.randint(1, 4)) * 10) for i in range(n))
    return Instance(n, k, jobs, tuple(float(rng.randint(0, 6)) for _ in range(k)), 1.0)


def test_model_dpop_tables_and_local_search_match_independent_oracles():
    rng = random.Random(3)
    for it in range(60):
        n, k = rng.randint(1, 6), rng.randint(2, 4)
        integral = it % 2 == 0
        inst = _int_instance(rng, n, k) if integral else generate_instance(n, k, 0.5, rng.uniform(0.2, 2.0), rng.randint(0, 999))
        arrays = LM.instance_arrays(inst)
        unary, pair = LM.cost_arrays(arrays, _weights(rng, integral))
        mu, ms = unary.tolist(), pair.tolist()
        x = [rng.randrange(k) for _ in range(n)]
        assert abs(LM.objective_np(unary, pair, x) - _objective(mu, ms, x)) < 1e-9

        best, best_x = None, None
        for y in itertools.product(range(k), repeat=n):
            v = _objective(mu, ms, y)
            if best is None or v < best - 1e-9:
                best, best_x = v, y
        assignment, cost, entries = DN.solve_dpop_np(unary, pair)
        assert abs(cost - best) < 1e-9 and abs(_objective(mu, ms, assignment) - cost) < 1e-9
        assert entries == k ** (n - 1)
        if integral:                      # exakte Gleichstände: lexikografisch kleinste Optimallösung
            assert tuple(assignment) == best_x

        @lru_cache(None)
        def f(j, prefix):
            if j == n:
                return 0.0
            return min(mu[j][a] + sum(ms[i][j] for i, ai in enumerate(prefix) if ai == a) + f(j + 1, prefix + (a,)) for a in range(k))

        tables = DN.util_tables_np(unary, pair)
        for j in range(n):
            for sep in itertools.product(range(k), repeat=j):
                assert abs(float(tables[j][sep]) - f(j, sep)) < 1e-9

        found = LS.local_search(unary, pair, k, restarts=5, seed=it)
        cost_ls = _objective(mu, ms, found)
        assert cost_ls >= best - 1e-9
        for j in range(n):
            for a in range(k):
                y = list(found)
                y[j] = a
                assert _objective(mu, ms, y) >= cost_ls - 1e-7           # lokales Optimum bezüglich Einzel-Umzuweisung


def _my_features(prefix, j, unary, pair, dur, pos, k, spec):
    n = len(unary)
    coupling = [sum(pair[i][l] for l in range(j, n)) for i in range(j)]
    agg = lambda values: [sum(values[i] for i in range(j) if prefix[i] == a) for a in range(k)]       # noqa: E731
    base = (agg(coupling) if spec.interaction else []) + agg(dur) + agg([1.0] * j) + agg(pos)
    if spec.lb:
        lb = sum(min(unary[l][a] + sum(pair[i][l] for i in range(j) if prefix[i] == a) for a in range(k)) for l in range(j, n))
        base += [lb, lb * lb / 100.0]
    if spec.wf:
        loads, remaining = agg(dur), sum(dur[j:])
        lo, hi = 0.0, max(loads) + remaining + 1.0
        for _ in range(200):                      # Wasserfüllung per Bisektion
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sum(max(v, mid) for v in loads) < sum(loads) + remaining else (lo, mid)
        wf = sum(max(v, (lo + hi) / 2) ** 2 for v in loads)
        base += [wf / 100.0, math.sqrt(wf)]
    cols = [1.0] + base
    if spec.quad:
        cols += [base[a] * base[b] for a in range(len(base)) for b in range(a, len(base))]
    return np.array(cols)


def test_message_features_match_loop_implementation():
    rng = random.Random(8)
    for _ in range(25):
        n, k = rng.randint(2, 6), rng.randint(2, 4)
        inst = generate_instance(n, k, 0.5, rng.uniform(0.2, 2.0), rng.randint(0, 999))
        arrays = LM.instance_arrays(inst)
        unary, pair = LM.cost_arrays(arrays, _weights(rng, False))
        for name, spec in MSG.FEATURE_SETS.items():
            if spec.greedy:
                continue
            j = rng.randint(0, n - 1)
            prefix = [rng.randrange(k) for _ in range(j)]
            got = MSG._features(np.array(prefix, dtype=int).reshape(1, j), j, unary, pair, arrays.dur, arrays.pos, k, spec)[0]
            want = _my_features(prefix, j, unary.tolist(), pair.tolist(), arrays.dur.tolist(), arrays.pos.tolist(), k, spec)
            assert got.shape == want.shape == (MSG.n_features(k, name),)
            assert np.allclose(got, want, rtol=1e-7, atol=1e-7), (name, j)


def test_documented_message_sizes_and_exact_limits():
    assert (MSG.n_features(3, "quad_lb_wf"), MSG.n_features(4, "quad_lb_wf")) == (153, 231)
    limits = {k: max(n for n in range(1, 40) if DN.exact_feasible(n, k)) for k in (2, 3, 4)}
    assert limits == {2: 23, 3: 14, 4: 12}
