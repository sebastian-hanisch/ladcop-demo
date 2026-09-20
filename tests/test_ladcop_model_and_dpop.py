import numpy as np
import pytest

import cn_constants as C
from cn_protocol import run_protocol
from cn_scenario import Instance, Job, generate_instance
from cn_schedule import schedule_from_assignment
from dcop_bruteforce import solve_bruteforce_dcop
from dcop_dpop import solve_dcop
from dcop_graph import objective_value, pairwise_cost, unary_cost
from ladcop_dpop_np import entries, exact_feasible, solve_dpop_np, util_tables_np
from ladcop_model import (
    HAND_BALANCE_WEIGHTS, VEHICLE_WEIGHTS, ModelWeights, cnp_solution, cost_arrays, instance_arrays, makespan_np,
    objective_np,
)


def _hand_instance():
    return Instance(
        n_jobs=3, n_agents=2, jobs=(Job(0, 1.0, 2.0), Job(1, 9.0, 3.0), Job(2, 5.0, 1.0)),
        agent_start_positions=(0.0, 10.0), travel_time_per_unit=1.0,
    )


# --- Modell -----------------------------------------------------------------------------------

def test_vehicle_weights_reproduce_dcop_demo_costs():
    for seed in range(8):
        instance = generate_instance(4 + seed % 4, 2 + seed % 3, 0.4, 0.5 + seed * 0.2, seed)
        unary, pair = cost_arrays(instance_arrays(instance), VEHICLE_WEIGHTS)
        for j in range(instance.n_jobs):
            for a in range(instance.n_agents):
                assert abs(unary[j, a] - unary_cost(instance, j, a)) < 1e-12
            for i in range(j):
                assert abs(pair[i, j] - pairwise_cost(instance, i, j, 0, 0)) < 1e-12
        rng = np.random.default_rng(seed)
        assignment = rng.integers(0, instance.n_agents, instance.n_jobs)
        assert abs(objective_np(unary, pair, assignment) - objective_value(
            instance, {j: int(a) for j, a in enumerate(assignment)})) < 1e-9


def test_real_makespan_and_cnp_match_vehicle_functions():
    for seed in range(40):
        instance = generate_instance(2 + seed % 9, 1 + seed % 4, 0.5, 0.3 + (seed % 5) * 0.4, seed)
        arrays = instance_arrays(instance)
        cnp_ms, cnp_assignment = cnp_solution(arrays)
        assert abs(cnp_ms - run_protocol(instance).makespan) < 1e-9
        schedules = {a: tuple(j for j, x in enumerate(cnp_assignment) if x == a) for a in range(instance.n_agents)}
        assert abs(schedule_from_assignment(instance, schedules)[1] - makespan_np(arrays, cnp_assignment)) < 1e-9


def test_weight_vector_roundtrip():
    weights = ModelWeights(0.7, 1.0, 0.3, 2.0, -0.1)
    assert ModelWeights.from_vector(weights.as_vector()) == weights


def test_duration_weight_is_vacuous():
    """Die Dauer eines Auftrags ist für jeden Agenten gleich: w_duration verschiebt nur eine Konstante."""
    for seed in range(6):
        arrays = instance_arrays(generate_instance(7, 3, 0.4, 1.0, seed))
        results = {
            w: solve_dpop_np(*cost_arrays(arrays, ModelWeights(1.0, w, 1.0, 1.0, 0.0)))[0] for w in (0.0, 1.0, 5.0)
        }
        assert results[0.0] == results[1.0] == results[5.0], f"seed={seed}"


@pytest.mark.parametrize("n_jobs,expected", [(6, 3), (7, 5), (8, 7)])
def test_count_weight_only_gives_balanced_split_closed_form(n_jobs, expected):
    """Nur w_count = 1: das Ziel ist Σ_a C(n_a, 2) - minimal bei ausgeglichener Aufteilung: 6/3 -> 3, 7/3 -> 5, 8/3 -> 7."""
    arrays = instance_arrays(generate_instance(n_jobs, 3, 0.3, 1.0, 1))
    unary, pair = cost_arrays(arrays, ModelWeights(0.0, 0.0, 0.0, 0.0, 1.0))
    assignment, cost, _ = solve_dpop_np(unary, pair)
    assert cost == expected
    counts = np.bincount(assignment, minlength=3)
    assert counts.max() - counts.min() <= 1


# --- Numpy-DPOP -----------------------------------------------------------------------------------

def test_hand_computed_example_matches_dcop_demo():
    unary, pair = cost_arrays(instance_arrays(_hand_instance()), VEHICLE_WEIGHTS)
    assignment, cost, table_size = solve_dpop_np(unary, pair)
    assert assignment == [0, 1, 0] and cost == 17.0 and table_size == 4
    tables = util_tables_np(unary, pair)
    assert np.allclose(tables[2], [[6.0, 10.0], [10.0, 6.0]])   # Tabelle von Auftrag 3 (echte Unentschieden)
    assert np.allclose(tables[1], [14.0, 18.0])
    assert float(tables[0]) == 17.0


def test_numpy_dpop_equals_vehicle_dpop_on_random_instances():
    for seed in range(12):
        instance = generate_instance(4 + seed % 4, 2 + seed % 3, 0.4, 1.0, seed)
        unary, pair = cost_arrays(instance_arrays(instance), VEHICLE_WEIGHTS)
        assignment, cost, _ = solve_dpop_np(unary, pair)
        reference = solve_dcop(instance)
        assert assignment == [reference.assignment[j] for j in range(instance.n_jobs)], f"seed={seed}"
        assert abs(cost - reference.total_cost) < 1e-9
        tables = util_tables_np(unary, pair)
        for j in range(1, instance.n_jobs):
            for separator, (value, _) in reference.util_tables[j].entries.items():
                assert abs(tables[j][separator] - value) < 1e-9, f"seed={seed} job={j}"


def test_numpy_dpop_equals_bruteforce_on_learned_and_hand_weights():
    for seed in range(8):
        instance = generate_instance(4 + seed % 3, 2 + seed % 2, 0.4, 1.0, seed)
        arrays = instance_arrays(instance)
        for weights in (VEHICLE_WEIGHTS, HAND_BALANCE_WEIGHTS, ModelWeights(0.6, 1.0, 0.8, 0.7, 0.1)):
            unary, pair = cost_arrays(arrays, weights)
            _, cost, _ = solve_dpop_np(unary, pair)
            brute = min(
                objective_np(unary, pair, x)
                for x in np.array(np.meshgrid(*([np.arange(instance.n_agents)] * instance.n_jobs), indexing="ij")).reshape(
                    instance.n_jobs, -1).T
            )
            assert abs(cost - brute) < 1e-9, f"seed={seed}"


def test_zero_weights_put_everything_on_agent_zero():
    """Alle Kosten 0 => überall Gleichstand => Tie-Break auf die niedrigste Agenten-ID."""
    arrays = instance_arrays(generate_instance(6, 3, 0.3, 1.0, 3))
    unary, pair = cost_arrays(arrays, ModelWeights(0.0, 0.0, 0.0, 0.0, 0.0))
    assignment, cost, _ = solve_dpop_np(unary, pair)
    assert assignment == [0] * 6 and cost == 0.0


def test_exact_feasibility_limits_match_the_documented_app_limits():
    assert entries(8, 3) == 3 ** 7
    assert exact_feasible(12, 4) and not exact_feasible(13, 4)   # k=4 bis n=12
    assert exact_feasible(14, 3) and not exact_feasible(15, 3)   # k=3 bis n=14
    assert exact_feasible(23, 2) and not exact_feasible(24, 2)   # k=2 bis n=23
    assert C.EXACT_ENTRY_LIMIT == 2 ** 22
