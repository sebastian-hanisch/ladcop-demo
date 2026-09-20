import numpy as np
import pytest

import cn_constants as C
from cn_scenario import generate_instance
from ladcop_dpop_np import solve_dpop_np, util_tables_np
from ladcop_learn_model import fitness, train_set, train_weights
from ladcop_messages import (
    FEATURE_SETS, LADDER, all_prefixes, fit_messages, message_error, message_value, n_features, solve_with_messages,
)
from ladcop_model import (
    HAND_BALANCE_WEIGHTS, VEHICLE_WEIGHTS, cnp_solution, cost_arrays, instance_arrays, makespan_np, objective_np,
)
from ladcop_search import local_search


def _problem(n_jobs, n_agents, seed, weights=HAND_BALANCE_WEIGHTS):
    arrays = instance_arrays(generate_instance(n_jobs, n_agents, 0.3, 1.0, seed))
    unary, pair = cost_arrays(arrays, weights)
    return arrays, unary, pair


# --- Nachrichten ------------------------------------------------------------------------------------

def test_message_sizes_are_independent_of_n_and_match_documented_values():
    assert n_features(3, "quad_lb_wf") == 153 and n_features(4, "quad_lb_wf") == 231
    assert n_features(3, "greedy") == 0
    assert n_features(4, "linear") == 1 + 4 * 4          # 1 + (Kopplung, Dauer, Anzahl, Position) je Agent
    assert n_features(4, "linear_generic") == 1 + 3 * 4
    sizes = [n_features(4, name) for name in LADDER]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)   # die Leiter wächst streng
    _, unary, pair = _problem(8, 4, 1)
    messages = fit_messages(unary, pair, np.ones(8), np.ones(8), "quad_lb_wf", 300, 0)
    assert all(len(b) == 231 for b in messages.betas[:8]) and messages.size == 231


def test_closed_form_without_coupling_messages_recover_suffix_minima():
    """S = 0: die exakte UTIL-Tabelle von Auftrag j ist die Suffix-Summe der min_a U[l, a] - unabhängig vom Präfix. Schon
    eine lineare Nachricht (mit ungeschrumpftem Bias) trifft sie exakt."""
    arrays, unary, _ = _problem(6, 3, 2)
    pair = np.zeros((6, 6))
    tables = util_tables_np(unary, pair)
    messages = fit_messages(unary, pair, arrays.dur, arrays.pos, "linear_generic", 400, 0)
    suffix = np.cumsum(unary.min(1)[::-1])[::-1]
    for j in range(1, 6):
        assert np.allclose(tables[j], suffix[j])
        prefix = all_prefixes(j, 3)
        approx = message_value(messages, j, prefix, unary, pair, arrays.dur, arrays.pos)
        assert np.abs(approx - suffix[j]).max() < 1e-6


def test_capacity_limit_table_features_reproduce_exact_util_tables():
    """Mit One-Hot-Merkmalen des vollen Separators und allen k^j Präfixen interpoliert die Regression die exakte Tabelle
    (Kapazität = Tabellengröße: max. Abweichung 0)."""
    arrays, unary, pair = _problem(5, 3, 4)
    tables = util_tables_np(unary, pair)
    for j in range(1, 5):
        prefix = all_prefixes(j, 3)
        features = np.eye(len(prefix))          # ein Merkmal je Tabelleneintrag
        exact = tables[j][tuple(prefix.T)]
        solution, *_ = np.linalg.lstsq(features, exact, rcond=None)
        assert np.abs(features @ solution - exact).max() < 1e-9


def test_fitted_messages_have_moderate_error_on_the_exact_tables():
    arrays, unary, pair = _problem(8, 3, 6)
    tables = util_tables_np(unary, pair)
    messages = fit_messages(unary, pair, arrays.dur, arrays.pos, "quad_lb_wf", 3000, 0)
    errors = message_error(messages, tables, unary, pair, arrays.dur, arrays.pos)
    assert len(errors) == 6 and max(errors) < 0.6 and float(np.median(errors)) < 0.35


def test_greedy_messages_are_the_myopic_local_choice():
    arrays, unary, pair = _problem(6, 3, 3)
    messages = fit_messages(unary, pair, arrays.dur, arrays.pos, "greedy", 100, 0)
    assignment = solve_with_messages(messages, unary, pair, arrays.dur, arrays.pos)
    manual = []
    for j in range(6):
        cost = [unary[j, a] + sum(pair[i, j] for i in range(j) if manual[i] == a) for a in range(3)]
        manual.append(int(np.argmin(cost)))
    assert assignment == manual


def test_message_solution_never_beats_the_exact_optimum_on_the_same_model():
    for seed in range(10):
        arrays, unary, pair = _problem(7, 3, seed)
        exact = solve_dpop_np(unary, pair)[1]
        messages = fit_messages(unary, pair, arrays.dur, arrays.pos, "quad_lb_wf", 800, seed)
        assignment = solve_with_messages(messages, unary, pair, arrays.dur, arrays.pos)
        assert objective_np(unary, pair, assignment) >= exact - 1e-9, f"seed={seed}"


def test_fit_determinism_and_fit_seed_dependence():
    arrays, unary, pair = _problem(6, 3, 1)
    a = fit_messages(unary, pair, arrays.dur, arrays.pos, "quad_lb", 300, 5)
    b = fit_messages(unary, pair, arrays.dur, arrays.pos, "quad_lb", 300, 5)
    c = fit_messages(unary, pair, arrays.dur, arrays.pos, "quad_lb", 300, 6)
    assert all(np.array_equal(x, y) for x, y in zip(a.betas[:6], b.betas[:6]))
    assert any(not np.array_equal(x, y) for x, y in zip(a.betas[:6], c.betas[:6]))


# --- Lokale Suche -----------------------------------------------------------------------------------

def test_local_search_ends_in_a_local_optimum_and_never_beats_exact():
    for seed in range(8):
        arrays, unary, pair = _problem(7, 3, seed)
        assignment = local_search(unary, pair, 3, restarts=5, seed=seed)
        cost = objective_np(unary, pair, assignment)
        assert cost >= solve_dpop_np(unary, pair)[1] - 1e-9
        for j in range(7):
            for a in range(3):
                moved = list(assignment)
                moved[j] = a
                assert objective_np(unary, pair, moved) >= cost - 1e-9, f"seed={seed} job={j}"
    assert local_search(unary, pair, 3, 3, 1) == local_search(unary, pair, 3, 3, 1)


# --- Modell lernen ------------------------------------------------------------------------------------

def test_zero_iterations_reproduce_the_vehicle_model():
    result = train_weights(3, 0.3, 1.0, n_instances=10, iterations=0, learn_seed=0)
    assert result.weights == VEHICLE_WEIGHTS and result.history == ()


def test_training_is_deterministic_seed_dependent_and_improves_the_training_fitness():
    a = train_weights(3, 0.3, 1.0, 12, 4, 0)
    b = train_weights(3, 0.3, 1.0, 12, 4, 0)
    c = train_weights(3, 0.3, 1.0, 12, 4, 1)
    assert a.weights == b.weights and a.history == b.history
    assert a.weights != c.weights
    assert a.history[-1] <= a.start_fitness
    assert list(a.history) == sorted(a.history, reverse=True)          # bester Wert wird nie schlechter


def test_training_instances_are_decoupled_from_heldout_and_demo_seeds():
    assert C.TRAIN_SEED_BASE > C.HELDOUT_SEED_BASE + 10_000 and C.HELDOUT_SEED_BASE > 10_000
    a = train_set(3, 0.3, 1.0, 5, 0)
    b = train_set(3, 0.3, 1.0, 5, 0)
    assert [x[1] for x in a] == [x[1] for x in b]


def test_two_by_two_signs_vehicle_model_loses_to_cnp_learned_model_wins():
    """Modell-Effekt als Vorzeichen-Test über feste Held-out-Instanzen (n=8, k=3): das Fahrzeug-Modell ist im Mittel
    schlechter als Contract Net, das gelernte besser (gemessen +9.8 % / -11.3 %)."""
    weights = train_weights(3, 0.3, 1.0, 30, 12, 0).weights
    vehicle, learned = [], []
    for i in range(12):
        arrays = instance_arrays(generate_instance(8, 3, 0.3, 1.0, C.HELDOUT_SEED_BASE + i))
        cnp = cnp_solution(arrays)[0]
        vehicle.append(makespan_np(arrays, solve_dpop_np(*cost_arrays(arrays, VEHICLE_WEIGHTS))[0]) / cnp)
        learned.append(makespan_np(arrays, solve_dpop_np(*cost_arrays(arrays, weights))[0]) / cnp)
    assert np.mean(vehicle) > 1.03 and np.mean(learned) < 0.95
    assert fitness(weights.as_vector(), train_set(3, 0.3, 1.0, 10, 0)) < 1.0
