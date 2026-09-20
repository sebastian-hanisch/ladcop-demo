import pytest

import cn_constants as C
from cn_scenario import generate_instance
from ladcop_evaluation import (
    COLUMNS, capacity_ladder, compare_instance, heldout_sweep, learn_lottery, scaling_table, schedules_from_assignment,
    verdict,
)
from ladcop_messages import LADDER
from ladcop_learn_model import train_weights
from ladcop_model import HAND_BALANCE_WEIGHTS


def _cmp(**overrides):
    base = {
        "exact_feasible": True, "entries": 100, "message_size": 153,
        "vs_cnp": {"msg_learned": -10.0, "search_learned": -10.0, "dpop_learned": -12.0, "dpop_vehicle": 8.0},
        "fidelity_gap_pct": {"learned": 0.5},
    }
    base.update(overrides)
    return base


def test_compare_instance_columns_and_reference_clamp():
    instance = generate_instance(6, 3, 0.3, 1.0, 5)
    cmp = compare_instance(instance, HAND_BALANCE_WEIGHTS, "quad_lb_wf", 500, cpsat_limit=5.0)
    assert cmp["exact_feasible"] and set(COLUMNS) == set(cmp["cells"])
    assert all(cmp["cells"][name] is not None for name in COLUMNS)
    assert cmp["reference"] <= min(c["makespan"] for c in cmp["cells"].values()) + 1e-9   # CP-SAT/Klemmung
    assert all(len(cmp["cells"][name]["assignment"]) == 6 for name in COLUMNS)
    assert cmp["vs_cnp"]["cnp"] == 0.0
    # gelernte Nachrichten sind auf dem eigenen Ziel nie besser als der exakte DPOP (Lücke >= 0)
    assert cmp["fidelity_gap_pct"]["learned"] >= -1e-9


def test_compare_instance_exact_infeasible_branch():
    instance = generate_instance(14, 4, 0.3, 1.0, 3)          # 4^13 Einträge: weit über der Grenze
    cmp = compare_instance(instance, HAND_BALANCE_WEIGHTS, "quad_lb_wf", 500, include_cpsat=False)
    assert not cmp["exact_feasible"]
    assert cmp["cells"]["dpop_vehicle"] is None and cmp["cells"]["dpop_learned"] is None
    assert cmp["cells"]["msg_learned"] is not None and cmp["cells"]["search_learned"] is not None
    assert cmp["fidelity_gap_pct"] == {} and cmp["reference"] is None
    assert verdict(cmp)[1] == "exact_infeasible"


def test_verdict_cascade_order():
    assert verdict(_cmp(exact_feasible=False))[:2] == ("warning", "exact_infeasible")
    worse = _cmp(vs_cnp={"msg_learned": 6.0, "search_learned": 6.0, "dpop_learned": -12.0, "dpop_vehicle": 8.0})
    assert verdict(worse)[:2] == ("warning", "messages_worse_than_cnp")
    gap = _cmp(fidelity_gap_pct={"learned": 3.5})
    assert verdict(gap)[:2] == ("warning", "message_gap")
    beaten = _cmp(vs_cnp={"msg_learned": -5.0, "search_learned": -9.0, "dpop_learned": -12.0, "dpop_vehicle": 8.0})
    assert verdict(beaten)[:2] == ("warning", "search_beats_messages")
    assert verdict(_cmp())[:2] == ("success", "model_effect")
    neutral = _cmp(vs_cnp={"msg_learned": 0.0, "search_learned": 0.0, "dpop_learned": -1.0, "dpop_vehicle": 8.0})
    assert verdict(neutral)[:2] == ("info", "neutral")
    # Reihenfolge: exakt-unmöglich schlägt alles andere; "schlechter als CNP" schlägt die Lücke
    assert verdict(_cmp(exact_feasible=False, fidelity_gap_pct={"learned": 9.0}))[1] == "exact_infeasible"
    assert verdict(_cmp(vs_cnp=worse["vs_cnp"], fidelity_gap_pct={"learned": 9.0}))[1] == "messages_worse_than_cnp"


def test_heldout_sweep_structure_and_fixed_instances():
    a = heldout_sweep(6, 3, 0.3, 1.0, HAND_BALANCE_WEIGHTS, "quad_lb_wf", 400, n_instances=3, cpsat_limit=5.0)
    b = heldout_sweep(6, 3, 0.3, 1.0, HAND_BALANCE_WEIGHTS, "quad_lb_wf", 400, n_instances=3, cpsat_limit=5.0)
    assert a["n_instances"] == 3
    for name in ("cnp", "dpop_vehicle", "dpop_learned", "msg_learned", "search_learned", "cpsat"):
        assert name in a["stats"], name
    assert "msg_vehicle" not in a["stats"]
    assert a["stats"]["cnp"]["mean_pct"] == 0.0
    assert a["stats"]["msg_learned"]["mean_pct"] == b["stats"]["msg_learned"]["mean_pct"]      # feste Instanzen
    assert "msg_vs_exact" in a and 0.0 <= a["msg_vs_exact"]["better_frac"] <= 1.0


def test_capacity_ladder_grows_and_greedy_is_the_smallest_capacity():
    rows = capacity_ladder(7, 3, 0.3, 1.0, HAND_BALANCE_WEIGHTS, n_samples=400, n_instances=3)
    assert [r["feature_set"] for r in rows] == list(LADDER)
    sizes = [r["size"] for r in rows]
    assert sizes[0] == 0 and sizes == sorted(sizes)
    assert all(r["surrogate_gap_pct"] is not None and r["surrogate_gap_pct"] >= -1e-9 for r in rows)


def test_scaling_table_and_helpers():
    rows = scaling_table(4, [8, 12, 13, 16], "quad_lb_wf")
    assert [r["entries"] for r in rows] == [4 ** 7, 4 ** 11, 4 ** 12, 4 ** 15]
    assert [r["exact_feasible"] for r in rows] == [True, True, False, False]
    assert all(r["message_size"] == 231 for r in rows)
    assert schedules_from_assignment([0, 1, 0, 2], 3) == {0: (0, 2), 1: (1,), 2: (3,)}


def test_learn_lottery_structure():
    lot = learn_lottery(6, 3, 0.3, 1.0, n_train=10, iterations=3, first_seed=4, n_seeds=2, n_instances=3)
    assert [r["seed"] for r in lot["rows"]] == [4, 5]
    assert lot["min_pct"] <= lot["max_pct"] and lot["std_pct"] >= 0.0


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_presets_produce_expected_bands_and_verdict(name):
    p = C.PRESETS[name]
    instance = generate_instance(
        p["n_jobs"], p["n_agents"], p["duration_variability"], p["travel_time_per_unit"], p["seed"],
    )
    weights = train_weights(
        p["n_agents"], p["duration_variability"], p["travel_time_per_unit"], p["train_instances"], p["cem_iterations"],
        p["learn_seed"],
    ).weights
    cmp = compare_instance(instance, weights, p["feature_set"], p["samples"], 0, include_cpsat=False)
    bands = C.PRESET_EXPECTED_BANDS[name]
    for column, (lo, hi) in bands.items():
        assert lo <= cmp["vs_cnp"][column] <= hi, f"{name}: {column}={cmp['vs_cnp'][column]:.1f}"
    assert verdict(cmp)[1] == C.PRESET_EXPECTED_VERDICTS[name], name
