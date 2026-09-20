"""Auswertung: alle Verfahren auf DERSELBEN Instanz bzw. demselben festen Held-out-Set.

Spalten: Contract Net (roh) / DPOP mit Fahrzeug-Modell / DPOP mit gelerntem Modell / gelernte Nachrichten (auf dem
Fahrzeug- und auf dem gelernten Modell) / lokale Suche (gelerntes Modell) / CP-SAT (echtes Optimum bzw. beste gefundene
Lösung). Ein 2x2-Design trennt sauber den MODELL-Effekt (Fahrzeug -> gelernt) vom LÖSER-Effekt (exakt -> Nachrichten).

Prozentangaben "vs. CNP": (Verfahren - CNP) / CNP * 100 - negativ heißt besser als Contract Net."""

import time

import numpy as np

import cn_constants as C
from cn_ortools_reference import solve_with_ortools
from cn_scenario import generate_instance
from ladcop_dpop_np import entries, exact_feasible, solve_dpop_np
from ladcop_learn_model import train_weights
from ladcop_messages import (
    DEFAULT_FEATURE_SET, FEATURE_SETS, LADDER, fit_messages, n_features, solve_with_messages,
)
from ladcop_model import (
    VEHICLE_WEIGHTS, cnp_solution, cost_arrays, instance_arrays, makespan_np, objective_np,
)
from ladcop_search import local_search

COLUMNS = ("cnp", "dpop_vehicle", "dpop_learned", "msg_vehicle", "msg_learned", "search_learned", "cpsat")
COLUMN_LABELS = {
    "cnp": "Contract Net (roh)",
    "dpop_vehicle": "DPOP, Fahrzeug-Modell (exakt)",
    "dpop_learned": "DPOP, gelerntes Modell (exakt)",
    "msg_vehicle": "Gelernte Nachrichten, Fahrzeug-Modell",
    "msg_learned": "Gelernte Nachrichten, gelerntes Modell",
    "search_learned": "Lokale Suche, gelerntes Modell",
    "cpsat": "CP-SAT (zentral)",
}


def pct_vs(value, reference):
    return (value - reference) / reference * 100.0 if reference > 0 else 0.0


def schedules_from_assignment(assignment, n_agents):
    """Zuteilung (Liste Agent je Auftrag) -> {agent: Tupel Auftragsindizes} für die Gantt-Funktionen."""
    schedules = {a: [] for a in range(n_agents)}
    for job, agent in enumerate(assignment):
        schedules[agent].append(job)
    return {a: tuple(jobs) for a, jobs in schedules.items()}


def _cell(arrays, assignment, seconds, unary=None, pair=None):
    return {
        "assignment": [int(a) for a in assignment], "makespan": makespan_np(arrays, assignment), "seconds": seconds,
        "surrogate": None if unary is None else objective_np(unary, pair, assignment),
    }


def compare_instance(
    instance, weights_learned, feature_set=None, n_samples=C.DEFAULT_SAMPLES, fit_seed=0,
    cpsat_limit=C.ORTOOLS_TIME_LIMIT_SECONDS, include_cpsat=True, include_vehicle_messages=True,
):
    """Alle Spalten für eine Instanz (exakte Spalten nur, wenn die größte Tabelle in die Grenze passt)."""
    feature_set = feature_set or DEFAULT_FEATURE_SET
    arrays = instance_arrays(instance)
    n, k = instance.n_jobs, instance.n_agents
    feasible = exact_feasible(n, k)
    cnp_makespan, cnp_assignment = cnp_solution(arrays)
    cells = {"cnp": _cell(arrays, cnp_assignment, 0.0)}
    models = {"vehicle": cost_arrays(arrays, VEHICLE_WEIGHTS), "learned": cost_arrays(arrays, weights_learned)}
    exact_cost = {}

    for name, (unary, pair) in models.items():
        if feasible:
            started = time.perf_counter()
            assignment, cost, _ = solve_dpop_np(unary, pair)
            cells[f"dpop_{name}"] = _cell(arrays, assignment, time.perf_counter() - started, unary, pair)
            exact_cost[name] = cost
        else:
            cells[f"dpop_{name}"] = None
    for name, (unary, pair) in models.items():
        if name == "vehicle" and not include_vehicle_messages:
            cells["msg_vehicle"] = None
            continue
        messages = fit_messages(unary, pair, arrays.dur, arrays.pos, feature_set, n_samples, fit_seed)
        started = time.perf_counter()
        assignment = solve_with_messages(messages, unary, pair, arrays.dur, arrays.pos)
        cell = _cell(arrays, assignment, messages.fit_seconds + time.perf_counter() - started, unary, pair)
        cell["fit_seconds"] = messages.fit_seconds
        cells[f"msg_{name}"] = cell
    unary, pair = models["learned"]
    started = time.perf_counter()
    assignment = local_search(unary, pair, k, restarts=10, seed=fit_seed)
    cells["search_learned"] = _cell(arrays, assignment, time.perf_counter() - started, unary, pair)

    cells["cpsat"] = None
    if include_cpsat:
        result = solve_with_ortools(instance, time_limit_seconds=cpsat_limit)
        if result.feasible:
            cells["cpsat"] = {
                "assignment": [result.assignment[j] for j in range(n)], "makespan": result.makespan,
                "seconds": result.wall_time_ms / 1000.0, "optimal": result.optimal, "surrogate": None,
            }

    # CP-SAT rundet Zeiten AUF: sein Wert kann knapp über einem tatsächlich erreichbaren Zeitplan liegen. Ist der Lauf
    # bewiesen optimal, ist das echte Optimum <= jeder zulässigen Lösung - der Wert wird dann nie größer als der beste
    # bekannte Zeitplan angesetzt. Bei unbewiesenen Läufen (Zeitlimit) bleibt der Solver-Wert unverändert: dann kann ein
    # dezentrales Verfahren den Incumbent sogar schlagen.
    reference = None
    if cells["cpsat"] is not None:
        others = [c["makespan"] for name, c in cells.items() if c is not None and name != "cpsat"]
        cells["cpsat"]["makespan_raw"] = cells["cpsat"]["makespan"]
        if cells["cpsat"]["optimal"]:
            cells["cpsat"]["makespan"] = min([cells["cpsat"]["makespan"]] + others)
        reference = cells["cpsat"]["makespan"]
    fidelity = {}
    for name in ("vehicle", "learned"):
        cell = cells.get(f"msg_{name}")
        if cell is not None and name in exact_cost:
            fidelity[name] = pct_vs(cell["surrogate"], exact_cost[name])
    return {
        "n_jobs": n, "n_agents": k, "exact_feasible": feasible, "entries": entries(n, k),
        "message_size": n_features(k, feature_set), "feature_set": feature_set, "cells": cells,
        "cnp_makespan": cnp_makespan, "reference": reference,
        "vs_cnp": {name: pct_vs(c["makespan"], cnp_makespan) for name, c in cells.items() if c is not None},
        "fidelity_gap_pct": fidelity,
    }


def _heldout_instances(n_jobs, n_agents, var, travel, n_instances):
    return [generate_instance(n_jobs, n_agents, var, travel, C.HELDOUT_SEED_BASE + i) for i in range(n_instances)]


def heldout_sweep(
    n_jobs, n_agents, var, travel, weights_learned, feature_set=None, n_samples=C.DEFAULT_SAMPLES, n_instances=None,
    cpsat_limit=C.ORTOOLS_TIME_LIMIT_SECONDS,
):
    """Feste Held-out-Instanzen (unabhängig vom Demo-Seed): Statistik je Spalte. Das Fahrzeug-Nachrichten-Feld entfällt
    (Kosten), der 2x2-Vergleich nutzt dpop_vehicle/dpop_learned/msg_learned."""
    n_instances = n_instances or (C.N_HELDOUT_LARGE if n_jobs >= C.LARGE_N_THRESHOLD else C.N_HELDOUT)
    per_column = {name: [] for name in COLUMNS}
    per_column["cnp"] = []
    fidelity = []
    for instance in _heldout_instances(n_jobs, n_agents, var, travel, n_instances):
        cmp = compare_instance(
            instance, weights_learned, feature_set, n_samples, cpsat_limit=cpsat_limit, include_vehicle_messages=False,
        )
        for name in COLUMNS:
            cell = cmp["cells"].get(name)
            per_column[name].append(None if cell is None else cell["makespan"])
        if "learned" in cmp["fidelity_gap_pct"]:
            fidelity.append(cmp["fidelity_gap_pct"]["learned"])
    cnp = np.array(per_column["cnp"])
    stats = {}
    for name in COLUMNS:
        values = per_column[name]
        if any(v is None for v in values):
            continue
        ms = np.array(values)
        ratio = (ms - cnp) / cnp * 100.0
        stats[name] = {
            "mean_pct": float(ratio.mean()), "win_frac": float((ms < cnp - 1e-9).mean()),
            "lose_frac": float((ms > cnp + 1e-9).mean()), "worst_pct": float(ratio.max()), "mean_makespan": float(ms.mean()),
        }
    if "cpsat" in stats:
        optimum = np.minimum.reduce([np.array(per_column[n]) for n in COLUMNS if n in stats])
        for name in stats:
            stats[name]["gap_to_best_pct"] = float(((np.array(per_column[name]) - optimum) / optimum * 100.0).mean())
    out = {"n_instances": n_instances, "stats": stats, "fidelity_gap_mean_pct": float(np.mean(fidelity)) if fidelity else None}
    if "dpop_learned" in per_column and all(v is not None for v in per_column["dpop_learned"]) and "msg_learned" in stats:
        exact_ms, msg_ms = np.array(per_column["dpop_learned"]), np.array(per_column["msg_learned"])
        out["msg_vs_exact"] = {
            "better_frac": float((msg_ms < exact_ms - 1e-9).mean()), "worse_frac": float((msg_ms > exact_ms + 1e-9).mean()),
            "mean_pct": float(((msg_ms - exact_ms) / cnp * 100.0).mean()),
        }
    return out


def capacity_ladder(
    n_jobs, n_agents, var, travel, weights_learned, n_samples=C.DEFAULT_SAMPLES, n_instances=10,
):
    """Wächst die Nachrichten-Kapazität (Merkmale), nähert sich das Ergebnis dem exakten DPOP: je Stufe mittlerer Makespan
    vs. CNP und Surrogat-Lücke zum exakten Optimum (nur wo exakt machbar)."""
    feasible = exact_feasible(n_jobs, n_agents)
    rows = []
    instances = _heldout_instances(n_jobs, n_agents, var, travel, n_instances)
    for name in LADDER:
        ratios, gaps = [], []
        for instance in instances:
            arrays = instance_arrays(instance)
            unary, pair = cost_arrays(arrays, weights_learned)
            cnp = cnp_solution(arrays)[0]
            messages = fit_messages(unary, pair, arrays.dur, arrays.pos, name, n_samples, 0)
            assignment = solve_with_messages(messages, unary, pair, arrays.dur, arrays.pos)
            ratios.append(makespan_np(arrays, assignment) / cnp)
            if feasible:
                exact = solve_dpop_np(unary, pair)[1]
                gaps.append(pct_vs(objective_np(unary, pair, assignment), exact))
        rows.append({
            "feature_set": name, "label": FEATURE_SETS[name].label, "size": n_features(n_agents, name),
            "vs_cnp_pct": (float(np.mean(ratios)) - 1.0) * 100.0, "surrogate_gap_pct": float(np.mean(gaps)) if gaps else None,
        })
    return rows


def scaling_table(n_agents, n_values, feature_set):
    """Analytisch: größte exakte Tabelle k^(n-1) gegen die feste Nachrichtengröße F."""
    size = n_features(n_agents, feature_set)
    return [
        {"n_jobs": n, "entries": entries(n, n_agents), "exact_feasible": exact_feasible(n, n_agents), "message_size": size}
        for n in n_values
    ]


def learn_lottery(
    n_jobs, n_agents, var, travel, n_train, iterations, first_seed, n_seeds=C.N_LEARN_LOTTERY_SEEDS, n_instances=10,
):
    """Trainiert das Modell mit mehreren Lern-Seeds (CEM-Zufall) und misst je Lauf das exakte DPOP-Ergebnis auf festen
    Held-out-Instanzen - wie stark hängt das gelernte Modell vom Zufall des Lernens ab?"""
    instances = _heldout_instances(n_jobs, n_agents, var, travel, n_instances)
    rows = []
    for offset in range(n_seeds):
        seed = first_seed + offset
        result = train_weights(n_agents, var, travel, n_train, iterations, seed)
        ratios = []
        for instance in instances:
            arrays = instance_arrays(instance)
            unary, pair = cost_arrays(arrays, result.weights)
            assignment = solve_dpop_np(unary, pair)[0]
            ratios.append(makespan_np(arrays, assignment) / cnp_solution(arrays)[0])
        rows.append({"seed": seed, "weights": result.weights, "vs_cnp_pct": (float(np.mean(ratios)) - 1.0) * 100.0})
    values = [r["vs_cnp_pct"] for r in rows]
    return {"rows": rows, "std_pct": float(np.std(values)), "min_pct": min(values), "max_pct": max(values)}


def verdict(cmp):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten). Der Text wird in der App gesetzt."""
    vs = cmp["vs_cnp"]
    data = {
        "msg_pct": vs.get("msg_learned"), "search_pct": vs.get("search_learned"), "dpop_learned_pct": vs.get("dpop_learned"),
        "dpop_vehicle_pct": vs.get("dpop_vehicle"), "gap_pct": cmp["fidelity_gap_pct"].get("learned"),
        "entries": cmp["entries"], "message_size": cmp["message_size"],
    }
    if not cmp["exact_feasible"]:
        return "warning", "exact_infeasible", data
    if data["msg_pct"] is not None and data["msg_pct"] >= C.MESSAGES_WORSE_THAN_CNP_PCT:
        return "warning", "messages_worse_than_cnp", data
    if data["gap_pct"] is not None and data["gap_pct"] >= C.MESSAGE_GAP_WARNING_PCT:
        return "warning", "message_gap", data
    if data["search_pct"] is not None and data["msg_pct"] is not None and data["search_pct"] < data["msg_pct"] - 0.5:
        return "warning", "search_beats_messages", data
    if data["dpop_learned_pct"] is not None and data["dpop_learned_pct"] <= -C.MODEL_EFFECT_SUCCESS_PCT:
        return "success", "model_effect", data
    return "info", "neutral", data
