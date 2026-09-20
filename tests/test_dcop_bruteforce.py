from cn_scenario import Instance, Job
from dcop_bruteforce import solve_bruteforce_dcop
from dcop_graph import objective_value


def _instance(jobs, n_agents=2, agent_start_positions=None, travel_time_per_unit=1.0):
    if agent_start_positions is None:
        agent_start_positions = tuple(0.0 for _ in range(n_agents))
    return Instance(
        n_jobs=len(jobs), n_agents=n_agents, jobs=jobs,
        agent_start_positions=agent_start_positions, travel_time_per_unit=travel_time_per_unit,
    )


def test_bruteforce_matches_hand_computation():
    jobs = (
        Job(index=0, position=1.0, duration=2.0),
        Job(index=1, position=9.0, duration=3.0),
        Job(index=2, position=5.0, duration=1.0),
    )
    instance = _instance(jobs, n_agents=2, agent_start_positions=(0.0, 10.0))
    cost, assignment = solve_bruteforce_dcop(instance)
    assert cost == 17.0
    assert assignment == {0: 0, 1: 1, 2: 0}
    assert objective_value(instance, assignment) == cost


def test_bruteforce_tie_break_is_deterministic_among_equal_cost_assignments():
    # Beide Agenten bei derselben Position -> Unaerkosten sind fuer jeden Auftrag
    # unabhaengig vom Agenten (Summe konstant = 1+2+3 = 6). Die Paarkosten
    # bevorzugen, benachbarte Auftraege zu TRENNEN (0 Paarkosten bei
    # verschiedenen Agenten) - das guenstigste ist, Auftrag 2 (am weitesten
    # von den anderen beiden) allein zu lassen: Gesamtkosten 6 + |0-1| = 7.0.
    # Mehrere Zuteilungen erreichen dieses Minimum (Agenten sind symmetrisch) -
    # itertools.product iteriert Auftrag 0 am langsamsten, Auftrag 2 am
    # schnellsten, und "nur bei striktem <" behaelt die ERSTE gefundene
    # Minimalloesung - das macht das Ergebnis deterministisch und testbar.
    jobs = tuple(Job(index=i, position=float(i), duration=1.0) for i in range(3))
    instance = _instance(jobs, n_agents=2, agent_start_positions=(0.0, 0.0))
    cost, assignment = solve_bruteforce_dcop(instance)
    assert cost == 7.0
    assert assignment == {0: 0, 1: 0, 2: 1}
    assert objective_value(instance, assignment) == cost
