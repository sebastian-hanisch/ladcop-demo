"""Erschöpfende Referenzlösung für DPOPs EIGENES Summen-Ziel (siehe
`dcop_graph.objective_value`) - der Korrektheits-Check für die DPOP-
Implementierung selbst, unabhängig von der (separaten) Frage, ob dieses Ziel
mit dem echten Makespan übereinstimmt (siehe `dcop_evaluation.py`).

Nur für kleine Instanzen praktikabel (n_agents**n_jobs Kombinationen) - beim
Regler-Maximum dieser Demo (n_jobs=8, n_agents=4: 65536 Kombinationen) noch
gut interaktiv."""

from itertools import product

from dcop_graph import objective_value


def solve_bruteforce_dcop(instance):
    best_cost = None
    best_assignment = None
    for values in product(range(instance.n_agents), repeat=instance.n_jobs):
        assignment = dict(enumerate(values))
        cost = objective_value(instance, assignment)
        if best_cost is None or cost < best_cost:
            best_cost, best_assignment = cost, assignment
    return best_cost, best_assignment
