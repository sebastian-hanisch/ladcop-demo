"""DPOP (Distributed Pseudotree Optimization Procedure, Petcu & Faltings 2005) -
ein EXAKTER Löser für DCOPs - angewandt auf das DCOP aus `dcop_graph.py`
(vollständiger Constraint-Graph). Der Pseudo-Baum EINES vollständigen
Graphen entartet immer zu einer Kette (Tiefensuche hat nie einen Grund
zurückzuspringen - jeder unbesuchte Knoten ist Nachbar): Auftrag 0 ist die
Wurzel, Auftrag n-1 das Blatt, und Auftrag k's Separator (seine Vorfahren)
ist exakt (0, ..., k-1) - Tabellengröße n_agents**k.

UTIL-Phase (bottom-up, Blatt zuerst): jeder Auftrag k minimiert über seinen
eigenen Wert die Summe aus seinen lokalen Kosten (Unär- + Paarkosten zu allen
Vorfahren) UND der von seinem Kind bereits berechneten Tabelle (an der Stelle,
die seinem eigenen gewählten Wert entspricht) - und sendet das Ergebnis
(indiziert nur noch über seinen EIGENEN Separator, eine Spalte kürzer) an
seinen Baum-Elternteil. Die Wurzel (Auftrag 0, leerer Separator) erhält am
Ende einen einzelnen Bestwert.

VALUE-Phase (top-down, Wurzel zuerst): die Wurzel entnimmt ihren eigenen Wert
direkt ihrer Tabelle; jeder weitere Auftrag schlägt in seiner EIGENEN
(während der UTIL-Phase behaltenen) Tabelle an der Stelle nach, die den
bereits fixierten Werten seiner Vorfahren entspricht.

Tie-Break: Agenten-IDs aufsteigend durchlaufen, nur bei echter Verbesserung
(striktes <) aktualisieren - das deckt sich exakt mit einer Brute-Force-
Enumeration in `itertools.product`-Reihenfolge (siehe dcop_bruteforce.py)."""

from dataclasses import dataclass
from itertools import product

from dcop_graph import pairwise_cost, unary_cost


@dataclass(frozen=True)
class UtilTable:
    job_index: int
    separator: tuple  # (0, ..., job_index - 1)
    entries: dict  # {tuple[int, ...] (Länge job_index): (best_cost, best_agent)}


@dataclass(frozen=True)
class DPOPResult:
    assignment: dict  # job_index -> agent_id
    total_cost: float
    util_tables: tuple  # UtilTable je Auftrag, aufsteigend nach job_index


def util_table_size(n_agents, job_index):
    return n_agents ** job_index


def _util_phase(instance):
    n = instance.n_jobs
    k_agents = instance.n_agents
    tables_by_job = {}  # job_index -> UtilTable

    for job_index in range(n - 1, -1, -1):
        separator = tuple(range(job_index))
        child_table = tables_by_job.get(job_index + 1)  # None fürs Blatt
        entries = {}

        for sep_vals in product(range(k_agents), repeat=job_index):
            best_cost = None
            best_agent = None
            for agent in range(k_agents):
                cost = unary_cost(instance, job_index, agent)
                for ancestor_index, ancestor_agent in enumerate(sep_vals):
                    cost += pairwise_cost(instance, ancestor_index, job_index, ancestor_agent, agent)
                if child_table is not None:
                    cost += child_table.entries[sep_vals + (agent,)][0]
                if best_cost is None or cost < best_cost:
                    best_cost, best_agent = cost, agent
            entries[sep_vals] = (best_cost, best_agent)

        tables_by_job[job_index] = UtilTable(job_index=job_index, separator=separator, entries=entries)

    return [tables_by_job[j] for j in range(n)]


def _value_phase(util_tables):
    assignment = {}
    for job_index, table in enumerate(util_tables):
        sep_vals = tuple(assignment[i] for i in range(job_index))
        _, best_agent = table.entries[sep_vals]
        assignment[job_index] = best_agent
    return assignment


def solve_dcop(instance):
    util_tables = _util_phase(instance)
    assignment = _value_phase(util_tables)
    total_cost = util_tables[0].entries[()][0]
    return DPOPResult(assignment=assignment, total_cost=total_cost, util_tables=tuple(util_tables))
