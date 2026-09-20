"""Exakter DPOP in numpy - dasselbe Verfahren wie `dcop_dpop.solve_dcop` (Kette über den vollständigen Constraint-
Graphen, UTIL bottom-up, VALUE top-down, Tie-Break: niedrigste Agenten-ID), aber als vektorisierte Tensor-Operationen
statt reiner Python-Schleifen und mit beliebigen Kosten-Arrays. Neu und gegen `dcop_dpop` kreuzgeprüft - `dcop_dpop.py`
bleibt die unveränderte Referenz.

UTIL-Tabelle von Auftrag j (Nachricht an j-1): Tensor der Form (k,)*j, indiziert über die Werte der Aufträge 0..j-1,
also k**j Einträge - die exponentielle Schwäche von DPOP."""

import numpy as np

import cn_constants as C


def entries(n_jobs, n_agents):
    """Größte UTIL-Tabelle (die von Auftrag n-1): k^(n-1) Einträge."""
    return n_agents ** (n_jobs - 1)


def exact_feasible(n_jobs, n_agents, limit=C.EXACT_ENTRY_LIMIT):
    return entries(n_jobs, n_agents) <= limit


def _join_table(unary, pair, j, util_next):
    """Tabelle von Auftrag j über (x_0..x_{j-1}, x_j): eigene Kosten + Paarkosten zu den Vorfahren + Kind-Nachricht."""
    k = unary.shape[1]
    table = np.broadcast_to(unary[j].reshape((1,) * j + (k,)), (k,) * j + (k,)).copy()
    agents = np.arange(k)
    own = agents.reshape((1,) * j + (k,))
    for i in range(j):
        ancestor = agents.reshape((1,) * i + (k,) + (1,) * (j - i))
        table += pair[i, j] * (ancestor == own)
    if util_next is not None:
        table += util_next
    return table


def util_tables_np(unary, pair):
    """Alle UTIL-Nachrichten [T_0, ..., T_{n-1}]; T_j hat die Form (k,)*j (T_0 ist ein Skalar = das Optimum). Nur für
    kleine n gedacht (speichert alle Tabellen)."""
    n, k = unary.shape
    tables = [None] * n
    util = None
    for j in range(n - 1, -1, -1):
        util = _join_table(unary, pair, j, util).min(-1)
        tables[j] = util
    return tables


def solve_dpop_np(unary, pair):
    """UTIL bottom-up, VALUE top-down. Gibt (Zuteilung, Kosten, größte Tabelle) zurück."""
    n, k = unary.shape
    best_choice = [None] * n
    util = None
    for j in range(n - 1, -1, -1):
        table = _join_table(unary, pair, j, util)
        best_choice[j] = table.argmin(-1).astype(np.int8)
        util = table.min(-1)
    assignment = []
    for j in range(n):
        assignment.append(int(best_choice[j][tuple(assignment)]))
    return assignment, float(util), entries(n, k)
