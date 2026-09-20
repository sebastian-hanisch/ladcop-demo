"""Gelernte, größenbeschränkte UTIL-Nachrichten. Statt der exakten Tabelle T_j (k^j Einträge, indiziert über die Werte
der Aufträge 0..j-1) lernt jeder Auftrag eine Funktion V_j(Präfix) ≈ T_j aus wenigen Merkmalen des Präfixes - die
"Nachricht" sind die F Regressions-Koeffizienten, unabhängig von n und j.

Gefittet wird stufenweise rückwärts (approximative dynamische Programmierung): V_j soll den Bellman-Wert
min_a [lokal(j, a | Präfix) + V_{j+1}(Präfix + a)] treffen, an zufälligen Präfixen (Ridge-Regression, ein Sweep).
Die Zuteilung entsteht vorwärts gierig: Auftrag j wählt den Agenten, der lokal + V_{j+1} minimiert.

Das ist KEIN veröffentlichtes Verfahren, sondern eine kleine ehrliche Rekonstruktion der Idee "Nachrichten approximieren
statt exakt rechnen". Verteilte Semantik: das constraint-abgeleitete Merkmal (LB) braucht S[i, l] für Vorfahr i und
Nachfahr l - ein echt verteiltes Verfahren müsste eine O(n²)-Interaktionsmatrix die Kette hochreichen; die
Nachrichtengröße ist daher polynomiell, nicht "F Zahlen"."""

import time
from dataclasses import dataclass

import numpy as np

import cn_constants as C


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    greedy: bool = False        # keine Nachricht (V = 0): rein lokal-gierig
    interaction: bool = True    # constraint-abgeleitet: Kopplung der Präfix-Aufträge an die noch offenen Aufträge
    lb: bool = False            # untere Schranke aus dem Constraint-Graph (Σ_l min_a ...)
    wf: bool = False            # Lastausgleichs-Schranke (water filling)
    quad: bool = False          # alle paarweisen Produkte der Merkmale


# Reihenfolge = Kapazitätsleiter (klein -> groß)
FEATURE_SETS = {
    "greedy": FeatureSpec("greedy", "Keine Nachricht (rein gierig)", greedy=True),
    "linear_generic": FeatureSpec("linear_generic", "Linear, generische Auftragsmerkmale", interaction=False),
    "linear": FeatureSpec("linear", "Linear, + Constraint-Kopplung"),
    "linear_lb": FeatureSpec("linear_lb", "Linear + Schranke (LB)", lb=True),
    "linear_lb_wf": FeatureSpec("linear_lb_wf", "Linear + LB + Lastausgleich", lb=True, wf=True),
    "quad_generic": FeatureSpec("quad_generic", "Quadratisch, generische Merkmale", interaction=False, quad=True),
    "quad": FeatureSpec("quad", "Quadratisch, + Constraint-Kopplung", quad=True),
    "quad_lb": FeatureSpec("quad_lb", "Quadratisch + LB", lb=True, quad=True),
    "quad_lb_wf": FeatureSpec("quad_lb_wf", "Quadratisch + LB + Lastausgleich (voll)", lb=True, wf=True, quad=True),
}
DEFAULT_FEATURE_SET = "quad_lb_wf"
LADDER = ("greedy", "linear", "linear_lb", "linear_lb_wf", "quad", "quad_lb", "quad_lb_wf")


def n_features(n_agents, feature_set):
    """Größe der Nachricht F (Koeffizienten je Stufe) - unabhängig von n und j."""
    spec = FEATURE_SETS[feature_set]
    if spec.greedy:
        return 0
    base = (4 if spec.interaction else 3) * n_agents + (2 if spec.lb else 0) + (2 if spec.wf else 0)
    return 1 + base + (base * (base + 1) // 2 if spec.quad else 0)


def _one_hot(prefix, k):
    count, j = prefix.shape
    out = np.zeros((count, j, k))
    out[np.arange(count)[:, None], np.arange(j)[None, :], prefix] = 1.0
    return out


def _features(prefix, j, unary, pair, dur, pos, k, spec):
    """Merkmale eines Präfixes (Aufträge 0..j-1 zugeteilt), Form [N, F]."""
    count = prefix.shape[0]
    n = unary.shape[0]
    if j > 0:
        one_hot = _one_hot(prefix, k)
        coupling = pair[:j, j:].sum(1) if j < n else np.zeros(j)
        agg_coupling = np.einsum("nik,i->nk", one_hot, coupling)
        agg_dur = np.einsum("nik,i->nk", one_hot, dur[:j])
        agg_count = one_hot.sum(1)
        agg_pos = np.einsum("nik,i->nk", one_hot, pos[:j])
    else:
        one_hot = None
        agg_coupling = agg_dur = agg_count = agg_pos = np.zeros((count, k))
    base = ([agg_coupling] if spec.interaction else []) + [agg_dur, agg_count, agg_pos]
    if spec.lb:
        cross = np.einsum("nik,il->nkl", one_hot, pair[:j, j:]) if j > 0 else np.zeros((count, k, n - j))
        lb = (unary[j:].T[None, :, :] + cross).min(1).sum(1)
        base += [lb[:, None], (lb ** 2)[:, None] / 100.0]
    if spec.wf:
        remaining = dur[j:].sum()
        loads = np.sort(agg_dur, 1)
        level = ((remaining + np.cumsum(loads, 1)) / np.arange(1, k + 1)[None, :]).min(1)
        wf = (np.maximum(agg_dur, level[:, None]) ** 2).sum(1)
        base += [wf[:, None] / 100.0, np.sqrt(wf)[:, None]]
    b = np.concatenate(base, 1)
    columns = [np.ones((count, 1)), b]
    if spec.quad:
        upper = np.triu_indices(b.shape[1])
        columns.append((b[:, :, None] * b[:, None, :])[:, upper[0], upper[1]])
    return np.concatenate(columns, 1)


def _local_costs(prefix, j, unary, pair, k):
    """Kosten von Auftrag j bei Agent a gegeben das Präfix: unär + Paarkosten zu den Präfix-Aufträgen desselben Agenten."""
    out = np.tile(unary[j], (prefix.shape[0], 1))
    for a in range(k):
        out[:, a] += ((prefix == a) * pair[:j, j][None, :]).sum(1)
    return out


@dataclass(frozen=True)
class Messages:
    """Gefittete Nachrichten: betas[j] ist der Koeffizientenvektor der Stufe j (j = 0..n-1)."""
    feature_set: str
    betas: tuple
    n_samples: int
    fit_seconds: float

    @property
    def size(self):
        return 0 if not self.betas or self.betas[0] is None else len(self.betas[0])


def fit_messages(unary, pair, dur, pos, feature_set=DEFAULT_FEATURE_SET, n_samples=C.DEFAULT_SAMPLES, seed=0, ridge=C.RIDGE):
    """Fittet V_j für alle Stufen. `greedy` fittet nichts (V = 0)."""
    started = time.perf_counter()
    n, k = unary.shape
    spec = FEATURE_SETS[feature_set]
    if spec.greedy:
        return Messages(feature_set, tuple([None] * (n + 1)), 0, time.perf_counter() - started)
    rng = np.random.default_rng([seed, 41])
    betas = [None] * (n + 1)
    for j in range(n - 1, -1, -1):
        prefix = rng.integers(0, k, (n_samples, j))
        target = _local_costs(prefix, j, unary, pair, k)
        if j < n - 1:
            for a in range(k):
                extended = np.concatenate([prefix, np.full((n_samples, 1), a)], 1)
                target[:, a] += _features(extended, j + 1, unary, pair, dur, pos, k, spec) @ betas[j + 1]
        y = target.min(1)
        feats = _features(prefix, j, unary, pair, dur, pos, k, spec)
        mean, sd = feats.mean(0), feats.std(0) + 1e-9
        sd[0], mean[0] = 1.0, 0.0
        scaled = (feats - mean) / sd
        penalty = ridge * n_samples * np.eye(feats.shape[1])
        penalty[0, 0] = 0.0          # der Bias wird nicht geschrumpft (sonst ist schon eine Konstante nicht exakt)
        w = np.linalg.solve(scaled.T @ scaled + penalty, scaled.T @ y)
        beta = np.concatenate([[0.0], w[1:] / sd[1:]])
        beta[0] = w[0] - (w[1:] * mean[1:] / sd[1:]).sum()
        betas[j] = beta
    return Messages(feature_set, tuple(betas), n_samples, time.perf_counter() - started)


def message_value(messages, j, prefix, unary, pair, dur, pos):
    """V_j(Präfix) für ein Batch von Präfixen [N, j]; greedy => 0."""
    spec = FEATURE_SETS[messages.feature_set]
    if spec.greedy:
        return np.zeros(prefix.shape[0])
    return _features(prefix, j, unary, pair, dur, pos, unary.shape[1], spec) @ messages.betas[j]


def solve_with_messages(messages, unary, pair, dur, pos):
    """Vorwärts gierige Zuteilung: Auftrag j wählt den Agenten mit minimalem lokal + V_{j+1}."""
    n, k = unary.shape
    prefix = np.zeros((1, 0), dtype=int)
    for j in range(n):
        value = _local_costs(prefix, j, unary, pair, k)
        if j < n - 1:
            for a in range(k):
                extended = np.concatenate([prefix, np.full((1, 1), a)], 1)
                value[:, a] += message_value(messages, j + 1, extended, unary, pair, dur, pos)
        prefix = np.concatenate([prefix, value.argmin(1)[:, None]], 1)
    return [int(x) for x in prefix[0]]


def all_prefixes(j, k):
    """Alle k^j Präfixe der Länge j als [k^j, j]-Array (für Tests/Vergleiche bei kleinem n)."""
    if j == 0:
        return np.zeros((1, 0), dtype=int)
    grids = np.meshgrid(*([np.arange(k)] * j), indexing="ij")
    return np.stack([g.ravel() for g in grids], 1)


def message_error(messages, exact_tables, unary, pair, dur, pos):
    """Je Stufe j >= 2: normierter RMSE zwischen gefitteter Nachricht und exakter UTIL-Tabelle über ALLE k^j Präfixe
    (RMSE / Standardabweichung der exakten Einträge). Stufe 0 ist ein Skalar, Stufe 1 hat nur k Einträge - dort ist
    das Verhältnis nicht aussagekräftig."""
    n, k = unary.shape
    out = []
    for j in range(2, n):
        prefix = all_prefixes(j, k)
        exact = exact_tables[j][tuple(prefix.T)]
        approx = message_value(messages, j, prefix, unary, pair, dur, pos)
        spread = exact.std()
        out.append(float(np.sqrt(np.mean((approx - exact) ** 2)) / spread) if spread > 1e-12 else 0.0)
    return out
