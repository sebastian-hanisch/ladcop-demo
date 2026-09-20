"""
Learning-Augmented DCOP an der Kran-Auftragsvergabe – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der "Konzepte"-Reihe, Multi-Agenten-Koordinations-Linie - der Konvergenzknoten aus dcop-demo (Modell:
DCOP, Löser: DPOP) und den lernenden Stücken (marl-demo, mappo-demo). Zwei getrennte Stellen, an denen Lernen in ein
DCOP eintritt: das KOSTENMODELL (Gewichte per Evolutionsstrategie aus echtem Makespan-Feedback) und die UTIL-NACHRICHTEN
(größenbeschränkte, per Regression gefittete Näherungen statt exponentieller Tabellen). Frische Forschungsrichtung,
kein etablierter Standard - dies ist eine kleine ehrliche Rekonstruktion, keine Reproduktion einer veröffentlichten Methode.
"""

import time

import numpy as np
import streamlit as st

import cn_constants as C
from cn_evaluation import stats_up_to_step
from cn_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cn_protocol import run_protocol
from cn_scenario import generate_instance
from cn_visualization import build_bid_chart, build_schedule_figure
from dcop_visualization import build_dcop_schedule_figure
from ladcop_dpop_np import entries, exact_feasible, util_tables_np
from ladcop_evaluation import (
    COLUMN_LABELS,
    capacity_ladder,
    compare_instance,
    heldout_sweep,
    learn_lottery,
    scaling_table,
    schedules_from_assignment,
    verdict,
)
from ladcop_learn_model import train_weights
from ladcop_messages import FEATURE_SETS, all_prefixes, fit_messages, message_value, n_features
from ladcop_model import HAND_BALANCE_WEIGHTS, VEHICLE_WEIGHTS, cost_arrays, instance_arrays
from ladcop_visualization import (
    build_capacity_ladder_chart,
    build_cem_curve,
    build_comparison_bars,
    build_heldout_chart,
    build_learn_lottery_chart,
    build_scaling_chart,
    build_util_scatter,
    build_weights_chart,
)

st.set_page_config(page_title="Learning-Augmented DCOP – Sebastian Hanisch", layout="wide")

SCATTER_ENTRY_LIMIT = 2 ** 18   # exakte Tabellen nur bis hierher für den Approx-vs-exakt-Vergleich berechnen


def _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    return generate_instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)


@st.cache_data(show_spinner=False)
def _compute_cnp(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed):
    instance = _instance(n_jobs, n_agents, duration_variability, travel_time_per_unit, seed)
    return instance, run_protocol(instance)


@st.cache_data(show_spinner=False)
def _compute_model(n_agents, duration_variability, travel_time_per_unit, train_instances, iterations, learn_seed):
    """Das gelernte Kostenmodell - hängt nur von Agentenzahl, Streuung, Anfahrt und den Lernreglern ab, NICHT vom
    Demo-Seed oder der Auftragszahl (Trainingsinstanzen haben feste Größe und eigene Seeds)."""
    return train_weights(n_agents, duration_variability, travel_time_per_unit, train_instances, iterations, learn_seed)


@st.cache_data(show_spinner=False)
def _compute_comparison(scenario_key, model_key, feature_set, samples):
    n_jobs, n_agents, var, travel, seed = scenario_key
    instance = _instance(n_jobs, n_agents, var, travel, seed)
    weights = _compute_model(*model_key).weights
    return compare_instance(instance, weights, feature_set, samples, fit_seed=0)


@st.cache_data(show_spinner=False)
def _compute_util_stages(scenario_key, model_key, feature_set, samples):
    """Gefittete Nachricht vs. exakte UTIL-Tabelle je Stufe (nur wenn die exakten Tabellen klein genug sind)."""
    n_jobs, n_agents, var, travel, seed = scenario_key
    if entries(n_jobs, n_agents) > SCATTER_ENTRY_LIMIT or FEATURE_SETS[feature_set].greedy:
        return None
    arrays = instance_arrays(_instance(n_jobs, n_agents, var, travel, seed))
    unary, pair = cost_arrays(arrays, _compute_model(*model_key).weights)
    tables = util_tables_np(unary, pair)
    messages = fit_messages(unary, pair, arrays.dur, arrays.pos, feature_set, samples, 0)
    rng = np.random.default_rng(3)
    stages = {}
    for j in range(2, n_jobs):
        prefix = all_prefixes(j, n_agents) if n_agents ** j <= 4000 else rng.integers(0, n_agents, (4000, j))
        exact = tables[j][tuple(prefix.T)]
        approx = message_value(messages, j, prefix, unary, pair, arrays.dur, arrays.pos)
        stages[j] = (exact, approx)
    return stages


@st.cache_data(show_spinner=False)
def _compute_sweep(n_jobs, n_agents, var, travel, model_key, feature_set, samples):
    weights = _compute_model(*model_key).weights
    return heldout_sweep(n_jobs, n_agents, var, travel, weights, feature_set, samples)


@st.cache_data(show_spinner=False)
def _compute_ladder(n_jobs, n_agents, var, travel, model_key, samples):
    weights = _compute_model(*model_key).weights
    return capacity_ladder(n_jobs, n_agents, var, travel, weights, samples)


@st.cache_data(show_spinner=False)
def _compute_learn_lottery(n_jobs, n_agents, var, travel, train_instances, iterations, learn_seed):
    return learn_lottery(min(n_jobs, 10), n_agents, var, travel, train_instances, iterations, learn_seed)


def _fmt_int(n):
    """Tausendertrennung deutsch (Punkt)."""
    return f"{n:,}".replace(",", ".")


def _fmt_seconds(seconds):
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds:.2f} s"


def _start_owner(session_key, key):
    st.session_state[session_key] = key


def _delta_metric(column, label, value, reference, reference_label, help_text=None):
    """Delta-Regel des Portfolios: Wert DIESER Karte minus Referenz, niedriger ist besser."""
    delta = value - reference
    if abs(delta) < 1e-6:
        column.metric(label, f"{value:.1f} min", delta="±0.0 min", delta_color="off", help=help_text)
    else:
        column.metric(
            label, f"{value:.1f} min", delta=f"{delta:+.1f} min ggü. {reference_label}",
            delta_color="inverse", help=help_text,
        )


st.title("🧠 Learning-Augmented DCOP an der Kran-Auftragsvergabe")
st.warning(
    "**Frische Forschungsrichtung, kein etablierter Standard.** \"Learning-augmented DCOP\" ist ein aktives, junges "
    "Forschungsfeld (verwandte breite Bereiche: neural approximate dynamic programming, GNN-basierte kombinatorische "
    "Optimierung, gelerntes Message Passing - mittlere Sicherheit, keine konkreten Zitate). Diese App ist eine kleine "
    "ehrliche numpy-Rekonstruktion der Idee, keine Reproduktion einer veröffentlichten Methode."
)
st.markdown(
    """
Zusammenführung von **dcop-demo** (Modell: **DCOP**, Löser: **DPOP**) und den lernenden Stücken. An **zwei getrennten
Stellen** kann in einem DCOP gelernt werden - und die Demo trennt sie sauber: das **Kostenmodell** (was wird minimiert?)
und die **UTIL-Nachrichten** (wie wird es gelöst?). Das zentrale CP-SAT bleibt der praktische Sieger; gelernte und
dezentrale Verfahren tauschen Optimalität gegen Dezentralität und Reichweite.
"""
)
st.caption(
    "Gemessene Kernaussage: das Modell zu lernen bringt viel (aber der Gewinn ist von Hand herleitbar), die Nachrichten zu "
    "lernen bringt Reichweite jenseits der exponentiellen Tabellen - aber keine bessere Qualität, und eine simple lokale "
    "Suche schlägt beides ab etwa 16 Aufträgen."
)

with st.expander("Wie funktioniert diese Demo?", expanded=True):
    st.markdown(
        r"""
**DCOP = Modell, DPOP = Löser.** Wie in dcop-demo: eine Variable pro Auftrag (Wert = Agent), Unärkosten plus Paarkosten
über einen vollständigen Constraint-Graphen. DPOP löst das exakt - über UTIL-Tabellen mit $k^{j}$ Einträgen für Auftrag
$j$ - und hat zwei Schwächen: die **Tabellengröße explodiert**, und das **Modell ist nur ein Surrogat** des echten
Makespans (der Fahrzeug-DPOP verliert deshalb oft gegen Contract Net).

**Lernen an Stelle 1 - das Modell.** Die Kosten werden über wenige Gewichte parametrisiert (Anfahrt, Distanz, Last-Produkt,
Anzahl). Eine **Evolutionsstrategie (CEM)** sucht Gewichte, bei denen die *exakte* DPOP-Lösung einen kleinen **echten
Makespan** hat. Der Löser bleibt unverändert. (Die Dauer-Gewichtung ist wirkungslos: die Dauer eines Auftrags ist für jeden
Agenten gleich.)

**Lernen an Stelle 2 - die Nachrichten.** Statt der exakten UTIL-Tabelle lernt jeder Auftrag eine Funktion aus wenigen
Merkmalen des Präfixes (wie ausgelastet ist jeder Agent, wie stark koppelt der Präfix an die noch offenen Aufträge, eine
untere Schranke aus dem Constraint-Graphen ...). Die **Nachricht sind die Koeffizienten** - ihre Größe hängt nicht von der
Auftragszahl ab. Gefittet wird per Regression (approximative dynamische Programmierung), die Zuteilung entsteht gierig.

**Trennung der Effekte (2x2).** *Modell-Effekt* = Fahrzeug-Modell → gelerntes Modell (beides exakt gelöst); *Löser-Effekt* =
exakt → gelernte Nachrichten (dasselbe Modell). Dazu die stärkste ehrliche dezentrale Baseline: eine **lokale Suche** auf
demselben Ziel, und **CP-SAT** als zentraler Anker.

**Wo steckt hier "MARL"?** Nicht als tiefes Multi-Agenten-RL (das war mappo-demo): die Gewichte werden per
Evolutionsstrategie aus Rückmeldung des echten Makespans gelernt, die Nachrichten per Regression. Eine
Message-Passing-Policy per Evolutionsstrategie wurde getestet und brachte keinen Nutzen (siehe unten).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = C.PRESET_HELP
preset_names = list(C.PRESETS.keys())
for row_start in range(0, len(preset_names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, preset_names[row_start:row_start + 4]):
        with col:
            st.button(name, use_container_width=True, on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider(
        "Anzahl Aufträge", *bounds("n_jobs_slider"), key="n_jobs_slider",
        help="Der exakte DPOP wird nur gerechnet, solange die größte UTIL-Tabelle (k^(n-1) Einträge) höchstens 2^22 "
        "Einträge hat: 4 Agenten bis 12, 3 Agenten bis 14, 2 Agenten bis 23 Aufträge.",
    )
    n_agents = st.slider("Anzahl Agenten (Kräne)", *bounds("n_agents_slider"), key="n_agents_slider")
    duration_variability = st.slider(
        "Streuung der Auftragsdauer", *bounds("duration_variability_slider"), key="duration_variability_slider",
    )
    travel_time_per_unit = st.slider(
        "Anfahrtszeit pro Positionseinheit", *bounds("travel_time_per_unit_slider"),
        key="travel_time_per_unit_slider",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.button(
        "🎲 Neue Instanz generieren",
        use_container_width=True,
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für Auftragspositionen und -dauern (trainiert nichts neu).",
    )

    st.markdown("**Modell lernen**")
    train_instances = st.slider(
        "Trainingsinstanzen", *bounds("train_instances_slider"), key="train_instances_slider",
        help=f"Kleine Instanzen ({C.TRAIN_JOBS} Aufträge), auf denen der exakte DPOP billig ist.",
    )
    cem_iterations = st.slider("CEM-Iterationen", *bounds("cem_iterations_slider"), key="cem_iterations_slider")
    learn_seed = st.number_input(
        "Lern-Seed", *bounds("learn_seed_input"), key="learn_seed_input", step=1,
        help="Zufall des Lernens (Trainingsinstanzen und Evolutionsstrategie) - unabhängig vom Szenario-Seed.",
    )

    st.markdown("**Nachrichten lernen**")
    feature_set = st.selectbox(
        "Merkmale der Nachricht", options=list(FEATURE_SETS), format_func=lambda k: FEATURE_SETS[k].label,
        key="feature_set_select",
        help="Kapazität der Nachricht: von rein gierig (keine Nachricht) bis quadratisch mit Constraint-Merkmalen. "
        "'Generisch' nutzt nur Auftragsmerkmale, 'Constraint' zusätzlich Größen aus dem Constraint-Graphen.",
    )
    if feature_set == "greedy":
        # Ohne Nachricht gibt es nichts zu fitten: der Regler wäre wirkungslos - ausgeblendet, Wert bleibt erhalten.
        st.session_state["samples_select"] = st.session_state["samples_select"]
        samples = st.session_state["samples_select"]
    else:
        samples = st.select_slider(
            "Samples pro Stufe", options=C.SAMPLES_CHOICES, key="samples_select",
            format_func=lambda x: _fmt_int(x),
            help="Zufällige Präfixe, an denen jede Stufe gefittet wird. Mehr Samples = genauere Nachricht, längere Rechenzeit.",
        )

sync_query_params(
    n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, train_instances, cem_iterations, learn_seed,
    feature_set, samples,
)

scenario_key = (int(n_jobs), int(n_agents), duration_variability, travel_time_per_unit, int(seed))
model_key = (
    int(n_agents), duration_variability, travel_time_per_unit, int(train_instances), int(cem_iterations), int(learn_seed),
)

with st.spinner("Führe Contract Net Protocol aus..."):
    instance, cnp_result = _compute_cnp(*scenario_key)
with st.spinner("Lerne das Kostenmodell (Evolutionsstrategie)..."):
    model = _compute_model(*model_key)
with st.spinner("Löse alle Verfahren (bei vielen Aufträgen kann CP-SAT bis zum Zeitlimit laufen)..."):
    cmp = _compute_comparison(scenario_key, model_key, feature_set, samples)
cells = cmp["cells"]
ortools_makespan = cells["cpsat"]["makespan"] if cells["cpsat"] is not None else None
feasible = cmp["exact_feasible"]

# --- Phase 1: Ausgangslage ---------------------------------------------------

st.markdown("## 🎯 Phase 1: Ausgangslage - Contract Net und der Fahrzeug-DPOP")

if "cn_step" not in st.session_state or st.session_state.get("cn_step_owner") != scenario_key:
    st.session_state["cn_step"] = instance.n_jobs - 1
    st.session_state["cn_step_owner"] = scenario_key

max_step = instance.n_jobs - 1
step_col, play_col = st.columns([5, 1])
with step_col:
    if max_step == 0:
        step = 0
        st.caption("Nur ein Auftrag - kein Regler nötig.")
    else:
        step = st.slider("Schritt (Auftragsvergabe)", 0, max_step, key="cn_step")
with play_col:
    auto_play_cnp = st.button("▶️ Abspielen", use_container_width=True, key="cnp_play")

chart_col, bid_col = st.columns([3, 2])
schedule_slot = chart_col.empty()
bid_slot = bid_col.empty()


def _render_cnp(current_step):
    schedule_slot.plotly_chart(
        build_schedule_figure(instance, cnp_result, current_step, ortools_makespan),
        use_container_width=True, key=f"cnp_schedule_{current_step}",
    )
    bid_slot.plotly_chart(
        build_bid_chart(cnp_result.steps[current_step]),
        use_container_width=True, key=f"cnp_bids_{current_step}",
    )


if auto_play_cnp:
    for s in range(0, max_step + 1):
        _render_cnp(s)
        time.sleep(0.4)
    step = max_step
else:
    _render_cnp(step)

live = stats_up_to_step(cnp_result, step)
lm1, lm2 = st.columns(2)
lm1.metric("Aufträge bisher vergeben", f"{live['jobs_awarded']} / {instance.n_jobs}")
lm2.metric("Aktuell schlechteste freie Zeit", f"{live['worst_agent_free_time']:.1f} min")

if feasible:
    v_cell = cells["dpop_vehicle"]
    st.caption(
        f"**Fahrzeug-DPOP** (das Modell aus dcop-demo, exakt gelöst): Makespan **{v_cell['makespan']:.1f} min** "
        f"({cmp['vs_cnp']['dpop_vehicle']:+.1f} % gegenüber Contract Net). Sein Modell kennt keinen Lastausgleich - "
        "deshalb verliert er häufig gegen Contract Net."
    )
else:
    st.caption(
        f"**Fahrzeug-DPOP**: nicht berechenbar - die größte UTIL-Tabelle hätte {_fmt_int(cmp['entries'])} Einträge "
        f"(Grenze {_fmt_int(C.EXACT_ENTRY_LIMIT)})."
    )

st.markdown("---")

# --- Phase 2: Modell lernen --------------------------------------------------

st.markdown("## 📈 Phase 2: Das Kostenmodell lernen")
st.caption(
    "Fitness = mittlerer echter Makespan der *exakten* DPOP-Lösung relativ zu Contract Net auf kleinen Trainingsinstanzen "
    f"({C.TRAIN_JOBS} Aufträge, eigene Seeds). Der Löser bleibt der unveränderte exakte DPOP - gelernt werden nur die Gewichte."
)
m1, m2, m3 = st.columns(3)
m1.metric("Fahrzeug-Modell (Training)", f"{(model.start_fitness - 1) * 100:+.1f} %", help="Mittlerer Makespan vs. Contract Net auf den Trainingsinstanzen.")
final_fit = model.history[-1] if model.history else model.start_fitness
m2.metric("Gelerntes Modell (Training)", f"{(final_fit - 1) * 100:+.1f} %")
m3.metric("Lernzeit", _fmt_seconds(model.seconds), help="Auf diesem Rechner; das Ergebnis ist gecacht.")
left, right = st.columns([2, 3])
with left:
    st.plotly_chart(build_cem_curve(model.history, model.start_fitness), use_container_width=True, key="cem_curve")
with right:
    st.plotly_chart(
        build_weights_chart(VEHICLE_WEIGHTS, HAND_BALANCE_WEIGHTS, model.weights), use_container_width=True,
        key="weights_chart",
    )
st.caption(
    "**Ehrlich eingeordnet:** ein von Hand abgeleiteter Term (die Summe der quadrierten Lasten, Gewicht 'Last-Produkt' = 2) "
    "leistet im Prototyp dasselbe wie das gelernte Modell - der Lerner entdeckt eine bekannte Korrektur in etwa einer Sekunde, "
    "er schlägt sie nicht. Das Fahrzeug-Modell kennt keinen Lastausgleich."
)

lottery_key = (scenario_key[:4], model_key)
lottery_on = st.session_state.get("learn_lottery_owner") == lottery_key
if not lottery_on:
    est = C.N_LEARN_LOTTERY_SEEDS * (model.seconds + 0.5)
    st.button(
        f"🎰 Lern-Seed-Lotterie starten ({C.N_LEARN_LOTTERY_SEEDS} Lernläufe, ca. {est:.0f} s)",
        on_click=_start_owner, args=("learn_lottery_owner", lottery_key), key="learn_lottery_start",
        help="Trainiert das Modell mit mehreren Lern-Seeds und misst das exakte DPOP-Ergebnis auf festen Held-out-Instanzen.",
    )
else:
    with st.spinner("Trainiere mehrere Lernläufe..."):
        lot = _compute_learn_lottery(
            scenario_key[0], scenario_key[1], duration_variability, travel_time_per_unit, train_instances,
            cem_iterations, learn_seed,
        )
    st.plotly_chart(build_learn_lottery_chart(lot), use_container_width=True, key="learn_lottery_chart")
    st.caption(
        f"Held-out (bis 10 Aufträge, exakter DPOP): {lot['min_pct']:+.1f} % bis {lot['max_pct']:+.1f} % gegenüber Contract Net, "
        f"Streuung über Lern-Seeds **{lot['std_pct']:.1f} Punkte**."
    )

st.markdown("---")

# --- Phase 3: Nachrichten lernen ---------------------------------------------

st.markdown("## 📨 Phase 3: Die UTIL-Nachrichten lernen")
st.caption(
    "Exakt: Auftrag j schickt eine Tabelle mit k^j Einträgen. Gelernt: F Koeffizienten - unabhängig von der Auftragszahl. "
    "Das Ziel ist auch hier das DCOP-Ziel des gelernten Modells, nicht der echte Makespan."
)
F = cmp["message_size"]
p1, p2, p3 = st.columns(3)
p1.metric("Nachrichtengröße F", f"{F}" if F else "0 (keine Nachricht)", help="Koeffizienten je Stufe; unabhängig von n und j.")
p2.metric("Größte exakte Tabelle", _fmt_int(cmp["entries"]), help="k^(n-1) Einträge (die von Auftrag n).")
msg_cell = cells["msg_learned"]
p3.metric("Fit-Zeit der Nachrichten", _fmt_seconds(msg_cell.get("fit_seconds", 0.0)))

scale_ns = list(range(4, C.N_JOBS_MAX + 1))
st.plotly_chart(
    build_scaling_chart(scaling_table(int(n_agents), scale_ns, feature_set), FEATURE_SETS[feature_set].label),
    use_container_width=True, key="scaling_chart",
)
st.caption(
    "**Verteilte Semantik:** das Constraint-Merkmal (untere Schranke) braucht die Kopplung zwischen Vorfahr und Nachfahr - ein "
    "echt verteiltes Verfahren müsste dafür eine O(n²)-Interaktionsmatrix die Kette hochreichen. Die Nachrichtengröße ist "
    "daher polynomiell, nicht 'F Zahlen'."
)

stages = _compute_util_stages(scenario_key, model_key, feature_set, samples)
st.markdown("**Gefittete Nachricht vs. exakte UTIL-Tabelle**")
if stages is None:
    reason = (
        "Mit dem Merkmal-Set 'rein gierig' gibt es keine Nachricht." if FEATURE_SETS[feature_set].greedy
        else f"Die exakten Tabellen wären zu groß für den Vergleich (> {_fmt_int(SCATTER_ENTRY_LIMIT)} Einträge)."
    )
    st.caption(reason)
else:
    stage_keys = sorted(stages)
    stage_owner = (scenario_key, model_key, feature_set, samples)
    if "util_stage" not in st.session_state or st.session_state.get("util_stage_owner") != stage_owner:
        st.session_state["util_stage"] = stage_keys[-1]
        st.session_state["util_stage_owner"] = stage_owner
    if len(stage_keys) == 1:
        stage = stage_keys[0]
        st.caption("Nur eine Stufe - kein Regler nötig.")
    else:
        stage = st.slider("Stufe (Auftrag j)", stage_keys[0], stage_keys[-1], key="util_stage")
    exact_values, approx_values = stages[stage]
    rmse = float(np.sqrt(np.mean((approx_values - exact_values) ** 2)))
    spread = float(exact_values.std())
    sc1, sc2 = st.columns([3, 2])
    sc1.plotly_chart(build_util_scatter(exact_values, approx_values, stage), use_container_width=True, key=f"util_scatter_{stage}")
    sc2.metric("Normierter Fehler (RMSE / Std)", f"{rmse / spread:.2f}" if spread > 1e-12 else "0.00")
    sc2.caption(
        f"Stufe {stage}: {_fmt_int(len(exact_values))} Einträge verglichen (ab {_fmt_int(4000)} Präfixen zufällig gezogen). "
        "Nahe 0 heißt: die Nachricht trifft die exakte Tabelle; die Diagonale ist perfekt."
    )

ladder_key = (scenario_key, model_key, samples)
if st.session_state.get("ladder_owner") != ladder_key:
    st.button(
        "🪜 Kapazitätsleiter berechnen (Merkmale schrittweise erweitern, ca. 30 s)",
        on_click=_start_owner, args=("ladder_owner", ladder_key), key="ladder_start",
        help="Rechnet auf 10 festen Held-out-Instanzen die Nachrichten mit wachsender Kapazität.",
    )
else:
    with st.spinner("Rechne die Kapazitätsleiter..."):
        rows = _compute_ladder(scenario_key[0], scenario_key[1], duration_variability, travel_time_per_unit, model_key, samples)
    st.plotly_chart(build_capacity_ladder_chart(rows), use_container_width=True, key="ladder_chart")
    gaps = [r["surrogate_gap_pct"] for r in rows if r["surrogate_gap_pct"] is not None]
    st.caption(
        "Mit wachsender Kapazität nähert sich das Ergebnis dem exakten DPOP des Modells."
        + (f" Surrogat-Lücke zum exakten Optimum: von {gaps[0]:.1f} % (gierig) auf {gaps[-1]:.1f} % (voll)." if gaps else
           " (Die Surrogat-Lücke ist hier nicht berechenbar - der exakte DPOP wäre zu groß.)")
    )

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------

st.subheader("📐 Modell-Effekt, Löser-Effekt und der zentrale Anker")

cnp_ms = cmp["cnp_makespan"]
if cells["cpsat"] is not None:
    cp_note = "beweist Optimalität" if cells["cpsat"].get("optimal") else "Zeitlimit erreicht: beste gefundene Lösung, nicht bewiesen"
else:
    cp_note = "kein Ergebnis im Zeitlimit"


def _metric_or_missing(column, label, name, help_text=None):
    cell = cells.get(name)
    if cell is None:
        column.metric(
            label, "nicht möglich",
            help=f"Die größte UTIL-Tabelle hätte {_fmt_int(cmp['entries'])} Einträge (Grenze {_fmt_int(C.EXACT_ENTRY_LIMIT)}).",
        )
    else:
        _delta_metric(column, label, cell["makespan"], cnp_ms, "CNP", help_text)


st.markdown("**Frage 1 - Modell-Effekt: Bringt ein gelerntes Kostenmodell etwas? (beide Modelle exakt gelöst)**")
q1 = st.columns(3)
q1[0].metric("Contract Net (roh)", f"{cnp_ms:.1f} min")
_metric_or_missing(q1[1], "DPOP, Fahrzeug-Modell", "dpop_vehicle")
_metric_or_missing(q1[2], "DPOP, gelerntes Modell", "dpop_learned")

st.markdown("**Frage 2 - Löser-Effekt: Was kosten gelernte Nachrichten? (dasselbe gelernte Modell)**")
q2 = st.columns(3)
_metric_or_missing(q2[0], "DPOP, gelerntes Modell (exakt)", "dpop_learned")
_delta_metric(
    q2[1], "Gelernte Nachrichten", cells["msg_learned"]["makespan"], cnp_ms, "CNP",
    help_text=f"Nachrichtengröße F = {F}, Fit {_fmt_seconds(cells['msg_learned'].get('fit_seconds', 0.0))}.",
)
_delta_metric(
    q2[2], "Lokale Suche (10 Neustarts)", cells["search_learned"]["makespan"], cnp_ms, "CNP",
    help_text=f"Dasselbe DCOP-Ziel, {_fmt_seconds(cells['search_learned']['seconds'])}.",
)
if "learned" in cmp["fidelity_gap_pct"]:
    st.caption(
        f"Auf dem **DCOP-Ziel** des gelernten Modells liegen die Nachrichten {cmp['fidelity_gap_pct']['learned']:.1f} % über dem "
        "exakten Optimum - der reale Makespan kann trotzdem in beide Richtungen abweichen (das Ziel ist nur ein Surrogat)."
    )

st.markdown("**Frage 3 - Der zentrale Anker: CP-SAT**")
q3 = st.columns(3)
if cells["cpsat"] is not None:
    _delta_metric(q3[0], "CP-SAT (zentral)", cells["cpsat"]["makespan"], cnp_ms, "CNP", help_text=cp_note)
    decentral = [n for n in ("dpop_learned", "msg_learned", "search_learned") if cells.get(n) is not None]
    best_name = min(decentral, key=lambda n: cells[n]["makespan"])
    _delta_metric(
        q3[1], f"Bester dezentraler: {COLUMN_LABELS[best_name].split(',')[0]}", cells[best_name]["makespan"],
        cells["cpsat"]["makespan"], "CP-SAT",
    )
    q3[2].metric("CP-SAT-Zeit", _fmt_seconds(cells["cpsat"]["seconds"]), help=cp_note)
else:
    q3[0].metric("CP-SAT (zentral)", "kein Ergebnis im Zeitlimit")

level, code, data = verdict(cmp)
if code == "exact_infeasible":
    extra = ""
    if data["msg_pct"] is not None and data["search_pct"] is not None:
        extra = (
            f" Die gelernten Nachrichten liegen bei {data['msg_pct']:+.1f} % gegenüber Contract Net, die lokale Suche bei "
            f"{data['search_pct']:+.1f} %."
        )
    st.warning(
        f"⚠️ **Der exakte DPOP ist hier nicht mehr machbar**: die größte UTIL-Tabelle hätte {_fmt_int(data['entries'])} Einträge "
        f"(Grenze {_fmt_int(C.EXACT_ENTRY_LIMIT)}), die gelernte Nachricht dagegen nur {data['message_size']} Zahlen." + extra
    )
elif code == "messages_worse_than_cnp":
    st.warning(
        f"⚠️ **Die gelernten Nachrichten sind schlechter als Contract Net**: {data['msg_pct']:+.1f} %. Das gelernte Modell "
        f"allein (exakt gelöst) liegt bei {data['dpop_learned_pct']:+.1f} % - die Näherung der Nachrichten verspielt seinen Vorteil."
    )
elif code == "message_gap":
    st.warning(
        f"⚠️ **Die Nachrichten verfehlen das Optimum des Modells** um {data['gap_pct']:.1f} % auf dem DCOP-Ziel. Real: "
        f"Nachrichten {data['msg_pct']:+.1f} % vs. exakt {data['dpop_learned_pct']:+.1f} % gegenüber Contract Net. "
        + (
            " Real ist die Näherung sogar BESSER als der exakte DPOP - Surrogat-Zufall, keine Tugend: der exakte DPOP löst "
            "das Ziel, nicht den Makespan."
            if data["msg_pct"] < data["dpop_learned_pct"] - 0.5
            else " Mehr Merkmale (Kapazitätsleiter) oder mehr Samples verkleinern die Lücke."
        )
    )
elif code == "search_beats_messages":
    st.warning(
        f"⚠️ **Eine simple lokale Suche schlägt die gelernten Nachrichten**: {data['search_pct']:+.1f} % gegenüber "
        f"{data['msg_pct']:+.1f} % (beide gegenüber Contract Net) - auf demselben DCOP-Ziel."
    )
elif code == "model_effect":
    st.success(
        f"✅ **Modell-Effekt**: der exakte DPOP mit dem gelernten Modell liegt bei {data['dpop_learned_pct']:+.1f} % gegenüber "
        f"Contract Net, das Fahrzeug-Modell bei {data['dpop_vehicle_pct']:+.1f} %. Die gelernten Nachrichten erreichen "
        f"{data['msg_pct']:+.1f} % - kein nennenswerter Löser-Verlust hier."
    )
else:
    st.info(
        f"Kaum Unterschied: gelerntes Modell {data['dpop_learned_pct']:+.1f} %, Nachrichten {data['msg_pct']:+.1f} % gegenüber "
        "Contract Net."
    )
if cells["cpsat"] is not None:
    best_ms = min(cells[n]["makespan"] for n in ("dpop_learned", "msg_learned", "search_learned") if cells.get(n) is not None)
    if best_ms < cells["cpsat"]["makespan"] - 1e-6:
        st.info(
            f"🏁 **CP-SAT hat im Zeitlimit nicht mehr gefunden**: sein Incumbent liegt bei {cells['cpsat']['makespan']:.1f} min "
            f"({cp_note}), das beste dezentrale Verfahren bei {best_ms:.1f} min. Bei dieser Größe wird der zentrale Solver "
            "in 5 Sekunden nicht mehr bewiesen optimal - ein längeres Limit würde ihn wieder nach vorn bringen."
        )
    else:
        st.info(
            f"🏁 **CP-SAT bleibt vorn**: {cells['cpsat']['makespan']:.1f} min ({cp_note}) gegenüber "
            f"{best_ms:.1f} min des besten dezentralen Verfahrens ({(best_ms / cells['cpsat']['makespan'] - 1) * 100:+.1f} %). "
            "Gelernte und dezentrale Verfahren tauschen globale Optimalität gegen Dezentralität und Reichweite."
        )

st.plotly_chart(build_comparison_bars(cmp), use_container_width=True, key="comparison_bars")

available = [n for n in COLUMN_LABELS if cells.get(n) is not None and n != "cpsat"] + (["cpsat"] if cells["cpsat"] is not None else [])
if st.session_state.get("gantt_column") not in available:
    st.session_state["gantt_column"] = "msg_learned"
gantt_choice = st.selectbox(
    "Zeitplan anzeigen", options=available, format_func=lambda n: COLUMN_LABELS[n], key="gantt_column",
)
st.plotly_chart(
    build_dcop_schedule_figure(
        instance, schedules_from_assignment(cells[gantt_choice]["assignment"], instance.n_agents), ortools_makespan,
    ),
    use_container_width=True, key=f"gantt_{gantt_choice}",
)

st.markdown("**Held-out: feste Instanzen statt Einzelfall**")
sweep_key = (scenario_key[:4], model_key, feature_set, samples)
if st.session_state.get("sweep_owner") != sweep_key:
    n_inst = C.N_HELDOUT_LARGE if scenario_key[0] >= C.LARGE_N_THRESHOLD else C.N_HELDOUT
    per_instance = sum(c["seconds"] for c in cells.values() if c is not None) + 0.1
    st.button(
        f"📊 Held-out-Sweep starten ({n_inst} feste Instanzen, ca. {max(5, n_inst * per_instance):.0f} s)",
        on_click=_start_owner, args=("sweep_owner", sweep_key), key="sweep_start",
        help="Feste Instanzen (Seeds unabhängig vom Demo-Seed): mittlerer Makespan, Gewinn- und Verlustanteil je Verfahren.",
    )
else:
    with st.spinner("Rechne den Held-out-Sweep..."):
        sweep = _compute_sweep(
            scenario_key[0], scenario_key[1], duration_variability, travel_time_per_unit, model_key, feature_set, samples,
        )
    st.plotly_chart(build_heldout_chart(sweep), use_container_width=True, key="heldout_chart")
    stats = sweep["stats"]
    notes = []
    if "msg_vs_exact" in sweep:
        mv = sweep["msg_vs_exact"]
        notes.append(
            f"Nachrichten vs. exakter DPOP desselben Modells: besser auf {mv['better_frac'] * 100:.0f} %, schlechter auf "
            f"{mv['worse_frac'] * 100:.0f} % der Instanzen (im Mittel {mv['mean_pct']:+.1f} Punkte)."
            + (
                " 'Besser' ist Surrogat-Rauschen, keine Tugend: der exakte DPOP löst das Ziel, nicht den Makespan."
                if mv["better_frac"] > 0 else ""
            )
        )
    if "msg_learned" in stats and "search_learned" in stats:
        notes.append(
            f"Lokale Suche {stats['search_learned']['mean_pct']:+.1f} % vs. Nachrichten {stats['msg_learned']['mean_pct']:+.1f} % "
            "gegenüber Contract Net."
        )
    if "cpsat" in stats and "gap_to_best_pct" in stats["msg_learned"]:
        notes.append(f"Lücke der Nachrichten zur besten bekannten Lösung: {stats['msg_learned']['gap_to_best_pct']:.1f} %.")
    st.caption(" ".join(notes))

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**DCOP-Ziel.** Variable $x_j \in \{0,\dots,k-1\}$ je Auftrag, Ziel
$\sum_j u(j, x_j) + \sum_{i<j} p(i,j)\,[x_i = x_j]$ mit

$$
u(j,a) = w_t\,\tau\,|s_a - q_j| + w_d\,d_j, \qquad
p(i,j) = w_p\,\tau\,|q_i - q_j| + w_l\,\tfrac{d_i d_j}{10} + w_c .
$$

Fahrzeug: $(w_t, w_d, w_p, w_l, w_c) = (1, 1, 1, 0, 0)$. $w_d$ verschiebt für alle Zuteilungen dieselbe Konstante (wirkungslos).
$w_l$ erzeugt (summiert über die Paare eines Agenten) die quadrierte Last, $w_c$ die Paaranzahl $\sum_a \binom{n_a}{2}$.

**Modell lernen (CEM).** Fitness $f(w) = \mathbb{E}\big[\text{Makespan}(\text{DPOP}_w)/\text{Makespan}(\text{CNP})\big]$ über
Trainingsinstanzen; Population 16, Elite 4, $\sigma_0 = 0.6$. Ein-Schritt-Fall (nur $w_c=1$, sonst 0): das Optimum ist
$\sum_a \binom{n_a}{2}$ bei ausgeglichener Aufteilung.

**UTIL exakt.** $T_j(x_{<j}) = \min_{x_j}\big[u(j,x_j) + \sum_{i<j} p(i,j)[x_i=x_j] + T_{j+1}(x_{\le j})\big]$, Tabelle mit
$k^{j}$ Einträgen; das Optimum ist $T_0$.

**UTIL gelernt.** $V_j(x_{<j}) = \varphi_j(x_{<j})^\top \beta_j \approx T_j(x_{<j})$ mit $F$ Merkmalen (Auslastung je Agent,
Anzahl, Positionssumme, Kopplung an offene Aufträge, untere Schranke $\sum_{l \ge j} \min_a[\ldots]$, Lastausgleichs-
Schranke, optional alle Produkte). Stufenweise rückwärts per Ridge-Regression an zufälligen Präfixen auf das Ziel
$\min_a[\text{lokal}(j,a) + V_{j+1}(x_{<j},a)]$; Zuteilung vorwärts gierig. Kapazität = Tabellengröße reproduziert die exakte
Tabelle; ohne Kopplung ($p \equiv 0$) genügt eine konstante Nachricht (Suffix-Summe der $\min_a u$).

**Obere Schranke der Näherung.** Die Zuteilung der Nachrichten ist zulässig, ihr DCOP-Ziel also nie kleiner als das exakte
Optimum $T_0$.

Implementiert in `ladcop_model.py`, `ladcop_dpop_np.py` (numpy-DPOP, gegen `dcop_dpop.py` kreuzgeprüft), `ladcop_learn_model.py`,
`ladcop_messages.py`, `ladcop_search.py` und `ladcop_evaluation.py`.
        """
    )

with st.expander("🧪 Was nicht funktioniert hat (einmalige Messung, kein App-Abschnitt)"):
    st.markdown(
        """
Getestet wurde außerdem eine **Message-Passing-Policy** (Knoten = Aufträge, Kanten = die Paarkosten des Constraint-Graphen,
zwei Runden, Evolutionsstrategie, numpy-only): mit Constraint-Graph, ohne Graph (Pooling) und rein lokal - **innerhalb des
Seed-Rauschens gleich**. Die Graph-Struktur brachte hier keinen messbaren Nutzen, die Varianten wurden von Modell- und
Nachrichten-Lernen dominiert (Held-out gegenüber Contract Net, Mittel über 5 Seeds, gemischte Einstellungen):
        """
    )
    st.table({
        "Einstellung": ["n=8, k=3", "n=10, k=3", "n=12, k=3"],
        "Graph": ["−11.3 ± 0.4 %", "−6.5 ± 1.1 %", "−7.0 ± 1.8 %"],
        "Pooling": ["−10.8 ± 0.8 %", "−8.8 ± 0.9 %", "−8.5 ± 0.9 %"],
        "Lokal (ohne Kopplung)": ["−10.6 ± 1.2 %", "−7.1 ± 0.9 %", "−8.6 ± 0.4 %"],
    })
    st.caption(
        "Einmalige Messung; Parameterzahlen der Varianten sind nicht exakt gleich (780 / 588 / 288). 17-27 % der Held-out-"
        "Instanzen verlieren weiterhin gegen Contract Net."
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
