"""Visualisierungen der Learning-Augmented-DCOP-Demo: Gewichte, CEM-Kurve, Nachrichten- vs. Tabellengröße, Kapazitäts-
leiter, Approx-vs-exakt-UTIL, Vergleichsbalken, Held-out-Statistik und Lern-Seed-Lotterie. Der Gantt-Chart kommt
unverändert aus `dcop_visualization.build_dcop_schedule_figure`."""

import plotly.graph_objects as go

import cn_constants as C
from ladcop_evaluation import COLUMN_LABELS
from ladcop_model import WEIGHT_NAMES

VEHICLE_COLOR = "#7F7F7F"
HAND_COLOR = "#E69F00"
LEARNED_COLOR = "#0072B2"
EXACT_COLOR = "#D55E00"
OPT_COLOR = "#009E73"
COLUMN_COLORS = {
    "cnp": "#B0B0B0", "dpop_vehicle": "#7F7F7F", "dpop_learned": "#0072B2", "msg_vehicle": "#CC79A7",
    "msg_learned": "#D55E00", "search_learned": "#56B4E9", "cpsat": "#009E73",
}


def build_weights_chart(vehicle, hand, learned):
    """Die vier wirksamen Gewichte: Fahrzeug-Modell, von Hand abgeleiteter Lastausgleich und gelernt. (Die Dauer-
    Gewichtung ist wirkungslos und wird nicht gezeigt.)"""
    idx = [0, 2, 3, 4]
    labels = [WEIGHT_NAMES[i] for i in idx]
    fig = go.Figure()
    for name, weights, color in (("Fahrzeug-Modell", vehicle, VEHICLE_COLOR), ("Von Hand (quadrierte Last)", hand, HAND_COLOR),
                                 ("Gelernt", learned, LEARNED_COLOR)):
        values = weights.as_vector()[idx]
        fig.add_trace(go.Bar(
            x=labels, y=values, name=name, marker_color=color, text=[f"{v:.2f}" for v in values], textposition="outside",
        ))
    fig.update_layout(barmode="group", yaxis_title="Gewicht", height=320, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def build_cem_curve(history, start_fitness):
    """Bester Trainings-Fitnesswert je Iteration (mittlerer Makespan der exakten DPOP-Lösung vs. Contract Net, in %)."""
    iterations = list(range(0, len(history) + 1))
    values = [(start_fitness - 1.0) * 100.0] + [(h - 1.0) * 100.0 for h in history]
    fig = go.Figure(go.Scatter(x=iterations, y=values, mode="lines+markers", line=dict(color=LEARNED_COLOR, width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color=VEHICLE_COLOR, annotation_text="Contract Net", annotation_position="top left")
    fig.update_xaxes(title="CEM-Iteration (0 = Fahrzeug-Modell)", dtick=1 if len(history) <= 20 else None)
    fig.update_yaxes(title="Trainings-Makespan vs. Contract Net (%)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def build_scaling_chart(rows, message_label):
    """Größte exakte UTIL-Tabelle (k^(n-1) Einträge, exponentiell) gegen die feste Nachrichtengröße F (log-Skala),
    mit der Exakt-Grenze der App."""
    ns = [r["n_jobs"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=[r["entries"] for r in rows], mode="lines+markers", name="Exakte UTIL-Tabelle (Einträge)",
        line=dict(color=EXACT_COLOR, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=ns, y=[max(r["message_size"], 1) for r in rows], mode="lines+markers", name=f"Gelernte Nachricht ({message_label})",
        line=dict(color=LEARNED_COLOR, width=3),
    ))
    fig.add_hline(y=C.EXACT_ENTRY_LIMIT, line_dash="dash", line_color=VEHICLE_COLOR,
                  annotation_text="Exakt-Grenze der App (2²² Einträge)", annotation_position="top left")
    fig.update_yaxes(type="log", title="Zahlen pro Nachricht")
    fig.update_xaxes(title="Anzahl Aufträge", dtick=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return fig


def build_capacity_ladder_chart(rows):
    """Nachrichten-Kapazität (Merkmale) gegen den echten Makespan vs. Contract Net."""
    labels = [f"{r['label']}<br>F = {r['size']}" for r in rows]
    values = [r["vs_cnp_pct"] for r in rows]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=[LEARNED_COLOR if v <= 0 else EXACT_COLOR for v in values],
        text=[f"{v:+.1f} %" for v in values], textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=VEHICLE_COLOR, annotation_text="Contract Net", annotation_position="top left")
    fig.update_yaxes(title="Makespan vs. Contract Net (%)")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def build_util_scatter(exact, approx, stage):
    """Gefittete Nachricht gegen exakte UTIL-Einträge derselben Stufe (Diagonale = perfekt)."""
    low, high = float(min(exact.min(), approx.min())), float(max(exact.max(), approx.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exact, y=approx, mode="markers", marker=dict(color=LEARNED_COLOR, size=5, opacity=0.6),
                             name="Einträge"))
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line=dict(color=VEHICLE_COLOR, dash="dash"),
                             name="perfekt"))
    fig.update_xaxes(title=f"exakter UTIL-Wert (Stufe {stage})")
    fig.update_yaxes(title="gelernte Nachricht")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig


def build_comparison_bars(cmp):
    """Echter Makespan vs. Contract Net je Spalte (nur vorhandene Spalten)."""
    names = [n for n in COLUMN_LABELS if n in cmp["vs_cnp"]]
    values = [cmp["vs_cnp"][n] for n in names]
    fig = go.Figure(go.Bar(
        y=[COLUMN_LABELS[n] for n in names], x=values, orientation="h", marker_color=[COLUMN_COLORS[n] for n in names],
        text=[f"{v:+.1f} %" for v in values], textposition="outside",
    ))
    fig.update_xaxes(title="Makespan vs. Contract Net (%)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=320, margin=dict(l=10, r=40, t=20, b=10), showlegend=False)
    return fig


def build_heldout_chart(sweep):
    """Mittlerer Makespan vs. CNP je Spalte auf den festen Held-out-Instanzen, Beschriftung mit Gewinn-/Verlustanteil."""
    names = [n for n in COLUMN_LABELS if n in sweep["stats"] and n != "cnp"]
    values = [sweep["stats"][n]["mean_pct"] for n in names]
    texts = [
        f"{sweep['stats'][n]['mean_pct']:+.1f} %  (besser {sweep['stats'][n]['win_frac'] * 100:.0f} %, "
        f"schlechter {sweep['stats'][n]['lose_frac'] * 100:.0f} %)" for n in names
    ]
    fig = go.Figure(go.Bar(
        y=[COLUMN_LABELS[n] for n in names], x=values, orientation="h", marker_color=[COLUMN_COLORS[n] for n in names],
        text=texts, textposition="outside",
    ))
    fig.update_xaxes(title=f"mittlerer Makespan vs. Contract Net (%), {sweep['n_instances']} feste Held-out-Instanzen")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=340, margin=dict(l=10, r=200, t=20, b=10), showlegend=False)
    return fig


def build_learn_lottery_chart(lottery):
    """Held-out-Ergebnis des gelernten Modells (exakter DPOP) je Lern-Seed."""
    labels = [f"Lern-Seed {r['seed']}" for r in lottery["rows"]]
    values = [r["vs_cnp_pct"] for r in lottery["rows"]]
    fig = go.Figure(go.Scatter(x=labels, y=values, mode="markers", marker=dict(size=13, color=LEARNED_COLOR)))
    fig.add_hline(y=0, line_dash="dash", line_color=VEHICLE_COLOR, annotation_text="Contract Net", annotation_position="top left")
    fig.update_yaxes(title="Makespan vs. Contract Net (%)")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    return fig
