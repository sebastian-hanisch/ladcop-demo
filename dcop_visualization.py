"""Visualisierungen für die DPOP-Demo: ein Gantt-Chart für eine beliebige
Zuteilung (analog zu task-swap-demos `build_negotiation_schedule_figure`),
ein Balkendiagramm je UTIL-Tabelle (gekappt bei zu vielen Einträgen - das
kappt nur die ANZEIGE, nicht die tatsächliche Berechnung) und ein
Wachstumsdiagramm der Tabellengrößen über alle Aufträge (log-Skala - die
primäre "eigene Schwäche"-Grafik dieses Stücks)."""

import plotly.graph_objects as go

from cn_visualization import AGENT_COLORS, lock_axes


def _intervals_from_schedules(instance, schedules):
    intervals = {a: [] for a in range(instance.n_agents)}
    for agent_id in range(instance.n_agents):
        position = instance.agent_start_positions[agent_id]
        free_time = 0.0
        for job_index in schedules.get(agent_id, ()):
            job = instance.jobs[job_index]
            travel = instance.travel_time(position, job.position)
            start = free_time + travel
            end = start + job.duration
            intervals[agent_id].append((job_index, start, end))
            free_time = end
            position = job.position
    return intervals


def build_dcop_schedule_figure(instance, schedules, ortools_makespan=None, highlight_jobs=frozenset()):
    intervals = _intervals_from_schedules(instance, schedules)
    fig = go.Figure()

    for agent_id in range(instance.n_agents):
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        for job_index, start, end in intervals[agent_id]:
            is_highlighted = job_index in highlight_jobs
            fig.add_trace(
                go.Bar(
                    x=[end - start], y=[f"Agent {agent_id + 1}"], base=start, orientation="h",
                    marker=dict(color=color, line=dict(color="black", width=3 if is_highlighted else 0)),
                    showlegend=False, hovertext=f"Auftrag {job_index + 1}", hoverinfo="text",
                    text=f"A{job_index + 1}", textposition="inside",
                )
            )

    if ortools_makespan is not None:
        fig.add_vline(
            x=ortools_makespan, line_dash="dash", line_color="gray",
            annotation_text="Zentrales Optimum", annotation_position="top",
        )

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Zeit (min)",
        yaxis_title=None,
        height=120 + 60 * instance.n_agents,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[f"Agent {a + 1}" for a in range(instance.n_agents)],
        autorange="reversed",
    )
    return lock_axes(fig)


def build_util_table_size_chart(instance, util_table_sizes):
    job_labels = [f"Auftrag {j + 1}" for j in range(instance.n_jobs)]
    fig = go.Figure(
        go.Bar(x=job_labels, y=list(util_table_sizes), marker_color="#0072B2",
               text=[f"{s:,}" for s in util_table_sizes], textposition="outside")
    )
    fig.update_layout(
        yaxis_title="Tabellengröße (Einträge)", yaxis_type="log",
        height=280, margin=dict(l=10, r=10, t=20, b=10),
    )
    return lock_axes(fig)


def build_util_step_bar(table, max_entries_shown=16):
    """Gibt (figure, n_truncated) zurück - n_truncated>0 heißt, dass nur die
    ersten max_entries_shown von table.entries angezeigt werden (die
    Berechnung selbst bleibt unberührt, nur die Anzeige wird begrenzt)."""
    items = list(table.entries.items())
    truncated = items[:max_entries_shown]
    n_truncated = len(items) - len(truncated)

    labels = [str(sep) if sep else "()" for sep, _ in truncated]
    costs = [cost for _, (cost, _agent) in truncated]
    best_agents = [agent for _, (_cost, agent) in truncated]

    fig = go.Figure(
        go.Bar(
            x=labels, y=costs, marker_color="#0072B2",
            text=[f"Agent {a + 1}" for a in best_agents], textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title=f"Separator-Werte {table.separator}" if table.separator else "(kein Separator - Wurzel)",
        yaxis_title="Beste Kosten",
        height=280, margin=dict(l=10, r=10, t=20, b=10),
    )
    return lock_axes(fig), n_truncated


def describe_value_step(job_index, agent_id):
    return f"Auftrag {job_index + 1} → Agent {agent_id + 1}"
