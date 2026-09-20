"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe
constraint-programming-demo/csp_presets.py). Zusätzlich zu den fünf Szenario-Reglern: Trainingsumfang und Lern-Seed
des Kostenmodells sowie Feature-Set und Sample-Budget der gelernten Nachrichten."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cn_constants as C
from ladcop_messages import FEATURE_SETS


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_jobs_slider": SettingSpec("jobs", int, C.DEFAULT_N_JOBS, C.N_JOBS_MIN, C.N_JOBS_MAX),
    "n_agents_slider": SettingSpec("agents", int, C.DEFAULT_N_AGENTS, C.N_AGENTS_MIN, C.N_AGENTS_MAX),
    "duration_variability_slider": SettingSpec(
        "var", float, C.DEFAULT_DURATION_VARIABILITY, C.DURATION_VARIABILITY_MIN, C.DURATION_VARIABILITY_MAX
    ),
    "travel_time_per_unit_slider": SettingSpec(
        "travel", float, C.DEFAULT_TRAVEL_TIME_PER_UNIT, C.TRAVEL_TIME_PER_UNIT_MIN, C.TRAVEL_TIME_PER_UNIT_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "train_instances_slider": SettingSpec(
        "ntrain", int, C.DEFAULT_TRAIN_INSTANCES, C.TRAIN_INSTANCES_MIN, C.TRAIN_INSTANCES_MAX
    ),
    "cem_iterations_slider": SettingSpec(
        "cem", int, C.DEFAULT_CEM_ITERATIONS, C.CEM_ITERATIONS_MIN, C.CEM_ITERATIONS_MAX
    ),
    "learn_seed_input": SettingSpec("lseed", int, C.DEFAULT_LEARN_SEED, C.LEARN_SEED_MIN, C.LEARN_SEED_MAX),
    "feature_set_select": SettingSpec("fset", str, "quad_lb_wf"),
    "samples_select": SettingSpec("samples", int, C.DEFAULT_SAMPLES, min(C.SAMPLES_CHOICES), max(C.SAMPLES_CHOICES)),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default
    if "force_regen" not in st.session_state:
        st.session_state["force_regen"] = False


def snap_samples(value):
    """Nächstliegende wählbare Sample-Zahl (Permalinks können beliebige Zahlen enthalten)."""
    return min(C.SAMPLES_CHOICES, key=lambda choice: abs(choice - value))


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    # Sonderfälle: Feature-Set muss ein bekanntes sein, die Sample-Zahl eine der wählbaren Stufen
    if st.session_state.get("feature_set_select") not in FEATURE_SETS:
        st.session_state["feature_set_select"] = "quad_lb_wf"
    st.session_state["samples_select"] = snap_samples(st.session_state.get("samples_select", C.DEFAULT_SAMPLES))
    st.session_state["permalink_loaded"] = True


def sync_query_params(
    n_jobs, n_agents, duration_variability, travel_time_per_unit, seed, train_instances, cem_iterations, learn_seed,
    feature_set, samples,
):
    try:
        st.query_params["jobs"] = str(int(n_jobs))
        st.query_params["agents"] = str(int(n_agents))
        st.query_params["var"] = str(duration_variability)
        st.query_params["travel"] = str(travel_time_per_unit)
        st.query_params["seed"] = str(int(seed))
        st.query_params["ntrain"] = str(int(train_instances))
        st.query_params["cem"] = str(int(cem_iterations))
        st.query_params["lseed"] = str(int(learn_seed))
        st.query_params["fset"] = str(feature_set)
        st.query_params["samples"] = str(int(samples))
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_jobs_slider"] = p["n_jobs"]
    st.session_state["n_agents_slider"] = p["n_agents"]
    st.session_state["duration_variability_slider"] = p["duration_variability"]
    st.session_state["travel_time_per_unit_slider"] = p["travel_time_per_unit"]
    st.session_state["seed_input"] = p["seed"]
    st.session_state["train_instances_slider"] = p["train_instances"]
    st.session_state["cem_iterations_slider"] = p["cem_iterations"]
    st.session_state["learn_seed_input"] = p["learn_seed"]
    st.session_state["feature_set_select"] = p["feature_set"]
    st.session_state["samples_select"] = p["samples"]
    st.session_state["force_regen"] = True


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
    st.session_state["force_regen"] = True
