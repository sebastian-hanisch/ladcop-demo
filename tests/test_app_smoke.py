"""Rauchtests: die App läuft für Default und jedes Preset ohne Exception durch (fängt u.a. StreamlitDuplicateElementId
bei st.plotly_chart, Slider-Grenzfälle und tote Regler)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cn_constants as C
from cn_presets import SETTING_SPECS, snap_samples

APP_TIMEOUT = 240
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
PRESET_BUTTONS = {name: i for i, name in enumerate(C.PRESETS)}


def _run():
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT)
    at.run()
    return at


def test_default_run_without_exception():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_runs_without_exception(name):
    at = _run()
    at.button[PRESET_BUTTONS[name]].click().run()
    assert not at.exception, f"{name}: {[e.value for e in at.exception]}"
    preset = C.PRESETS[name]
    assert at.session_state["n_jobs_slider"] == preset["n_jobs"]
    assert at.session_state["feature_set_select"] == preset["feature_set"]


def test_greedy_feature_set_hides_the_samples_control_but_keeps_its_value():
    at = _run()
    at.select_slider(key="samples_select").set_value(4000).run()
    at.selectbox(key="feature_set_select").set_value("greedy").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["samples_select"] == 4000
    assert not [w for w in at.select_slider if w.key == "samples_select"]      # kein toter Regler unter "greedy"
    at.selectbox(key="feature_set_select").set_value("quad_lb").run()
    assert not at.exception and at.session_state["samples_select"] == 4000
    assert [w for w in at.select_slider if w.key == "samples_select"]


def test_small_and_large_instances_run():
    at = _run()
    at.slider(key="n_jobs_slider").set_value(C.N_JOBS_MIN)
    at.slider(key="n_agents_slider").set_value(C.N_AGENTS_MIN).run()
    assert not at.exception, [e.value for e in at.exception]
    at.slider(key="n_jobs_slider").set_value(15)
    at.slider(key="n_agents_slider").set_value(4).run()     # exakt unmöglich: 4^14 Einträge
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_specs_are_unique_and_samples_snap():
    params = [spec.url_param for spec in SETTING_SPECS.values()]
    assert len(params) == len(set(params))
    assert snap_samples(2100) == 2000 and snap_samples(10 ** 6) == max(C.SAMPLES_CHOICES)
