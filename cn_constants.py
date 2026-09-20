"""Defaults, Slider-Grenzen und Presets für die Learning-Augmented-DCOP-Demo.
Die Szenario-Konstanten (POSITION_RANGE_MAX ... SPIKE_MULTIPLIER) sind wortgleich aus dcop-demo übernommen -
dasselbe Vehikel -, da `cn_scenario.py` sie importiert; alles Übrige ist neu."""

DEFAULT_N_JOBS = 8
DEFAULT_N_AGENTS = 3
DEFAULT_DURATION_VARIABILITY = 0.3
DEFAULT_TRAVEL_TIME_PER_UNIT = 1.0
DEFAULT_SEED = 6

# Bis 20 Aufträge: dort sieht man, wo der exakte DPOP unbrauchbar wird und die gelernten Nachrichten
# (bei kleinem Sample-Budget) auf Contract-Net-Niveau fallen. CP-SAT liefert dort nur noch unbewiesene Werte.
N_JOBS_MIN, N_JOBS_MAX = 3, 20
N_AGENTS_MIN, N_AGENTS_MAX = 2, 4
DURATION_VARIABILITY_MIN, DURATION_VARIABILITY_MAX = 0.0, 1.0
TRAVEL_TIME_PER_UNIT_MIN, TRAVEL_TIME_PER_UNIT_MAX = 0.2, 2.0

POSITION_RANGE_MAX = 20.0
DURATION_BASE_RANGE = (5, 15)
SPIKE_PROBABILITY_SCALE = 0.4
SPIKE_MULTIPLIER = 4.0

# CP-SAT-Referenzlauf: harte Zeitgrenze (bei n >= 16 typischerweise unbewiesen -> "beste gefundene Lösung").
ORTOOLS_TIME_LIMIT_SECONDS = 5.0

# --- Exakter DPOP ------------------------------------------------------------------------------
# Größte UTIL-Tabelle (k^(n-1) Einträge), die die App noch exakt rechnet (~200 MB bei 2^22 Einträgen):
# k=4 bis n=12, k=3 bis n=14, k=2 bis n=23.
EXACT_ENTRY_LIMIT = 2 ** 22

# --- Modell lernen (Evolutionsstrategie / CEM) -------------------------------------------------
TRAIN_INSTANCES_MIN, TRAIN_INSTANCES_MAX, DEFAULT_TRAIN_INSTANCES = 10, 60, 30
CEM_ITERATIONS_MIN, CEM_ITERATIONS_MAX, DEFAULT_CEM_ITERATIONS = 4, 20, 12
LEARN_SEED_MIN, LEARN_SEED_MAX, DEFAULT_LEARN_SEED = 0, 99, 0
TRAIN_JOBS = 6            # Trainingsinstanzen sind klein (der exakte DPOP ist dort billig), das Modell überträgt
CEM_POPULATION = 16
CEM_ELITE = 4
CEM_SIGMA0 = 0.6
TRAIN_SEED_BASE = 1_000_000       # Trainingsinstanzen: Seeds außerhalb von Demo- und Held-out-Seeds
HELDOUT_SEED_BASE = 100_000       # feste Held-out-Instanzen (unabhängig vom Demo-Seed)
N_HELDOUT = 20
N_HELDOUT_LARGE = 12              # ab n >= 16 weniger Instanzen (CP-SAT und Nachrichten werden teuer)
LARGE_N_THRESHOLD = 16
N_LEARN_LOTTERY_SEEDS = 5

# --- Nachrichten lernen (Regression je Instanz) -------------------------------------------------
SAMPLES_MIN, SAMPLES_MAX, DEFAULT_SAMPLES = 500, 12000, 2000
SAMPLES_CHOICES = (500, 1000, 2000, 4000, 6000, 12000)
RIDGE = 1e-3

# --- Verdict-Kaskade -----------------------------------------------------------------------------
MESSAGES_WORSE_THAN_CNP_PCT = 5.0     # gelernte Nachrichten >= 5 % schlechter als CNP
MESSAGE_GAP_WARNING_PCT = 3.0         # Nachrichten >= 3 % schlechter als der exakte DPOP desselben Modells
MODEL_EFFECT_SUCCESS_PCT = 3.0        # gelerntes Modell mindestens 3 % besser als CNP

# Preset-Werte empirisch kalibriert (Wegwerf-Sweeps, seither gelöscht) - nicht der erste Versuch übernommen.
# Gemessen mit lern_seed=0, 30 Trainingsinstanzen, 12 CEM-Iterationen, 2000 Samples, Feature-Set "quad_lb_wf"
# (% vs. Contract Net; Fahrzeug-DPOP / gelernt-DPOP / Nachrichten / lokale Suche):
#   Nachrichten = exakter DPOP:  n8k3 s6:  +25.5 / -23.6 / -23.6 / -12.9   (Surrogat-Lücke 0.0 %)
#   Modell-Effekt:               n10k3 s24: +18.1 / -10.9 / -10.9 / -10.9
#   Nachrichten verpassen ...:   n10k3 s14: +39.8 / -14.2 / +3.9 / -14.7    (Surrogat-Lücke +3.3 %)
#   Näherung schlägt exakt:      n10k3 s9:  -14.3 / -14.6 / -25.9 / -14.6   (Surrogat-Lücke +4.1 %)
#   Gelernt verliert:            n8k3 s24:  +59.1 / +28.8 / +28.8 / +28.8   (Contract Net nahe am Optimum)
#   Exakter DPOP explodiert:     n16k4 s20: -  / - / -14.1 / -19.2
#   Lokale Suche reicht:         n20k4 s4:  -  / - / -0.4 / -9.7
#   Kapazität zu klein:          n10k4 s9:  Nachrichten greedy +11.6, linear -2.4, voll -37.9 (= exakt)
_BASE = {
    "duration_variability": 0.3, "travel_time_per_unit": 1.0, "train_instances": DEFAULT_TRAIN_INSTANCES,
    "cem_iterations": DEFAULT_CEM_ITERATIONS, "learn_seed": 0, "feature_set": "quad_lb_wf", "samples": 2000,
}
PRESETS = {
    "Nachrichten = exakter DPOP": {**_BASE, "n_jobs": 8, "n_agents": 3, "seed": 6},
    "Modell-Effekt": {**_BASE, "n_jobs": 10, "n_agents": 3, "seed": 24},
    "Nachrichten verpassen den Gewinn": {**_BASE, "n_jobs": 10, "n_agents": 3, "seed": 14},
    "Näherung schlägt exakt": {**_BASE, "n_jobs": 10, "n_agents": 3, "seed": 9},
    "Gelernt verliert": {**_BASE, "n_jobs": 8, "n_agents": 3, "seed": 24},
    "Kapazität zu klein": {**_BASE, "n_jobs": 10, "n_agents": 4, "seed": 9, "feature_set": "greedy"},
    "Exakter DPOP explodiert": {**_BASE, "n_jobs": 16, "n_agents": 4, "seed": 20},
    "Lokale Suche reicht": {**_BASE, "n_jobs": 20, "n_agents": 4, "seed": 4},
}

PRESET_HELP = {
    "Nachrichten = exakter DPOP": "Das gelernte Modell dreht den Fahrzeug-DPOP von deutlich schlechter als Contract Net "
        "auf deutlich besser - und die gelernten Nachrichten treffen den exakten DPOP-Wert dieses Modells.",
    "Modell-Effekt": "Der Gewinn kommt vom gelernten Kostenmodell, nicht vom Löser: exakter DPOP, gelernte Nachrichten "
        "und lokale Suche liegen gleichauf.",
    "Nachrichten verpassen den Gewinn": "Die Nachrichten haben eine kleine Lücke zum exakten Optimum des Modells - "
        "und die reicht, um den ganzen Gewinn des gelernten Modells zu verlieren.",
    "Näherung schlägt exakt": "Die Näherung ist auf dem DCOP-Ziel schlechter als der exakte DPOP, im echten Makespan aber "
        "besser - Surrogat-Zufall, keine Tugend.",
    "Gelernt verliert": "Contract Net ist hier fast optimal; jedes DCOP-Modell verschlechtert das Ergebnis, das gelernte "
        "nur weniger stark.",
    "Kapazität zu klein": "Die Nachricht hat keinerlei Merkmale (rein gierig): schlechter als Contract Net. Die "
        "Kapazitätsleiter zeigt, wie sich das Ergebnis dem exakten DPOP nähert.",
    "Exakter DPOP explodiert": "16 Aufträge, 4 Agenten: die größte UTIL-Tabelle hätte über eine Milliarde Einträge. "
        "Die gelernten Nachrichten bleiben klein und nutzbar - eine lokale Suche ist trotzdem besser.",
    "Lokale Suche reicht": "20 Aufträge: die gelernten Nachrichten fallen (bei kleinem Sample-Budget) auf Contract-Net-"
        "Niveau, eine simple lokale Suche liegt klar darüber - und CP-SAT bleibt vorn.",
}

# Regressions-Bänder für tests/test_ladcop_evaluation.py::test_presets_produce_expected_bands (Prozent vs. Contract Net,
# je Spalte; exakte Spalten fehlen, wo der exakte DPOP nicht machbar ist).
PRESET_EXPECTED_BANDS = {
    "Nachrichten = exakter DPOP": {
        "dpop_vehicle": (21.0, 30.0), "dpop_learned": (-28.0, -19.0), "msg_learned": (-28.0, -19.0),
        "search_learned": (-17.0, -9.0),
    },
    "Modell-Effekt": {
        "dpop_vehicle": (13.0, 23.0), "dpop_learned": (-15.0, -7.0), "msg_learned": (-15.0, -7.0),
        "search_learned": (-15.0, -7.0),
    },
    "Nachrichten verpassen den Gewinn": {
        "dpop_vehicle": (35.0, 45.0), "dpop_learned": (-19.0, -10.0), "msg_learned": (0.0, 8.0),
        "search_learned": (-19.0, -10.0),
    },
    "Näherung schlägt exakt": {
        "dpop_vehicle": (-19.0, -10.0), "dpop_learned": (-19.0, -10.0), "msg_learned": (-30.0, -22.0),
        "search_learned": (-19.0, -10.0),
    },
    "Gelernt verliert": {
        "dpop_vehicle": (54.0, 64.0), "dpop_learned": (24.0, 33.0), "msg_learned": (24.0, 33.0),
        "search_learned": (24.0, 33.0),
    },
    "Kapazität zu klein": {
        "dpop_vehicle": (-31.0, -22.0), "dpop_learned": (-42.0, -33.0), "msg_learned": (8.0, 16.0),
        "search_learned": (-39.0, -30.0),
    },
    "Exakter DPOP explodiert": {"msg_learned": (-18.0, -10.0), "search_learned": (-23.0, -15.0)},
    "Lokale Suche reicht": {"msg_learned": (-4.0, 3.0), "search_learned": (-14.0, -6.0)},
}
# Erwartetes Verdict je Preset (Code aus ladcop_evaluation.verdict)
PRESET_EXPECTED_VERDICTS = {
    "Nachrichten = exakter DPOP": "model_effect", "Modell-Effekt": "model_effect",
    "Nachrichten verpassen den Gewinn": "message_gap", "Näherung schlägt exakt": "message_gap",
    "Gelernt verliert": "messages_worse_than_cnp", "Kapazität zu klein": "messages_worse_than_cnp",
    "Exakter DPOP explodiert": "exact_infeasible", "Lokale Suche reicht": "exact_infeasible",
}
