# Learning-Augmented DCOP an der Kran-Auftragsvergabe – Streamlit-Demo

Sechstes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **Multi-Agenten-Koordinations-Linie** - der **Konvergenzknoten** aus
[dcop-demo](../dcop-demo) (Modell: DCOP, Löser: DPOP) und den lernenden Stücken
[marl-demo](../marl-demo) / [mappo-demo](../mappo-demo). Wurzel der Linie ist [contract-net-demo](../contract-net-demo).

> **Frische Forschungsrichtung, kein etablierter Standard.** "Learning-augmented DCOP" ist ein aktives, junges
> Forschungsfeld (verwandte breite Bereiche - mittlere Sicherheit, keine konkreten Zitate: neural approximate dynamic
> programming, GNN-basierte kombinatorische Optimierung, gelerntes Message Passing). Diese Demo ist eine kleine ehrliche
> numpy-Rekonstruktion der Idee, **keine Reproduktion einer veröffentlichten Methode**.

## Was dieses Stück tut

dcop-demo zeigte zwei Schwächen von DPOP: die **UTIL-Tabellen wachsen exponentiell** (k^j Einträge für Auftrag j) und das
**Modell ist nur ein Surrogat** des echten Makespans (der Fahrzeug-DPOP verliert deshalb oft gegen Contract Net). Hier tritt
Lernen an **zwei getrennten Stellen** in das DCOP ein - und ein **2x2-Design** trennt die Effekte:

|  | exakt gelöst (DPOP) | gelernte Nachrichten |
|---|---|---|
| **Fahrzeug-Modell** | +8.6 … +13.5 % | +8.6 … +15.2 % |
| **gelerntes Modell** | −10.8 … −13.6 % | −7.6 … −12.5 % |

(Mittlerer echter Makespan gegenüber Contract Net, 24 Held-out-Instanzen, n=8…12, k=3…4 - Prototyp-Messung.)
**Modell-Effekt** ≈ −22…−26 Punkte, **Löser-Effekt** ≈ +1…+4 Punkte.

1. **Das Modell lernen (Evolutionsstrategie/CEM).** Fünf Gewichte (Anfahrt, Dauer, Distanz, Last-Produkt, Paar-Anzahl)
   werden aus dem Makespan-Feedback der *exakten* DPOP-Lösung gelernt (≈ 1 s). Ergebnis ≈ −12…−14 % gegenüber Contract Net.
   **Ehrlich eingeordnet:** ein von Hand abgeleiteter Term (Summe der quadrierten Lasten) leistet dasselbe - der Lerner
   entdeckt eine bekannte Korrektur, er schlägt sie nicht. Die Dauer-Gewichtung ist wirkungslos (die Dauer eines Auftrags ist
   für jeden Agenten gleich; das gilt auch für dcop-demos Unärkosten).
2. **Die Nachrichten lernen (Regression).** Statt der exakten Tabelle (k^j Einträge) eine feste Zahl F von Koeffizienten
   (153 bei k=3, 231 bei k=4, unabhängig von n): approximative dynamische Programmierung, rückwärts gefittet, vorwärts gierig
   zugeteilt. Kosten ≈ +1…+4 Punkte gegenüber dem exakten DPOP desselben Modells; **Reichweite** dort, wo der exakte DPOP
   nicht mehr rechnet (k=4 ab n=13, App-Grenze 2^22 Einträge: k=4 bis n=12, k=3 bis n=14, k=2 bis n=23).

**Ehrliche Grenzen:**
- Eine **lokale Suche** (10 Neustarts, 1–2 ms) auf demselben DCOP-Ziel ist ab n≥16 gleich gut oder besser als die gelernten
  Nachrichten; bei n=20, k=4 fallen die Nachrichten mit 2000 Samples auf Contract-Net-Niveau (mit 6000+ auf −6.5 %).
- **CP-SAT bleibt 12–20 % vor allem Dezentralen.** Zentral ist der praktische Industriestandard.
- Die Nachrichten schlagen den *exakten* DPOP desselben Modells auf 17–18 % der Instanzen (worse auf 23–53 %) - das ist
  Surrogat-Rauschen (der exakte DPOP löst das Ziel, nicht den Makespan), keine Tugend.
- **Verteilte Semantik:** das Constraint-Merkmal (untere Schranke) braucht die Kopplung zwischen Vorfahr und Nachfahr - ein
  echt verteiltes Verfahren müsste eine O(n²)-Interaktionsmatrix die Kette hochreichen. Die Nachrichtengröße ist polynomiell,
  nicht "F Zahlen".
- Nur das constraint-abgeleitete Lower-Bound-Merkmal verbessert die Nachrichten-Treue (Faktor 1.4–3 auf dem DCOP-Ziel) - ein
  Fidelity-, kein Makespan-Gewinn (die realen Unterschiede bleiben bei ±3 Punkten).
- **Korrektur zu dcop-demo:** die dortige Aussage "DPOP schlägt CNP in etwa der Hälfte der Instanzen" ist
  setting-abhängig (Held-out: Gewinne 27–50 %, Verluste 47–70 %, Mittel +5…+19 % schlechter) - siehe die Held-out-Sweeps.

### Was nicht funktioniert hat (einmalige Messung, kein App-Abschnitt)

Eine **Message-Passing-Policy** (Knoten = Aufträge, Kanten = Paarkosten des Constraint-Graphen, zwei Runden,
Evolutionsstrategie, numpy-only; 400 Iterationen, Batch 48, 5 Seeds, gemischte Trainings-Einstellungen) ist mit
Constraint-Graph, ohne Graph (Pooling) und rein lokal **innerhalb des Seed-Rauschens gleich** - kein messbarer Nutzen der
Graph-Struktur, und von Modell-/Nachrichten-Lernen dominiert (Held-out gegenüber Contract Net):

| Einstellung | Graph | Pooling | Lokal |
|---|---|---|---|
| n=8, k=3 | −11.3 ± 0.4 % | −10.8 ± 0.8 % | −10.6 ± 1.2 % |
| n=10, k=3 | −6.5 ± 1.1 % | −8.8 ± 0.9 % | −7.1 ± 0.9 % |
| n=12, k=3 | −7.0 ± 1.8 % | −8.5 ± 0.9 % | −8.6 ± 0.4 % |

Einmalige Messung, die Parameterzahlen der Varianten sind nicht exakt gleich (780 / 588 / 288); auf 17–27 % der
Held-out-Instanzen verliert die Policy weiterhin gegen Contract Net. Nicht getestet: PPO statt Evolutionsstrategie,
autoregressives Decodieren, längeres Training, ein instanzübergreifend amortisiertes Nachrichten-Netz.

## Modell-Familie

```
unär(j, a)       = w_travel · Anfahrt(Start_a → Auftrag j) + w_duration · Dauer_j     (w_duration wirkungslos)
paar(i, j, a=b)  = w_distance · Anfahrt(i → j) + w_load · Dauer_i · Dauer_j / 10 + w_count
paar(i, j, a≠b)  = 0
```
Fahrzeug = (1, 1, 1, 0, 0) - exakt das Modell aus dcop-demo. `w_load` erzeugt (über die Paare eines Agenten) die quadrierte
Last, `w_count` die Paaranzahl Σ_a C(n_a, 2) - beides Lastausgleichs-Terme, die dem Fahrzeug-Modell fehlen.

## Verifikation

- **Numpy-DPOP = `dcop_dpop`** (Zuteilung, Kosten und alle UTIL-Tabellen auf 12 Zufallsinstanzen) und **= Brute-Force**
  (n ≤ 7, k ≤ 3, mehrere Modelle); das 3-Auftrags/2-Agenten-Handbeispiel (Zuteilung [0,1,0], Kosten 17.0, Tabellen von
  dcop-demo). `dcop_dpop.py` bleibt die unveränderte Referenz.
- **Geschlossene Formen**: nur `w_count` ⇒ Optimum Σ C(n_a,2) bei ausgeglichener Aufteilung (n=6,k=3 → 3; n=7 → 5; n=8 → 7);
  ohne Kopplung (S=0) ist die exakte UTIL-Tabelle die Suffix-Summe der min_a U und wird von einer linearen Nachricht exakt
  getroffen; **Kapazität = Tabellengröße** reproduziert die exakten Tabellen (max. Abweichung 0).
- Null-Gewichte (alles auf Agent 0, Tie-Break), 0 CEM-Iterationen = Fahrzeug-Modell, Nachricht ohne Merkmale = gieriger
  lokaler Schritt, wirkungsloses Dauer-Gewicht, Nachrichten-Ziel ≥ exaktes Optimum, Determinismus, Seed-Entkopplung
  (Trainings-, Held-out- und Demo-Seeds liegen in getrennten Bereichen), Vorzeichen-Test des Modell-Effekts.
- **Verdict-Kaskade** (Reihenfolge), **Preset-Bänder und -Verdicts** für alle Presets, **AppTest-Rauchtests** (Default, jedes
  Preset, Exakt-unmöglich-Zweig, Kleinstinstanz, versteckter Sample-Regler behält seinen Wert).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, Phasen 1–3, Vergleich (Modell-Effekt / Löser-Effekt / CP-SAT), Held-out, Kapazitätsleiter |
| `cn_constants.py` | Defaults, Regler-Grenzen, Exakt-Grenze, Verdict-Schwellen, `PRESETS`, Bänder |
| `cn_presets.py` | `SettingSpec`/Permalink-Logik (Trainingsumfang, Lern-Seed, Feature-Set, Samples) |
| `ladcop_model.py` | Parametrisiertes DCOP-Modell, Gewichte, Kosten-Arrays, echter Makespan |
| `ladcop_dpop_np.py` | Exakter DPOP als numpy-Tensoroperationen (kreuzgeprüft) |
| `ladcop_learn_model.py` | Modell lernen (CEM) |
| `ladcop_messages.py` | Gelernte, größenbeschränkte UTIL-Nachrichten, Feature-Sets/Kapazitätsleiter |
| `ladcop_search.py` | Lokale Suche auf demselben DCOP-Ziel |
| `ladcop_evaluation.py` | Vergleich, Held-out-Sweep, Kapazitätsleiter, Lern-Seed-Lotterie, Verdict |
| `ladcop_visualization.py` | Gewichte, CEM-Kurve, Skalierung, Leiter, Approx-vs-exakt, Vergleichsbalken |
| `cn_*.py`, `dcop_*.py` | Vehikel und Referenz-DPOP (wortgleich aus dcop-demo) |
| `tests/` | Modell/DPOP, Lernen, Evaluation, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
