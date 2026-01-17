# 📊 Training Log System

## Übersicht

Das Training speichert automatisch alle wichtigen Metriken in `training_log.csv` für spätere Analyse.

## 🎯 Was wird geloggt?

### Bei jedem Update:
- `timestamp` - Zeitstempel
- `update` - Update-Nummer
- `avg_return` - Durchschnittliche Belohnung
- `max_stage` - Höchster erreichter Stage
- `device` - Verwendete Hardware (cpu/mps/cuda)

### Bei jeder Evaluation (alle 10 Updates):
- `eval_avg_return` - Evaluation Average Return
- `eval_max_stage` - Evaluation Max Stage
- `eval_coins` - Gesammelte Münzen
- `eval_flag_get` - Flagge erreicht (True/False)
- `eval_life` - Verbleibende Leben
- `eval_score` - Spiel-Score
- `eval_status` - Mario Status (small/tall/fireball)
- `eval_time` - Verbleibende Zeit
- `eval_x_pos` - Horizontale Position (Fortschritt)
- `eval_y_pos` - Vertikale Position

## 📈 Visualisierung

### Installation:
```bash
pip install pandas matplotlib
```

### Plots erstellen:
```bash
python plot_training.py
```

Dies erstellt:
1. **training_progress.png** - 6 Plots mit allen Metriken
2. **Konsolen-Ausgabe** mit Statistiken

### Plots enthalten:
1. Average Return über Zeit
2. Max Stage Fortschritt
3. Evaluation Performance
4. X Position (Level-Fortschritt)
5. Score & Münzen
6. Verbleibende Leben

## 🔄 Training unterbrechen und fortsetzen

Das CSV-System funktioniert nahtlos mit Unterbrechungen:

```bash
# Training starten
python ppo.py

# Mit Strg+C unterbrechen
# (Letzter Checkpoint bei Update 150)

# Später weiter trainieren
python ppo.py
# → CSV wird automatisch weitergeführt
# → Keine Duplikate, nahtlose Fortsetzung
```

## 📁 Dateien

- `training_log.csv` - Alle Training-Daten (wird automatisch erstellt)
- `plot_training.py` - Visualisierungs-Skript
- `training_progress.png` - Generierte Plots

## 💡 Tipps

### CSV-Datei direkt öffnen:
- Excel, Numbers, Google Sheets
- LibreOffice Calc
- Pandas/Python für eigene Analysen

### Eigene Analysen:
```python
import pandas as pd

# Lade Daten
df = pd.read_csv('training_log.csv')

# Beispiel: Bester Return
best_return = df['avg_return'].max()
best_update = df.loc[df['avg_return'].idxmax(), 'update']
print(f"Bester Return: {best_return} bei Update {best_update}")

# Beispiel: Wie oft Flagge erreicht?
evals = df[df['eval_flag_get'].notna()]
flags = evals['eval_flag_get'].sum()
print(f"Flagge erreicht: {flags} mal")
```

## 🎮 Beispiel CSV-Ausgabe:

```csv
timestamp,update,avg_return,max_stage,eval_avg_return,eval_max_stage,...
2026-01-06 20:15:30,1,12.45,1,,,,,,,,,cpu
2026-01-06 20:15:45,2,15.23,1,,,,,,,,,cpu
...
2026-01-06 20:18:30,10,45.67,1,42.30,1,3,False,2,450,small,385,1250,79,cpu
```

## 📊 Während des Trainings

Die CSV-Datei wird **nach jedem Update** geschrieben, d.h.:
- Daten sind auch bei Crash verfügbar
- Du kannst Plots während des Trainings erstellen
- Kein Datenverlust bei Unterbrechung

## ✅ Features

- ✅ Automatische Initialisierung
- ✅ Nahtlose Fortsetzung bei Neustart
- ✅ Zeitstempel für jedes Update
- ✅ Alle Evaluations-Details
- ✅ Hardware-Information (device)
- ✅ Kein Datenverlust
- ✅ Einfache Visualisierung

Viel Erfolg beim Training! 🚀
