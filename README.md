# LLM-Routing Energiemessung

Experimentelles Framework für die Bachelorarbeit _Energieeinsparung durch LLM-Routing_.

Ein kleines Router-Modell klassifiziert eingehende Prompts und wählt dynamisch
das passende Zielmodell sowie den Thinking-Modus. Ein Scheduler verwaltet den
VRAM pro GPU, lädt Modelle bei Bedarf, nutzt vLLM Sleep/Wake und ermöglicht auf
Multi-GPU-Systemen parallele Inferenz. Der Energieverbrauch wird über NVIDIA
NVML als GPU-Leistungszeitreihe gemessen und als dynamische Energie oberhalb
der Idle-Baseline ausgewertet.

## Voraussetzungen

- Python 3.11+
- Linux mit NVIDIA GPU und CUDA-kompatiblem Treiber (getestet mit RTX 3090 und A100). Sichergehen, dass NVIDIA-SMI im Persistenz-Modus läuft und die GPU-Leistungsdaten korrekt anzeigt. Um es zu aktivieren: `sudo nvidia-smi -pm 1`
- HuggingFace Account für Modelle und geschützte Datasets
- CUDA Toolkit zur dazugehörigen NVIDIA-Treiber-Version (für vLLM-Installation). Können unter https://developer.nvidia.com/cuda-toolkit-archive heruntergeladen werden. Ansonsten geht auch `sudo apt install nvidia-cuda-toolkit`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Die Projektdefaults erwarten `vllm>=0.20.0`. Der Dockerfile nutzt dafür
`pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`.

## HuggingFace Token

Lege `src/.env` aus der Vorlage an und trage deinen HuggingFace-Token ein:

```bash
cp src/.env.example src/.env
# HF_TOKEN=hf_... eintragen
```

Der Python-Code lädt `src/.env` automatisch und spiegelt den Token in die von
`huggingface_hub` erwarteten Variablennamen. Im Docker-Setup sourced ein
Startup-Hook dieselbe Datei zur Laufzeit; das Token wird nicht ins Image
kopiert.

Direkte HuggingFace-CLI-Aufrufe im Host-Terminal lesen `src/.env` nicht
automatisch. Dafür einmalig in der Shell exportieren:

```bash
set -a
source src/.env
set +a
hf auth whoami
```

## Konfiguration

Profile werden über `--profile` oder `LLM_ROUTING_PROFILE` gewählt.

| Profil  | Hardware | VRAM/GPU | GPUs | Router     | Zielmodelle (Router) | Baseline          |
| ------- | -------- | -------- | ---- | ---------- | -------------------- | ----------------- |
| `local` | RTX 3090 | 24 GB    | 1    | Qwen3.5-2B | 4B, 9B               | Qwen3.5-9B        |
| `uni`   | 2x A100  | 80 GB    | 2    | Qwen3.5-2B | 4B, 9B, 35B-A3B, 27B | Qwen3.5-122B-A10B |

Alle Modelle verwenden AWQ-4-bit-Quantisierung. Welche Modelle tatsächlich
geladen werden können, entscheidet der Scheduler anhand von VRAM-Budget,
KV-Cache-Reserve und CUDA-Kontext-Overhead.

Config-Dateien:

- `src/config/settings_local.yaml`
- `src/config/settings_uni.yaml`

Wichtige Runtime-Defaults:

- `max_model_len`: lokal `16384` für Zielmodelle und `8192` für den Router;
  Uni `32768` für Zielmodelle und `16384` für den Router
- `enforce_eager: false` für Zielmodelle, damit CUDA Graphs genutzt werden
- `gpu_memory_utilization: 0.96` in beiden Profilen
- Router-Ausgabe per `guided_choice` als Label wie `1N`, `2T`, `4T`

## Architektur

### LLM-as-a-Router

Der Router ist ein kleines Qwen3.5-2B-Modell im Non-Thinking-Modus. Er trifft
pro Anfrage zwei Entscheidungen:

1. Zielmodell: Tier 1 bis Tier 4
2. Thinking-Modus: `N` für direkte Antwort, `T` für Reasoning

Die Ausgabe ist ein Compound-Label wie `1N` oder `3T`. `guided_choice` stellt
sicher, dass nur gültige Labels generiert werden.

### Scheduler

Der Scheduler verwaltet Zielmodelle in drei Zuständen:

- `RUNNING`: Modell liegt im VRAM und ist inferenzbereit
- `SLEEPING`: Prozess lebt, Weights liegen via vLLM Sleep Mode im CPU-RAM
- `STOPPED`: kein Prozess, kein VRAM/RAM-Footprint

Zusätzlich bietet er Per-GPU-VRAM-Tracking, GPU-Pinning, LRU-Eviction,
Smart Preload und Execution Planning nach aktuellem Modellzustand.

### vLLM Server Mode

Jedes Zielmodell läuft als separater vLLM HTTP-Server mit Chat-Completions-API
und Sleep/Wake-Support. Der Router selbst läuft offline über `vllm.LLM`.

## Benchmarks

| Benchmark      | Typ               | Evaluation                               | Samples    |
| -------------- | ----------------- | ---------------------------------------- | ---------- |
| `livebench-da` | Datenanalyse      | task-spezifischer Ground-Truth-Vergleich | 150 (Test) |
| `gpqa`         | Graduate-Level MC | lm-eval Replay / Exact Match (A-D)       | 198        |
| `mmlu-pro`     | Alltagswissen MC  | lm-eval Replay / Exact Match (A-J)       | 1.000      |
| `bigcodebench` | Code-Generierung  | offizielle Testcase-Ausführung           | 148 (hard) |
| `mixed`        | gemischt          | pro Source-Benchmark                     | 400        |

`mixed` zieht standardmäßig 100 Samples aus jeder Suite: GPQA, LiveBench Data
Analysis, MMLU-Pro und BigCodeBench.

MMLU-Pro ist der Alltagswissens-Benchmark in diesem Setup. Gegenüber klassischem
MMLU erhöht er die Antwortoptionen von 4 auf bis zu 10 und reduziert dadurch
blindes Raten sowie reines Pattern-Matching. Genutzt werden nur die vier
alltagsnahen Kategorien `business`, `health`, `law` und `psychology`, jeweils
mit den ersten 250 Fragen. Damit umfasst der Lauf 1.000 Multiple-Choice-Fragen
mit lokalen A-J-Referenzantworten. Die Auswahl entspricht den lm-eval-Tasks
`mmlu_pro_business`, `mmlu_pro_health`, `mmlu_pro_law` und
`mmlu_pro_psychology`.

## Nutzung

### Interaktive CLI

```bash
python run.py
```

Die Textual-CLI führt durch Profil, Benchmark und Parameter. Ergebnisanzeige
und Experimentvergleich sind als Baum strukturiert: Benchmark, Experimenttyp
und Ergebnis-Unterpfad bilden Gruppen; einzelne Resultate bleiben als Blätter
auswählbar. Beim Vergleichen kannst du weiterhin beliebige Experimente aus
verschiedenen Pfaden auswählen.

Die CLI bietet:

- Baseline-, Router- und Random-Routing-Experimente
- Modell-Download-Verwaltung
- Benchmark-Auswahl: `livebench-da`, `gpqa`, `mmlu-pro`, `bigcodebench`, `mixed`
- Parameter: Prompts, Batch-Size, Temperatur (Default `0.0`), Alias
- Ergebnisdetails, Plot-Regeneration und Cross-Experiment-Vergleiche

### Schnelltest der Energiemessung

```bash
cd src
python -m experiments.test_energy
```

### Baseline-Experiment

```bash
cd src

# Default: mmlu-pro
python -m experiments.baseline --profile local

# Benchmark wählen
python -m experiments.baseline --profile uni --benchmark livebench-da

# Kurzer Testlauf
python -m experiments.baseline --profile local --benchmark mmlu-pro --prompts 25

# Einzelmodell-Baseline
python -m experiments.baseline --profile uni --benchmark gpqa --model qwen3.5-9b
```

Wichtige Optionen:

| Option              | Beschreibung                                                    |
| ------------------- | --------------------------------------------------------------- |
| `--profile, -p`     | `local` oder `uni`                                              |
| `--benchmark, -b`   | `mmlu-pro`, `livebench-da`, `gpqa`, `bigcodebench` oder `mixed` |
| `--prompts, -n`     | Anzahl der Prompts                                              |
| `--batch-size`      | Batch-Size                                                      |
| `--model, -m`       | einzelnes Modell aus dem Katalog als Baseline testen            |
| `--temperature, -t` | Temperatur für die Messung, Default `0.0`                       |
| `--alias`           | Anzeigename für Ergebnislisten und Plots                        |

### Router-Experiment

```bash
cd src

python -m experiments.router --profile uni --benchmark mmlu-pro
python -m experiments.router --profile uni --benchmark livebench-da
python -m experiments.router --profile uni --benchmark mixed --batch-size 64 --alias router_mixed_b64
```

### Random-Routing

```bash
cd src
python -m experiments.random_routing --profile uni --benchmark mmlu-pro --seed 42
```

Random-Routing nutzt denselben Scheduler, wählt Modell und Thinking-Modus aber
zufällig. Dadurch dient es als untere Vergleichsbasis für intelligentes Routing.

### Neu-Auswertung bestehender Resultate

```bash
python -m src.evaluation.reevaluate src/results/router_mmlu-pro_20260601_120000
python -m src.evaluation.reevaluate src/results/router_bigcodebench_20260601_120000 --no-plots
```

Die Neu-Auswertung arbeitet rein aus gespeicherten Generierungen:

- GPQA über lm-eval Replay
- LiveBench DA über task-spezifische Scoring-Funktionen
- MMLU-Pro über lm-eval Replay
- BigCodeBench über die offizielle Execution-Engine

### Cross-Experiment-Vergleich

```bash
cd src
python -m experiments.compare results/router_mmlu-pro_*/ results/baseline_mmlu-pro_*/

python -m experiments.compare \
  results/router_livebench-da_*/ \
  results/single_qwen3.5-122b-moe_livebench-da_*/ \
  --output results/comparison_livebench-da
```

Vergleichsplots werden in `results/comparison_*` geschrieben und enthalten
unter anderem Energie, Accuracy, EQ-Score, Pareto-Plot und Power-Overlay.

### Durchschnitt aus mehreren Runs

Für randomisierte Benchmarks wie `mixed` können mehrere kompatible Runs zu
einem Average-Result gebündelt werden. Dieses Ergebnis enthält mehrere
Messungen in einer `results.json` und kann anschließend wie ein normaler Run in
Vergleichen verwendet werden.

```bash
cd src
python -m experiments.average \
  results/router_mixed_run1/ \
  results/router_mixed_run2/ \
  results/router_mixed_run3/ \
  results/router_mixed_run4/ \
  --alias "Router mixed avg4"
```

In der interaktiven CLI geht das über `Experimente vergleichen`: mehrere Runs
mit Space auswählen und dann `Durchschnitt bilden` starten.

## Ergebnisformat

Jedes Experiment schreibt ein Ergebnisverzeichnis:

- `results.json`: Konfiguration und Messung(en)
- `config.json`: Konfigurationssnapshot
- `measurement_summary.csv`: flache Messungszusammenfassung
- `samples.csv`: flache Sample-Tabelle inklusive Benchmark-Metadaten
- `power_samples.csv`: GPU-Leistungszeitreihe
- `scheduler_events.csv`: Scheduler-Ereignisse bei Router/Random-Routing
- `average_summary.json`: Metadaten und Mittelwerte bei Average-Results
- `plots/`: automatisch generierte PNG-Plots

Beispiel:

```text
src/results/
├── baseline_mmlu-pro_20260601_120000/
│   ├── results.json
│   ├── measurement_summary.csv
│   ├── samples.csv
│   ├── power_samples.csv
│   └── plots/
├── router_mmlu-pro_20260601_123000/
│   └── ...
└── comparison_20260601_130000/
    └── ...
```

## Projektstruktur

```text
bachelorsthesis-code/
├── run.py
├── requirements.txt
├── Dockerfile
└── src/
    ├── cli.py              # Textual-TUI
    ├── benchmarks/         # GPQA, LiveBench DA, MMLU-Pro, BigCodeBench, Mixed
    ├── config/             # Profile und YAML-Konfigurationen
    ├── energy/             # NVML-Energiemessung
    ├── evaluation/         # Replay, Scoring und Post-Processing
    ├── experiments/        # Baseline, Router, Random, Compare
    ├── inference/          # vLLM Offline/HTTP-Client/Server-Manager
    ├── plotting/           # Matplotlib-Plots
    ├── routing/            # LLMRouter und RandomRouter
    ├── scheduler/          # Modellzustände, LRU, Sleep/Wake
    └── utils/              # Daten, Logging, Metriken, Prozesse
```

## GPU-Hinweise

Vor jedem Experiment wird die Idle-Power der GPU gemessen. Für belastbare
Messungen sollten keine anderen GPU-lastigen Prozesse laufen.

Persistenz-Modus kann stabilere Messungen liefern:

```bash
sudo nvidia-smi -pm 1
```

Power-Limits prüfen:

```bash
nvidia-smi -q -d POWER
```

## Troubleshooting

### Triton Compilation: `Python.h` fehlt

Wenn vLLM/Triton beim Kompilieren von Kernels `Python.h` nicht findet, fehlen
die Python-Entwicklungsheader:

```bash
sudo apt install python3.12-dev
```

Alternativ kann `enforce_eager: true` in der YAML-Konfiguration gesetzt werden.
Das ist langsamer und in bisherigen Messungen weniger energieeffizient, aber
nützlich für kontrollierte CUDA-Graphs-vs.-Eager-Vergleiche.
