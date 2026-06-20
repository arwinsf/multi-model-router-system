"""Interaktive TUI für LLM-Routing Experimente.

Fullscreen Terminal-UI mit Textual. Navigation mit Pfeiltasten,
Space zum Auswählen, Enter zum Bestätigen, Escape zurück.

Nutzung:
    python -m src.cli
    python run.py
"""

import json
import os
import re
import sys
import asyncio
import csv
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header,
    Footer,
    Static,
    OptionList,
    SelectionList,
    Input,
    Button,
    RichLog,
    Tree,
)
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection
from textual.containers import Vertical, Horizontal, Center
from textual import on, work
from rich.text import Text
from rich.table import Table

from src.utils.metrics import (
    EQ_SCORE_ENERGY_UNIT,
    compute_accuracy,
    compute_measurement_eq_score,
)

# ─── Konstanten ─────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).parent
RESULTS_DIR = SRC_DIR / "results"
PROJECT_ROOT = SRC_DIR.parent

EXPERIMENTS = [
    (
        "baseline",
        "Baseline-Experiment",
        "Always-Large mit Reasoning (einzelnes Modell)",
    ),
    (
        "router",
        "Router-Experiment",
        "Multi-Modell mit Scheduler und Thinking-Steuerung",
    ),
    (
        "random_routing",
        "Random-Routing-Experiment",
        "Zufällige Modellzuweisung (Vergleichsbasis, kein Router nötig)",
    ),
    ("download", "Modelle herunterladen", "Modelle aus Config-Profil herunterladen"),
    ("results", "Ergebnisse anzeigen", "Vorhandene Experiment-Resultate auslesen"),
    ("compare", "Experimente vergleichen", "Vergleichsplots über mehrere Experimente"),
    ("test_energy", "Energiemessung testen", "Schnelltest der GPU-Energiemessung"),
]

BENCHMARK_DESCRIPTIONS = {
    "livebench-da": "150 Data-Analysis-Aufgaben (LiveBench), objektive Auswertung",
    "gpqa": "~198 Graduate-Level MC-Fragen (Diamond), 4 Optionen (A–D)",
    "mmlu-pro": "1.000 Multiple-Choice-Fragen (Business, Health, Law, Psychology), A-J",
    "bigcodebench": "148 Code-Aufgaben (BigCodeBench-Hard, instruct), Testcase-Ausfuehrung",
    "mixed": "400 Fragen gleichverteilt aus allen 4 Suites",
}


# ─── Pfad-Setup (einmalig auf Modulebene) ───────────────────────────────────

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─── Hilfsfunktionen ────────────────────────────────────────────────────────


def _copy_to_clipboard(text: str) -> bool:
    """Kopiert text in die X11/Wayland-Zwischenablage.

    Probiert xclip, xsel und wl-copy der Reihe nach.
    Gibt True zurück, wenn erfolgreich.

    Hinweis: xclip bleibt als Prozess laufen, um die Zwischenablage zu
    bedienen (X11-Mechanismus). poll() == None nach kurzer Zeit = Erfolg.
    """
    import subprocess
    import time

    encoded = text.encode("utf-8")
    candidates = [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["wl-copy"],
    ]
    for cmd in candidates:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.stdin.write(encoded)
            proc.stdin.close()
            time.sleep(0.05)
            rc = proc.poll()
            # None  → Prozess läuft noch (xclip hält Clipboard) = Erfolg
            # 0     → Prozess sauber beendet (xsel / wl-copy)    = Erfolg
            if rc is None or rc == 0:
                return True
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return False


def _get_accuracy_stats(samples: list[dict]) -> tuple[float | None, int, int]:
    """Liefert Accuracy, Anzahl korrekter und Anzahl ausgewerteter Samples."""
    accuracy = compute_accuracy(samples)
    if accuracy is None:
        return None, 0, 0

    evaluated = [sample for sample in samples if sample.get("is_correct") is not None]
    correct = sum(1 for sample in evaluated if sample.get("is_correct"))
    return accuracy, correct, len(evaluated)


def _mean_numeric(values: list[float | int | None]) -> float | None:
    """Mittelwert fuer vorhandene numerische Werte."""
    numeric_values = [
        float(value) for value in values if isinstance(value, (int, float))
    ]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _measurement_accuracy_value(measurement: dict) -> float | None:
    stored = measurement.get("measurement_accuracy")
    if isinstance(stored, (int, float)):
        return float(stored)
    return compute_accuracy(measurement.get("samples", []))


def _build_average_metrics(measurements: list[dict]) -> dict[str, float | int | None]:
    """Berechnet Mittelwerte ueber mehrere Messungen fuer UI und Clipboard."""
    accuracies = [
        accuracy
        for measurement in measurements
        if (accuracy := _measurement_accuracy_value(measurement)) is not None
    ]
    eq_scores = [
        score
        for measurement in measurements
        if (score := compute_measurement_eq_score(measurement)) is not None
    ]
    return {
        "run_count": len(measurements),
        "total_samples": sum(
            len(measurement.get("samples", [])) for measurement in measurements
        ),
        "mean_accuracy": (sum(accuracies) / len(accuracies) if accuracies else None),
        "mean_dynamic_energy_joules": _mean_numeric(
            [
                measurement.get("measurement_dynamic_energy_joules")
                for measurement in measurements
            ]
        ),
        "mean_dynamic_energy_watthours": _mean_numeric(
            [
                measurement.get("measurement_dynamic_energy_watthours")
                for measurement in measurements
            ]
        ),
        "mean_millijoules_per_output_token": _mean_numeric(
            [
                measurement.get("measurement_millijoules_per_output_token")
                for measurement in measurements
            ]
        ),
        "mean_tokens_per_second": _mean_numeric(
            [
                measurement.get("measurement_tokens_per_second")
                for measurement in measurements
            ]
        ),
        "mean_duration_seconds": _mean_numeric(
            [
                measurement.get("measurement_duration_seconds")
                for measurement in measurements
            ]
        ),
        "mean_eq_score": (sum(eq_scores) / len(eq_scores) if eq_scores else None),
    }


def _normalize_result_dir(path: Path) -> Path:
    """Normalisiert canonical results.json-Pfade auf ihr Experiment-Verzeichnis."""
    return path.parent if path.name == "results.json" else path


def _normalize_comparison_dir(path: Path) -> Path:
    """Normalisiert canonical comparison_inputs.json-Pfade auf ihr Comparison-Verzeichnis."""
    return path.parent if path.name == "comparison_inputs.json" else path


def _format_result_relative_dir(path: Path) -> str:
    """Gibt den relativen Experiment-Pfad unter src/results zurück."""
    result_dir = _normalize_result_dir(path)
    try:
        return result_dir.relative_to(RESULTS_DIR).as_posix()
    except ValueError:
        return result_dir.name


def _format_comparison_relative_dir(path: Path) -> str:
    """Gibt den relativen Comparison-Pfad unter src/results zurück."""
    comparison_dir = _normalize_comparison_dir(path)
    try:
        return comparison_dir.relative_to(RESULTS_DIR).as_posix()
    except ValueError:
        return comparison_dir.name


def _discover_result_json_files() -> list[Path]:
    """Findet alle canonical results.json-Dateien rekursiv unter src/results."""
    if not RESULTS_DIR.exists():
        return []

    return sorted(
        (path for path in RESULTS_DIR.rglob("results.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _discover_comparison_manifest_files() -> list[Path]:
    """Findet alle Comparison-Manifeste rekursiv unter src/results."""
    if not RESULTS_DIR.exists():
        return []

    return sorted(
        (
            path
            for path in RESULTS_DIR.rglob("comparison_inputs.json")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _load_sidecar_result_info(result_path: Path) -> dict | None:
    """Lädt Result-Metadaten ohne die große results.json zu parsen."""
    result_dir = _normalize_result_dir(result_path)
    config_path = result_dir / "config.json"
    summary_path = result_dir / "measurement_summary.csv"
    average_path = result_dir / "average_summary.json"

    if not config_path.exists() or not summary_path.exists():
        return None

    cfg = _read_json_file(config_path)
    rows = _read_csv_rows(summary_path)
    if not rows:
        return None

    average_payload = _read_json_file(average_path) if average_path.exists() else {}
    average = (
        average_payload.get("average", {}) if isinstance(average_payload, dict) else {}
    )

    total_samples = _coerce_int(average.get("total_samples"))
    if total_samples is None:
        sample_counts = [
            _coerce_int(row.get("measurement_num_samples")) for row in rows
        ]
        total_samples = sum(value for value in sample_counts if value is not None)

    mean_accuracy = _coerce_float(average.get("mean_accuracy"))
    if mean_accuracy is None:
        accuracies = [_coerce_float(row.get("measurement_accuracy")) for row in rows]
        accuracies = [value for value in accuracies if value is not None]
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else None

    mean_mj = _coerce_float(average.get("mean_millijoules_per_output_token"))
    if mean_mj is None:
        mjs = [
            _coerce_float(row.get("measurement_millijoules_per_output_token"))
            for row in rows
        ]
        mjs = [value for value in mjs if value is not None]
        mean_mj = sum(mjs) / len(mjs) if mjs else None

    hw = cfg.get("hardware", {}) if isinstance(cfg, dict) else {}
    profile = cfg.get("profile") if isinstance(cfg, dict) else None
    n_gpus = hw.get("num_gpus") if isinstance(hw, dict) else None
    vram = hw.get("per_gpu_vram_gb") if isinstance(hw, dict) else None
    if n_gpus is not None and vram is not None:
        hw_str = f"{n_gpus}× {int(vram)}GB GPU"
        if profile:
            hw_str += f" ({profile})"
    elif profile:
        hw_str = f"({profile})"
    else:
        hw_str = None

    return {
        "display_name": cfg.get("experiment_alias") or result_dir.name,
        "folder_name": _format_result_relative_dir(result_path),
        "experiment_type": cfg.get("experiment_type", "?"),
        "benchmark": cfg.get("benchmark", "?"),
        "batch_size": cfg.get("batch_size", "?"),
        "measurement_mode": cfg.get("measurement_mode", "single"),
        "num_measurements": len(rows),
        "mtime": result_path.stat().st_mtime,
        "total_samples": total_samples or 0,
        "scenarios": sorted(
            {str(row.get("scenario", "?")) for row in rows if row.get("scenario")}
        ),
        "hw_str": hw_str,
        "avg_accuracy": mean_accuracy * 100 if mean_accuracy is not None else None,
        "avg_mj_per_token": mean_mj,
    }


def _compose_result_heading(display_name: str, path: Path) -> str:
    """Erzeugt einen UI-Titel mit optionalem Unterpfad-Kontext."""
    folder_name = _format_result_relative_dir(path)
    lines = [f"[bold cyan]{display_name}[/]"]
    if folder_name != display_name:
        lines.append(f"[dim]{folder_name}[/]")
    lines.append("[dim]Esc Zurück[/]")
    return "\n".join(lines)


def _compose_comparison_heading(display_name: str, path: Path) -> str:
    """Erzeugt einen UI-Titel für Comparison-Verzeichnisse."""
    folder_name = _format_comparison_relative_dir(path)
    lines = [f"[bold cyan]{display_name}[/]"]
    if folder_name != display_name:
        lines.append(f"[dim]{folder_name}[/]")
    lines.append("[dim]Esc Zurück[/]")
    return "\n".join(lines)


def load_profile_info() -> dict:
    """Lädt Informationen über verfügbare Profile."""
    from src.config import PROFILES, load_config

    profiles = {}
    for name, path in PROFILES.items():
        try:
            config = load_config(name)
            models = config.get("models", [])
            router = config.get("router", {})
            experiment = config.get("experiment", {})
            profiles[name] = {
                "config": config,
                "models": models,
                "router_model": router.get("model", "?"),
                "batch_size": experiment.get("batch_size", 32),
            }
        except Exception:
            pass
    return profiles


def get_available_benchmarks() -> list[str]:
    """Gibt verfügbare Benchmarks zurück."""
    from src.benchmarks import list_benchmarks

    return list_benchmarks()


def build_command(
    experiment_type: str,
    profile: str | None = None,
    benchmark: str | None = None,
    scenarios: list[str] | None = None,
    params: dict | None = None,
    selected_model: str | None = None,
) -> list[str]:
    """Baut den Aufrufbefehl zusammen."""
    if experiment_type == "test_energy":
        return [sys.executable, "-m", "experiments.test_energy"]

    module = f"experiments.{experiment_type}"
    cmd = [sys.executable, "-m", module]

    if profile:
        cmd.extend(["--profile", profile])
    if benchmark:
        cmd.extend(["--benchmark", benchmark])
    if selected_model and experiment_type == "baseline":
        cmd.extend(["--model", selected_model])
    if scenarios and experiment_type not in ("baseline", "random_routing"):
        cmd.extend(["--scenario", ",".join(scenarios)])
    if params:
        if params.get("num_prompts"):
            cmd.extend(["--prompts", str(params["num_prompts"])])
        if params.get("batch_size"):
            cmd.extend(["--batch-size", str(params["batch_size"])])
        if params.get("temperature") is not None:
            cmd.extend(["--temperature", str(params["temperature"])])
        if params.get("alias"):
            cmd.extend(["--alias", params["alias"]])
    return cmd


# ─── Screens ────────────────────────────────────────────────────────────────


class ProfileScreen(Screen):
    """Schritt 2: Profil wählen."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 2[/] — Profil wählen\n"
                "[dim]↑↓ navigieren · Enter auswählen · Esc zurück[/]",
                id="step",
            )
            self._keys = []
            options = []
            for name, info in self.app.profiles.items():
                model_names = [m["name"].split("/")[-1] for m in info["models"]]
                router_short = info["router_model"].split("/")[-1]
                options.append(
                    Option(
                        Text.assemble(
                            (name, "bold"),
                            "\n",
                            (
                                f"  Router: {router_short} · Modelle: {', '.join(model_names)}",
                                "dim",
                            ),
                        )
                    )
                )
                self._keys.append(name)
            yield OptionList(*options, id="options")
        yield Footer()

    @on(OptionList.OptionSelected, "#options")
    def on_select(self, event: Tree.NodeSelected) -> None:
        name = self._keys[event.option_index]
        self.app.wizard["profile"] = name
        self.app.push_screen(BenchmarkScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()


class BenchmarkScreen(Screen):
    """Schritt 3: Benchmark wählen."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 3[/] — Benchmark wählen\n"
                "[dim]↑↓ navigieren · Enter auswählen · Esc zurück[/]",
                id="step",
            )
            self._names = []
            options = []
            for name in self.app.benchmarks:
                desc = BENCHMARK_DESCRIPTIONS.get(name, "")
                options.append(
                    Option(
                        Text.assemble(
                            (name, "bold"),
                            ("\n  " + desc, "dim") if desc else ("", ""),
                        )
                    )
                )
                self._names.append(name)
            yield OptionList(*options, id="options")
        yield Footer()

    @on(OptionList.OptionSelected, "#options")
    def on_select(self, event: OptionList.OptionSelected) -> None:
        name = self._names[event.option_index]
        self.app.wizard["benchmark"] = name
        if self.app.wizard["experiment"] == "baseline":
            # Baseline: Modellauswahl anzeigen (Baseline-Modell + Katalog)
            self.app.push_screen(ModelSelectScreen())
        else:
            self.app.push_screen(ParametersScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ModelSelectScreen(Screen):
    """Schritt 3b: Modell wählen (Baseline / Einzelmodell-Test)."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 3b[/] — Modell wählen\n"
                "[dim]↑↓ navigieren · Enter auswählen · Esc zurück[/]\n"
                "[dim]Das Baseline-Modell ist die Standard-Auswahl.\n"
                "Andere Modelle werden einzeln im Thinking-Modus getestet.[/]",
                id="step",
            )
            profile = self.app.wizard["profile"]
            info = self.app.profiles[profile]
            models = info["models"]
            config = info["config"]
            baseline = config.get("baseline", {})

            self._model_choices = []
            options = []

            # Baseline-Modell als erste Option
            bl_id = baseline.get("model_id", "?")
            bl_name = baseline.get("model_name", "?").split("/")[-1]
            bl_vram = baseline.get("vram_gb", "?")
            options.append(
                Option(
                    Text.assemble(
                        (f"[Baseline] {bl_id}", "bold"),
                        "\n",
                        (f"  {bl_name} · {bl_vram} GB · Thinking aktiviert", "dim"),
                    )
                )
            )
            self._model_choices.append(None)  # None = Standard-Baseline

            # Katalog-Modelle
            for m in models:
                short = m["name"].split("/")[-1]
                desc = m.get("description", "")
                vram = m.get("vram_gb", "?")
                options.append(
                    Option(
                        Text.assemble(
                            (m["id"], "bold"),
                            "\n",
                            (f"  {short} · {vram} GB · {desc}", "dim"),
                        )
                    )
                )
                self._model_choices.append(m["id"])

            yield OptionList(*options, id="options")
        yield Footer()

    @on(OptionList.OptionSelected, "#options")
    def on_select(self, event: OptionList.OptionSelected) -> None:
        model_id = self._model_choices[event.option_index]
        self.app.wizard["selected_model"] = model_id  # None = Baseline
        if model_id:
            self.app.wizard["scenarios"] = [model_id]
        else:
            profile = self.app.wizard["profile"]
            baseline = self.app.profiles[profile]["config"].get("baseline", {})
            self.app.wizard["scenarios"] = [baseline.get("model_id", "")]
        self.app.push_screen(ParametersScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ScenarioScreen(Screen):
    """Schritt 3b: Modelle wählen (nur Baseline)."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 3b[/] — Modelle wählen\n"
                "[dim]↑↓ navigieren · Space an/abwählen · Esc zurück[/]",
                id="step",
            )
            profile = self.app.wizard["profile"]
            models = self.app.profiles[profile]["models"]
            selections = []
            for m in models:
                short = m["name"].split("/")[-1]
                desc = m.get("description", "")
                label = f'{m["id"]} ({short})'
                if desc:
                    label += f" — {desc}"
                selections.append(Selection(label, m["id"], True))
            yield SelectionList(*selections, id="scenarios")
            with Center(classes="button-row"):
                yield Button("Weiter", variant="primary", id="btn-next")
        yield Footer()

    @on(Button.Pressed, "#btn-next")
    def on_next(self) -> None:
        sel = self.query_one("#scenarios", SelectionList)
        selected = list(sel.selected)
        if not selected:
            self.notify("Mindestens ein Modell wählen.", severity="warning")
            return
        self.app.wizard["scenarios"] = selected
        self.app.push_screen(ParametersScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ParametersScreen(Screen):
    """Schritt 4: Parameter konfigurieren."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        profile = self.app.wizard.get("profile", "local")
        info = self.app.profiles.get(profile, {})
        benchmark = self.app.wizard.get("benchmark", "")
        from src.benchmarks import RECOMMENDED_SAMPLES

        rec = RECOMMENDED_SAMPLES.get(benchmark)
        default_samples = str(rec) if rec is not None else "alle"
        default_batch = str(info.get("batch_size", 32))

        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 4[/] — Parameter\n"
                "[dim]Tab zwischen Feldern · Esc zurück[/]",
                id="step",
            )
            with Vertical(id="form"):
                yield Static(
                    f'Anzahl Prompts [dim](leer/"alle" = Benchmark-Default)[/]:',
                    classes="label",
                )
                yield Input(
                    value=default_samples,
                    type="text",
                    id="num_prompts",
                )
                yield Static("Batch-Size:", classes="label")
                yield Input(
                    value=default_batch,
                    type="integer",
                    id="batch_size",
                )
                yield Static(
                    "Temperatur [dim](Default 0 fuer Messungen)[/]:",
                    classes="label",
                )
                yield Input(
                    value="0",
                    placeholder="0",
                    type="text",
                    id="temperature",
                )
                yield Static(
                    "Alias [dim](optional fuer Plots/Listen)[/]:",
                    classes="label",
                )
                yield Input(
                    value="",
                    placeholder="z.B. livebench bs16 graph",
                    type="text",
                    id="alias",
                )
            with Center(classes="button-row"):
                yield Button("Weiter", variant="primary", id="btn-next")
        yield Footer()

    @on(Button.Pressed, "#btn-next")
    def on_next(self) -> None:
        try:
            raw_prompts = self.query_one("#num_prompts", Input).value.strip().lower()
            if raw_prompts in ("alle", "all", ""):
                num_prompts = None  # Benchmark-Default
            else:
                num_prompts = int(raw_prompts)
                if num_prompts < 1:
                    self.notify("Prompts muss >= 1 sein.", severity="error")
                    return

            batch_size = int(self.query_one("#batch_size", Input).value or "0")
        except ValueError:
            self.notify("Bitte gültige Zahlen eingeben.", severity="error")
            return

        if batch_size < 1:
            self.notify("Batch-Size muss >= 1 sein.", severity="error")
            return

        params = {
            "num_prompts": num_prompts,
            "batch_size": batch_size,
        }

        try:
            temp_val = self.query_one("#temperature", Input).value.strip()
            if temp_val:
                params["temperature"] = float(temp_val)
        except Exception:
            pass

        try:
            alias_val = self.query_one("#alias", Input).value.strip()
            if alias_val:
                params["alias"] = alias_val
        except Exception:
            pass

        self.app.wizard["params"] = params
        self.app.push_screen(ConfirmScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ConfirmScreen(Screen):
    """Schritt 5: Zusammenfassung und Bestätigung."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        w = self.app.wizard
        params = w.get("params", {})

        table = Table(show_header=False, padding=(0, 2), expand=True)
        table.add_column("Parameter", style="dim", ratio=1)
        table.add_column("Wert", style="bold", ratio=2)
        table.add_row("Experiment", w.get("experiment_label", "?"))
        table.add_row("Profil", w.get("profile", "?"))
        table.add_row("Benchmark", w.get("benchmark", "?"))
        if w.get("scenarios"):
            table.add_row("Szenarien", ", ".join(w["scenarios"]))
        table.add_row("Prompts", str(params.get("num_prompts") or "Benchmark-Default"))
        table.add_row("Batch-Size", str(params.get("batch_size", "?")))
        if params.get("alias"):
            table.add_row("Alias", params["alias"])
        if params.get("temperature") is not None:
            table.add_row("Temperatur", str(params["temperature"]))

        cmd = build_command(
            w["experiment"],
            profile=w.get("profile"),
            benchmark=w.get("benchmark"),
            scenarios=w.get("scenarios"),
            params=w.get("params"),
            selected_model=w.get("selected_model"),
        )
        import shlex

        table.add_row("Kommando", shlex.join(cmd))

        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 5[/] — Bestätigung\n" "[dim]Esc zurück[/]",
                id="step",
            )
            yield Static(table, id="summary")
            with Horizontal(id="confirm-buttons"):
                yield Button("Experiment starten", variant="success", id="btn-start")
                yield Button("Zurück", variant="default", id="btn-back")
        yield Footer()

    @on(Button.Pressed, "#btn-start")
    def on_start(self) -> None:
        w = self.app.wizard
        cmd = build_command(
            w["experiment"],
            profile=w.get("profile"),
            benchmark=w.get("benchmark"),
            scenarios=w.get("scenarios"),
            params=w.get("params"),
            selected_model=w.get("selected_model"),
        )
        self.app.push_screen(
            ProcessScreen(cmd=cmd, title=w.get("experiment_label", "Experiment"))
        )

    @on(Button.Pressed, "#btn-back")
    def on_back(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ProcessScreen(Screen):
    """Experiment ausführen und Ausgabe streamen."""

    BINDINGS = [
        ("escape", "go_home", "Hauptmenü"),
    ]

    def __init__(self, cmd: list[str], title: str = "", cwd: str | None = None) -> None:
        super().__init__()
        self._cmd = cmd
        self._title = title
        self._cwd = cwd or str(SRC_DIR)
        self._process: asyncio.subprocess.Process | None = None
        self._output_lines: list[str] = []
        self._buffer: str = ""
        self._result_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                f"[bold]{self._title}[/]\n" "[dim]Esc = Hauptmenü[/]",
                id="step",
            )
            yield RichLog(highlight=True, markup=False, id="process-log")
            yield Static("", id="progress")
            yield Static("", id="status")
            with Center(classes="button-row"):
                yield Button(
                    "Ausgabe kopieren",
                    variant="primary",
                    id="btn-copy",
                )
                yield Button(
                    "Alias setzen",
                    variant="default",
                    id="btn-alias",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#btn-alias", Button).display = False
        self._execute_process()

    def _extract_result_path(self) -> Path | None:
        pattern = re.compile(r"Gespeichert:\s+(.+results\.json)$")
        for line in reversed(self._output_lines):
            match = pattern.search(line.strip())
            if not match:
                continue
            candidate = Path(match.group(1).strip())
            if not candidate.is_absolute():
                candidate = (Path(self._cwd) / candidate).resolve()
            return candidate
        return None

    @work(exclusive=True)
    async def _execute_process(self) -> None:
        import shlex

        log = self.query_one("#process-log", RichLog)
        progress = self.query_one("#progress", Static)
        status = self.query_one("#status", Static)

        log.write(Text.assemble(("$ " + shlex.join(self._cmd), "dim")))
        log.write("")

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd,
                env=env,
            )

            while True:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    break
                self._buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    display = line.rsplit("\r", 1)[-1]
                    log.write(Text.from_ansi(display))
                    self._output_lines.append(display)

                # Zeige aktuelle unvollständige Zeile (z.B. tqdm Fortschritt)
                current = self._buffer.rsplit("\r", 1)[-1].strip()
                if current:
                    progress.update(Text.from_ansi(current))
                else:
                    progress.update("")

            # Rest im Buffer ausgeben
            if self._buffer.strip():
                display = self._buffer.rsplit("\r", 1)[-1]
                log.write(Text.from_ansi(display))
                self._output_lines.append(display)
            self._buffer = ""
            progress.update("")

            returncode = await self._process.wait()
            self._process = None

            if returncode == 0:
                status.update("[bold green]✓ Erfolgreich abgeschlossen[/]")
                self._result_path = self._extract_result_path()
                if self._result_path and self._result_path.exists():
                    self.query_one("#btn-alias", Button).display = True
            else:
                status.update(f"[bold red]✗ Fehlercode {returncode}[/]")
        except asyncio.CancelledError:
            if self._process:
                self._process.terminate()
            raise
        except Exception as e:
            status.update(f"[bold red]Fehler: {e}[/]")

    @on(Button.Pressed, "#btn-copy")
    def on_copy(self) -> None:
        """Gesamte Ausgabe in die Zwischenablage kopieren."""
        text = "\n".join(self._output_lines)
        if not text:
            self.notify("Noch keine Ausgabe vorhanden.", severity="warning")
            return
        if _copy_to_clipboard(text):
            self.notify("Ausgabe in Zwischenablage kopiert.")
        else:
            # Fallback: in Datei schreiben
            import tempfile

            tmp = Path(tempfile.gettempdir()) / "llm_routing_output.txt"
            tmp.write_text(text, encoding="utf-8")
            self.notify(
                f"Kein Clipboard-Tool gefunden. Ausgabe gespeichert: {tmp}",
                severity="warning",
                timeout=8,
            )

    @on(Button.Pressed, "#btn-alias")
    def on_set_alias(self) -> None:
        if self._result_path is None:
            self.notify("Kein Ergebnis-Run erkannt.", severity="warning")
            return
        self.app.push_screen(AliasEditScreen(self._result_path))

    def action_go_home(self) -> None:
        if self._process:
            self._process.terminate()
        self.app.go_home()

    def on_unmount(self) -> None:
        """Subprozess beenden, wenn der Screen entfernt wird (App-Exit, Escape)."""
        if self._process:
            self._process.terminate()


class AliasEditScreen(Screen):
    """Setzt einen optionalen Alias für einen bestehenden Run."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                f"[bold cyan]Alias setzen[/]\n[dim]{_format_result_relative_dir(self._path)}[/]",
                id="step",
            )
            yield Static(
                "[dim]Optionaler Anzeigename fuer Results, Compare und Vergleichsplots.[/]",
            )
            yield Input(
                value="", placeholder="z.B. livebench bs16 graph", id="alias-input"
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Speichern", variant="success", id="btn-save-alias")
                yield Button("Alias löschen", variant="warning", id="btn-clear-alias")
                yield Button("Abbrechen", variant="default", id="btn-cancel-alias")
        yield Footer()

    def on_mount(self) -> None:
        from src.utils.data import get_experiment_alias, load_results

        try:
            data = load_results(self._path)
            current_alias = get_experiment_alias(data.get("config", {})) or ""
            self.query_one("#alias-input", Input).value = current_alias
        except Exception:
            pass

    @on(Button.Pressed, "#btn-save-alias")
    def on_save_alias(self) -> None:
        from src.utils.data import set_experiment_alias

        alias = self.query_one("#alias-input", Input).value.strip()
        set_experiment_alias(self._path, alias)
        self.app.pop_screen()
        if alias:
            self.app.notify(f"Alias gesetzt: {alias}")
        else:
            self.app.notify("Alias entfernt.")

    @on(Button.Pressed, "#btn-clear-alias")
    def on_clear_alias(self) -> None:
        from src.utils.data import set_experiment_alias

        set_experiment_alias(self._path, None)
        self.app.pop_screen()
        self.app.notify("Alias entfernt.")

    @on(Button.Pressed, "#btn-cancel-alias")
    def on_cancel_alias(self) -> None:
        self.app.pop_screen()

    def action_go_back(self) -> None:
        self.app.pop_screen()


def _load_result_files() -> list[tuple[Path, dict]]:
    """Lädt alle canonical Result-Dateien mit Metadaten."""
    files = _discover_result_json_files()
    if not files:
        return []

    result = []
    for f in files:
        relative_dir = _format_result_relative_dir(f)
        info = {
            "name": relative_dir,
            "total_samples": 0,
            "scenarios": [],
        }

        # Zusätzliche Metadaten aus JSON laden (Accuracy, Energie, Typ, Benchmark)
        try:
            sidecar_info = _load_sidecar_result_info(f)
            if sidecar_info is not None:
                info.update(sidecar_info)
            else:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                from src.utils.data import (
                    get_experiment_display_name,
                    normalize_results,
                )

                normalized = normalize_results(data)
                cfg = normalized.get("config", {})
                measurements = normalized.get("measurements", [])
                info["display_name"] = get_experiment_display_name(f, cfg)
                info["folder_name"] = relative_dir
                info["experiment_type"] = cfg.get("experiment_type", "?")
                info["benchmark"] = cfg.get("benchmark", "?")
                info["batch_size"] = cfg.get("batch_size", "?")
                info["measurement_mode"] = cfg.get("measurement_mode", "single")
                info["num_measurements"] = len(measurements)
                info["mtime"] = f.stat().st_mtime
                info["total_samples"] = sum(
                    len(measurement.get("samples", [])) for measurement in measurements
                )
                info["scenarios"] = sorted(
                    {
                        str(measurement.get("scenario", "?"))
                        for measurement in measurements
                        if isinstance(measurement, dict)
                    }
                )
                hw = cfg.get("hardware", {})
                profile = cfg.get("profile", None)
                n_gpus = hw.get("num_gpus")
                vram = hw.get("per_gpu_vram_gb")
                if n_gpus is not None and vram is not None:
                    info["hw_str"] = f"{n_gpus}× {int(vram)}GB GPU"
                    if profile:
                        info["hw_str"] += f" ({profile})"
                elif profile:
                    info["hw_str"] = f"({profile})"
                else:
                    info["hw_str"] = None
                accs, mjs = [], []
                for measurement in measurements:
                    samples = measurement.get("samples", [])
                    accuracy, _, _ = _get_accuracy_stats(samples)
                    if accuracy is not None:
                        accs.append(accuracy * 100)
                    mj = measurement.get("measurement_millijoules_per_output_token")
                    if mj is not None:
                        mjs.append(mj)
                info["avg_accuracy"] = sum(accs) / len(accs) if accs else None
                info["avg_mj_per_token"] = sum(mjs) / len(mjs) if mjs else None
        except Exception:
            pass

        info.setdefault("display_name", _normalize_result_dir(f).name)
        info.setdefault("folder_name", relative_dir)

        result.append((f, info))
    return result


def _split_portable_relative_path(value: str) -> tuple[str, ...]:
    """Splits stored result paths independent of Windows/Unix separators."""
    normalized = value.replace("\\", "/")
    return tuple(
        part for part in PurePosixPath(normalized).parts if part not in ("", ".")
    )


def _result_tree_parts(path: Path, info: dict | None = None) -> tuple[str, ...]:
    """Returns the real directory parts under src/results, including the run folder."""
    result_dir = _normalize_result_dir(path)
    try:
        relative = result_dir.resolve().relative_to(RESULTS_DIR.resolve())
        return tuple(relative.parts)
    except ValueError:
        folder_name = str(
            (info or {}).get("folder_name") or _format_result_relative_dir(path)
        )
        return _split_portable_relative_path(folder_name)


def _format_result_leaf(path: Path, info: dict, selected: bool | None = None) -> Text:
    from datetime import datetime

    display_name = info.get("display_name", path.parent.name)
    folder_name = info.get("folder_name", _format_result_relative_dir(path))
    total = info.get("total_samples", 0)
    mtime = info.get("mtime")
    date_str = datetime.fromtimestamp(mtime).strftime("%d.%m. %H:%M") if mtime else ""
    acc = info.get("avg_accuracy")
    mj = info.get("avg_mj_per_token")
    batch_size = info.get("batch_size")
    hw_str = info.get("hw_str")
    num_measurements = info.get("num_measurements") or 1
    measurement_mode = info.get("measurement_mode")

    prefix = ""
    if selected is not None:
        prefix = "[x] " if selected else "[ ] "

    acc_str = f" · Acc {acc:.0f}%" if acc is not None else ""
    mj_str = f" · {mj:.0f} mJ/tok" if mj is not None else ""
    batch_str = f" · Batch {batch_size}" if batch_size is not None else ""
    if measurement_mode == "average":
        run_str = f" · Mittel {num_measurements} Runs"
    elif num_measurements > 1:
        run_str = f" · {num_measurements} Messungen"
    else:
        run_str = ""
    hw_part = f" · {hw_str}" if hw_str else ""
    folder_part = f" · {folder_name}" if display_name != folder_name else ""

    return Text.assemble(
        (f"{prefix}{display_name}", "bold"),
        "\n",
        (
            f"  {date_str} · {total} Samples{run_str}{acc_str}{mj_str}{batch_str}{hw_part}{folder_part}",
            "dim",
        ),
    )


def _populate_result_tree(
    tree: Tree,
    file_list: list[tuple[Path, dict]],
    selected_paths: list[str] | set[str] | None = None,
) -> None:
    tree.clear()
    tree.root.label = "Ergebnisse"
    tree.root.expand()

    nodes = {}
    sortable = sorted(
        file_list,
        key=lambda item: (
            _result_tree_parts(item[0], item[1]),
            -(item[1].get("mtime") or 0),
        ),
    )

    for path, info in sortable:
        parent = tree.root
        group_path: list[str] = []
        tree_parts = _result_tree_parts(path, info)
        group_parts = tree_parts[:-1]
        for group_label in group_parts:
            group_path.append(group_label)
            key = tuple(group_path)
            if key not in nodes:
                nodes[key] = parent.add(group_label, expand=True)
            parent = nodes[key]

        path_str = str(path)
        selected = selected_paths is not None and path_str in selected_paths
        parent.add_leaf(
            _format_result_leaf(
                path, info, selected if selected_paths is not None else None
            ),
            data={"path": path_str, "info": info},
        )


def _build_result_lookup() -> dict[Path, dict]:
    """Erzeugt einen Lookup von Experiment-Verzeichnissen auf Result-Metadaten."""
    lookup: dict[Path, dict] = {}
    for result_path, info in _load_result_files():
        lookup[_normalize_result_dir(result_path).resolve()] = {
            "result_path": result_path,
            **info,
        }
    return lookup


def _resolve_comparison_experiment_dirs(payload: dict) -> list[Path]:
    """Löst Manifest-Inputpfade robust auf Experiment-Verzeichnisse auf."""
    source_dirs: list[Path] = []
    seen: set[Path] = set()

    for raw_path in payload.get("experiment_dirs", []):
        if not raw_path:
            continue

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

        resolved = candidate.resolve()
        if resolved.is_file() and resolved.name == "results.json":
            resolved = resolved.parent

        if resolved in seen:
            continue

        seen.add(resolved)
        source_dirs.append(resolved)

    return source_dirs


def _build_comparison_info(
    manifest_path: Path,
    result_lookup: dict[Path, dict],
) -> tuple[Path, dict]:
    """Lädt Metadaten für ein Comparison-Verzeichnis."""
    comparison_dir = _normalize_comparison_dir(manifest_path)
    relative_dir = _format_comparison_relative_dir(manifest_path)
    info = {
        "display_name": comparison_dir.name,
        "folder_name": relative_dir,
        "mtime": comparison_dir.stat().st_mtime,
        "source_infos": [],
        "experiment_count": 0,
        "benchmarks": [],
        "experiment_types": [],
        "plot_files": [],
        "missing_count": 0,
        "created_at": None,
        "filters": {},
    }

    try:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        info["manifest_error"] = str(exc)
        payload = {}

    info["created_at"] = payload.get("created_at")
    filters = payload.get("filters")
    if isinstance(filters, dict):
        info["filters"] = filters

    source_infos = []
    benchmarks = set()
    experiment_types = set()

    for source_dir in _resolve_comparison_experiment_dirs(payload):
        resolved_source = source_dir.resolve()
        source_meta = result_lookup.get(resolved_source)
        exists = source_meta is not None or (resolved_source / "results.json").exists()

        display_name = (
            source_meta.get("display_name", resolved_source.name)
            if source_meta
            else resolved_source.name
        )
        folder_name = (
            source_meta.get("folder_name", _format_result_relative_dir(resolved_source))
            if source_meta
            else _format_result_relative_dir(resolved_source)
        )
        experiment_type = (
            source_meta.get("experiment_type", "?") if source_meta else "?"
        )
        benchmark = source_meta.get("benchmark", "?") if source_meta else "?"

        if benchmark and benchmark != "?":
            benchmarks.add(str(benchmark))
        if experiment_type and experiment_type != "?":
            experiment_types.add(str(experiment_type))

        source_infos.append(
            {
                "path": resolved_source,
                "display_name": display_name,
                "folder_name": folder_name,
                "experiment_type": experiment_type,
                "benchmark": benchmark,
                "exists": exists,
            }
        )

    info["source_infos"] = source_infos
    info["experiment_count"] = len(source_infos)
    info["benchmarks"] = sorted(benchmarks)
    info["experiment_types"] = sorted(experiment_types)
    info["missing_count"] = sum(1 for source in source_infos if not source["exists"])
    info["plot_files"] = [
        plot_path.name for plot_path in sorted(comparison_dir.glob("*.png"))
    ]

    return manifest_path, info


def _load_comparison_files() -> list[tuple[Path, dict]]:
    """Lädt alle Comparison-Manifeste mit Metadaten."""
    manifest_files = _discover_comparison_manifest_files()
    if not manifest_files:
        return []

    result_lookup = _build_result_lookup()
    return [
        _build_comparison_info(manifest_path, result_lookup)
        for manifest_path in manifest_files
    ]


class ResultsScreen(Screen):
    """Vorhandene Ergebnisse auflisten."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Ergebnisse[/]\n"
                "[dim]↑↓ navigieren · Enter für Details · Esc Zurück[/]",
                id="step",
            )
            self._file_list = _load_result_files()
            if not self._file_list:
                yield Static(
                    "[red bold]Keine Ergebnis-Dateien gefunden.[/]\n"
                    "[dim]Führe zuerst ein Experiment durch.[/]",
                    id="no-results",
                )
            else:
                yield Tree("Ergebnisse", id="file-tree")
        yield Footer()

    def on_mount(self) -> None:
        if self._file_list:
            tree = self.query_one("#file-tree", Tree)
            _populate_result_tree(tree, self._file_list)

    @on(Tree.NodeSelected, "#file-tree")
    def on_select(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if not isinstance(data, dict) or "path" not in data:
            return
        self.app.push_screen(ResultDetailScreen(Path(data["path"])))

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ResultDetailScreen(Screen):
    """Zeigt Details einer Ergebnis-Datei."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._display_name = _normalize_result_dir(path).name

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                _compose_result_heading(self._display_name, self._path), id="step"
            )
            yield RichLog(highlight=True, markup=True, id="detail-log")
            # Terminal-Plots (plotext) — werden in on_mount befüllt
            try:
                from textual_plotext import PlotextPlot

                with Horizontal(id="plot-row"):
                    yield PlotextPlot(id="plot-power")
                    yield PlotextPlot(id="plot-accuracy")
            except ImportError:
                pass
            with Horizontal(classes="button-row"):
                yield Button("Kopieren", variant="primary", id="btn-copy")
                yield Button(
                    "Neu auswerten",
                    variant="success",
                    id="btn-reevaluate",
                )
                yield Button("Plots neu", variant="default", id="btn-replot")
                yield Button("Alias setzen", variant="default", id="btn-alias")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#detail-log", RichLog)
        if self._try_render_average_sidecar(log):
            return

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.write(Text(f"Fehler beim Laden: {e}", style="bold red"))
            return

        if not data:
            log.write(Text("Datei ist leer.", style="bold red"))
            return

        # Normalisiere zu Messungs-Format (unterstützt alt + neu)
        from src.utils.data import get_experiment_display_name, normalize_results

        normalized = normalize_results(data)
        self._display_name = get_experiment_display_name(
            self._path,
            normalized.get("config", {}),
        )
        self.query_one("#step", Static).update(
            _compose_result_heading(self._display_name, self._path)
        )
        measurements = normalized.get("measurements", [])
        if not measurements:
            log.write(Text("Keine Messdaten gefunden.", style="bold red"))
            return

        multiple_entries = len(measurements) > 1

        # Prüfen ob Router-Experiment
        is_router = any(
            s.get("routing_decision")
            for measurement in measurements
            for s in measurement.get("samples", [])
        )

        total_samples = sum(len(m.get("samples", [])) for m in measurements)

        if multiple_entries:
            average_metrics = _build_average_metrics(measurements)
            average_table = Table(
                title=f"Durchschnitt über {len(measurements)} Messungen",
                expand=True,
                padding=(0, 1),
            )
            average_table.add_column("Metrik", style="dim")
            average_table.add_column("Wert", style="bold")
            average_table.add_row(
                "Samples gesamt", str(average_metrics["total_samples"])
            )
            mean_accuracy = average_metrics.get("mean_accuracy")
            if mean_accuracy is not None:
                average_table.add_row(
                    "Accuracy",
                    f"{mean_accuracy:.4f} ({mean_accuracy * 100:.3f}%)",
                )
            mean_energy = average_metrics.get("mean_dynamic_energy_joules")
            mean_energy_wh = average_metrics.get("mean_dynamic_energy_watthours")
            if mean_energy is not None:
                wh_str = (
                    f" / {mean_energy_wh * 1000:.3f} mWh"
                    if mean_energy_wh is not None
                    else ""
                )
                average_table.add_row(
                    "Dynamische Energie", f"{mean_energy:.2f} J{wh_str}"
                )
            mean_mj = average_metrics.get("mean_millijoules_per_output_token")
            if mean_mj is not None:
                average_table.add_row("Energie/Token (mJ)", f"{mean_mj:.1f}")
            mean_eq = average_metrics.get("mean_eq_score")
            if mean_eq is not None:
                average_table.add_row(
                    f"EQ-Score (Qualitaet/{EQ_SCORE_ENERGY_UNIT})",
                    f"{mean_eq:.3f}",
                )
            mean_tps = average_metrics.get("mean_tokens_per_second")
            if mean_tps is not None:
                average_table.add_row("Tokens/s", f"{mean_tps:.1f}")
            mean_duration = average_metrics.get("mean_duration_seconds")
            if mean_duration is not None:
                average_table.add_row("Dauer (s)", f"{mean_duration:.1f}")
            log.write(average_table)
            log.write("")

        # Übersichtstabelle: eine Zeile pro Messung
        overview = Table(
            title="Einzelmessungen" if multiple_entries else "Übersicht",
            expand=True,
            padding=(0, 1),
        )
        overview.add_column("ID", style="bold")
        overview.add_column("Szenario", style="bold")
        overview.add_column("Samples", justify="right")
        if is_router:
            overview.add_column("Routing", justify="right")
        overview.add_column("Accuracy", justify="right")
        overview.add_column("dyn. Energie", justify="right")
        overview.add_column("mJ/Token", justify="right")
        overview.add_column(f"EQ (Qual./{EQ_SCORE_ENERGY_UNIT})", justify="right")
        overview.add_column("Tok/s", justify="right")

        for measurement in measurements:
            samples = measurement.get("samples", [])
            n = len(samples)
            measurement_id = str(measurement.get("measurement_id", "?"))
            sc = measurement.get("scenario", "?")

            # Routing
            route_str = ""
            if is_router:
                from collections import Counter

                route_counts = Counter(s.get("routing_decision", "?") for s in samples)
                failed = sum(1 for s in samples if s.get("routing_failed"))
                parts = [f"{k}: {v}" for k, v in sorted(route_counts.items())]
                route_str = ", ".join(parts)
                if failed:
                    route_str += f" ({failed} err)"

            # Accuracy
            accuracy, _, _ = _get_accuracy_stats(samples)
            acc = f"{accuracy * 100:.1f}%" if accuracy is not None else "-"

            # Energie (aus Messungs-Dict, nicht aus Samples)
            dyn_e = measurement.get("measurement_dynamic_energy_joules")
            dyn_e_wh = measurement.get("measurement_dynamic_energy_watthours")
            if dyn_e is not None:
                wh_str = f" / {dyn_e_wh * 1000:.2f} mWh" if dyn_e_wh is not None else ""
                energy_str = f"{dyn_e:.1f} J{wh_str}"
            else:
                energy_str = "-"

            ept_val = measurement.get("measurement_millijoules_per_output_token")
            ept_str = f"{ept_val:.1f}" if ept_val is not None else "-"

            eq_val = compute_measurement_eq_score(measurement)
            eq_str = f"{eq_val:.2f}" if eq_val is not None else "-"

            tps_val = measurement.get("measurement_tokens_per_second")
            tps_str = f"{tps_val:.1f}" if tps_val is not None else "-"

            if is_router:
                overview.add_row(
                    measurement_id,
                    sc,
                    str(n),
                    route_str,
                    acc,
                    energy_str,
                    ept_str,
                    eq_str,
                    tps_str,
                )
            else:
                overview.add_row(
                    measurement_id,
                    sc,
                    str(n),
                    acc,
                    energy_str,
                    ept_str,
                    eq_str,
                    tps_str,
                )

        log.write(overview)
        log.write("")

        # Detail pro Messung
        for measurement in measurements:
            samples = measurement.get("samples", [])
            n = len(samples)
            sc = measurement.get("scenario", "?")
            measurement_id = measurement.get("measurement_id", "?")

            detail = Table(
                title=f"{sc} — ID {measurement_id}" if multiple_entries else sc,
                expand=True,
                padding=(0, 1),
            )
            detail.add_column("Metrik", style="dim")
            detail.add_column("Wert", style="bold")

            model_name = measurement.get("model") or (
                samples[0].get("model", "?") if samples else "?"
            )
            detail.add_row("Modell", model_name)
            detail.add_row("Benchmark", measurement.get("benchmark", "?"))
            detail.add_row("Samples", str(n))

            # Routing (Router-Experiment)
            if is_router and samples:
                from collections import Counter

                route_counts = Counter(s.get("routing_decision", "?") for s in samples)
                failed = sum(1 for s in samples if s.get("routing_failed"))
                parts = []
                for k, v in sorted(route_counts.items()):
                    pct = v * 100 // n if n else 0
                    parts.append(f"{k}: {v} ({pct}%)")
                detail.add_row("Routing-Verteilung", ", ".join(parts))
                if failed:
                    detail.add_row("Routing-Fehler", str(failed))

            # Accuracy
            accuracy, correct_count, evaluated_count = _get_accuracy_stats(samples)
            if accuracy is None:
                detail.add_row("Accuracy", "-")
            else:
                accuracy_text = (
                    f"{correct_count}/{evaluated_count} ({accuracy * 100:.1f}%)"
                )
                if evaluated_count < n:
                    accuracy_text += f" · {evaluated_count}/{n} ausgewertet"
                detail.add_row("Accuracy", accuracy_text)

            # Energiemetriken (direkt aus Messungs-Dict)
            measurement_mj = measurement.get("measurement_millijoules_per_output_token")
            if measurement_mj is not None:
                detail.add_row("Energie/Token (mJ)", f"{measurement_mj:.1f}")

            # EQ-Score
            eq = compute_measurement_eq_score(measurement)
            if eq is not None:
                detail.add_row(
                    f"EQ-Score (Qualitaet/{EQ_SCORE_ENERGY_UNIT})",
                    f"{eq:.2f}",
                )

            measurement_tps = measurement.get("measurement_tokens_per_second")
            if measurement_tps is not None:
                detail.add_row("Tokens/s", f"{measurement_tps:.1f}")

            measurement_in = measurement.get("measurement_total_input_tokens")
            measurement_out = measurement.get("measurement_total_output_tokens")
            if measurement_in is not None or measurement_out is not None:
                detail.add_row(
                    "Tokens (in/out/ges.)",
                    f"{measurement_in or 0} / {measurement_out or 0} / {(measurement_in or 0) + (measurement_out or 0)}",
                )

            measurement_dyn_e = measurement.get("measurement_dynamic_energy_joules")
            measurement_dyn_e_wh = measurement.get(
                "measurement_dynamic_energy_watthours"
            )
            measurement_total_e_wh = measurement.get("measurement_energy_watthours")
            measurement_dyn_p = measurement.get("measurement_dynamic_power_watts")
            measurement_dur = measurement.get("measurement_duration_seconds")
            if measurement_dyn_e is not None:
                wh_str = (
                    f" / {measurement_dyn_e_wh * 1000:.3f} mWh"
                    if measurement_dyn_e_wh is not None
                    else ""
                )
                detail.add_row(
                    "Dynamische Energie", f"{measurement_dyn_e:.2f} J{wh_str}"
                )
            if measurement_total_e_wh is not None:
                detail.add_row(
                    "Gesamtenergie (inkl. Idle)",
                    f"{measurement_total_e_wh * 1000:.3f} mWh",
                )
            if measurement_dyn_p is not None:
                detail.add_row("Dynamische Leistung (W)", f"{measurement_dyn_p:.1f}")
            if measurement_dur is not None:
                detail.add_row("Dauer (s)", f"{measurement_dur:.1f}")

            measurement_avg_p = measurement.get("measurement_avg_power_watts")
            if measurement_avg_p is not None:
                detail.add_row("Gesamtleistung (W)", f"{measurement_avg_p:.1f}")

            measurement_idle_p = measurement.get("measurement_idle_power_watts")
            if measurement_idle_p is not None:
                detail.add_row("Idle-Leistung (W)", f"{measurement_idle_p:.1f}")

            measurement_min_p = measurement.get("measurement_min_power_watts")
            measurement_max_p = measurement.get("measurement_max_power_watts")
            if measurement_min_p is not None and measurement_max_p is not None:
                detail.add_row(
                    "Min/Max Leistung (W)",
                    f"{measurement_min_p:.1f} / {measurement_max_p:.1f}",
                )

            log.write(detail)
            log.write("")

        # Rohdaten-Info
        log.write(Text(f"Datei: {self._path}", style="dim"))
        log.write(Text(f"Gesamt: {total_samples} Samples", style="dim"))
        # Hinweis auf detaillierte Plots im Experiment-Verzeichnis
        plots_dir = self._path.parent / "plots"
        if plots_dir.is_dir() and any(plots_dir.iterdir()):
            log.write(Text(f"Detaillierte Plots: {plots_dir}", style="dim cyan"))

        self._summary_text = self._build_plain_summary(measurements)

        # Terminal-Plots mit plotext befüllen
        self._populate_plots(data, measurements)

    def _try_render_average_sidecar(self, log: RichLog) -> bool:
        """Rendert Average-Details aus kleinen Sidecar-Artefakten."""
        result_dir = _normalize_result_dir(self._path)
        average_path = result_dir / "average_summary.json"
        summary_path = result_dir / "measurement_summary.csv"
        config_path = result_dir / "config.json"

        if not average_path.exists() or not summary_path.exists():
            return False

        try:
            average_payload = _read_json_file(average_path)
            rows = _read_csv_rows(summary_path)
            config = _read_json_file(config_path) if config_path.exists() else {}
        except Exception:
            return False

        if not average_payload or not rows:
            return False

        average = average_payload.get("average", {})
        details = average_payload.get("average_details", {})
        sources = average_payload.get("source_experiments", [])
        averaging = average_payload.get("averaging", {})

        self._display_name = config.get("experiment_alias") or result_dir.name
        self.query_one("#step", Static).update(
            _compose_result_heading(self._display_name, self._path)
        )

        summary = Table(
            title=f"Durchschnitt über {average.get('run_count', len(rows))} Messungen",
            expand=True,
            padding=(0, 1),
        )
        summary.add_column("Metrik", style="dim")
        summary.add_column("Wert", style="bold")
        summary.add_row("Samples gesamt", str(average.get("total_samples", "-")))

        mean_accuracy = _coerce_float(average.get("mean_accuracy"))
        if mean_accuracy is not None:
            formatted = details.get("mean_accuracy_percent_formatted")
            summary.add_row(
                "Accuracy",
                f"{mean_accuracy:.4f} ({formatted or f'{mean_accuracy * 100:.3f}%'})",
            )

        weighted_accuracy = _coerce_float(details.get("weighted_accuracy"))
        if weighted_accuracy is not None:
            formatted = details.get("weighted_accuracy_percent_formatted")
            summary.add_row(
                "Accuracy gewichtet",
                f"{weighted_accuracy:.4f} ({formatted or f'{weighted_accuracy * 100:.3f}%'})",
            )

        mean_energy = _coerce_float(average.get("mean_dynamic_energy_joules"))
        mean_energy_wh = _coerce_float(average.get("mean_dynamic_energy_watthours"))
        if mean_energy is not None:
            wh_str = (
                f" / {mean_energy_wh * 1000:.3f} mWh"
                if mean_energy_wh is not None
                else ""
            )
            summary.add_row("Dynamische Energie", f"{mean_energy:.2f} J{wh_str}")

        mean_mj = _coerce_float(average.get("mean_millijoules_per_output_token"))
        if mean_mj is not None:
            summary.add_row("Energie/Token (mJ)", f"{mean_mj:.1f}")

        mean_eq = _coerce_float(average.get("mean_eq_score"))
        if mean_eq is not None:
            summary.add_row(
                f"EQ-Score (Qualitaet/{EQ_SCORE_ENERGY_UNIT})",
                f"{mean_eq:.3f}",
            )

        mean_tps = _coerce_float(average.get("mean_tokens_per_second"))
        if mean_tps is not None:
            summary.add_row("Tokens/s", f"{mean_tps:.1f}")

        mean_duration = _coerce_float(average.get("mean_duration_seconds"))
        if mean_duration is not None:
            summary.add_row("Dauer (s)", f"{mean_duration:.1f}")

        created_at = averaging.get("created_at")
        if created_at:
            summary.add_row("Erstellt", str(created_at))

        log.write(summary)
        log.write("")

        overview = Table(title="Einzelmessungen", expand=True, padding=(0, 1))
        overview.add_column("ID", style="bold")
        overview.add_column("Szenario", style="bold")
        overview.add_column("Samples", justify="right")
        overview.add_column("Accuracy", justify="right")
        overview.add_column("dyn. Energie", justify="right")
        overview.add_column("mJ/Token", justify="right")
        overview.add_column(f"EQ (Qual./{EQ_SCORE_ENERGY_UNIT})", justify="right")
        overview.add_column("Tok/s", justify="right")

        for row in rows:
            measurement_id = row.get("measurement_id", "?")
            scenario = row.get("scenario", "?")
            sample_count = row.get("measurement_num_samples") or "-"

            accuracy = _coerce_float(row.get("measurement_accuracy"))
            accuracy_str = f"{accuracy * 100:.1f}%" if accuracy is not None else "-"

            dyn_energy = _coerce_float(row.get("measurement_dynamic_energy_joules"))
            dyn_wh = _coerce_float(row.get("measurement_dynamic_energy_watthours"))
            if dyn_energy is not None:
                wh_str = f" / {dyn_wh * 1000:.2f} mWh" if dyn_wh is not None else ""
                energy_str = f"{dyn_energy:.1f} J{wh_str}"
            else:
                energy_str = "-"

            mj = _coerce_float(row.get("measurement_millijoules_per_output_token"))
            mj_str = f"{mj:.1f}" if mj is not None else "-"

            eq = _coerce_float(row.get("measurement_eq_score"))
            eq_str = f"{eq:.2f}" if eq is not None else "-"

            tps = _coerce_float(row.get("measurement_tokens_per_second"))
            tps_str = f"{tps:.1f}" if tps is not None else "-"

            overview.add_row(
                str(measurement_id),
                str(scenario),
                str(sample_count),
                accuracy_str,
                energy_str,
                mj_str,
                eq_str,
                tps_str,
            )

        log.write(overview)

        if sources:
            log.write("")
            source_table = Table(title="Quell-Experimente", expand=True, padding=(0, 1))
            source_table.add_column("#", justify="right")
            source_table.add_column("Name", style="bold")
            source_table.add_column("Samples", justify="right")
            source_table.add_column("Pfad", style="dim")
            for source in sources:
                source_table.add_row(
                    str(source.get("source_index", "?")),
                    str(source.get("display_name", "?")),
                    str(source.get("sample_count", "?")),
                    str(source.get("path", "?")),
                )
            log.write(source_table)

        log.write("")
        log.write(Text(f"Datei: {self._path}", style="dim"))
        log.write(Text(f"Average-Summary: {average_path}", style="dim cyan"))

        self._summary_text = self._build_average_sidecar_plain_summary(
            average,
            details,
            rows,
            sources,
        )

        # Plotext-Widgets ausblenden: Sidecar-Fast-Path bleibt bewusst leichtgewichtig.
        for widget_id in ("#plot-power", "#plot-accuracy"):
            try:
                self.query_one(widget_id).display = False
            except Exception:
                pass

        return True

    def _build_average_sidecar_plain_summary(
        self,
        average: dict,
        details: dict,
        rows: list[dict],
        sources: list[dict],
    ) -> str:
        lines = [f"Ergebnisse: {self._display_name}", "=" * 60, ""]
        lines.append(
            f"Durchschnitt über {average.get('run_count', len(rows))} Messungen"
        )
        lines.append(f"  Samples gesamt: {average.get('total_samples', '-')}")

        mean_accuracy = _coerce_float(average.get("mean_accuracy"))
        if mean_accuracy is not None:
            formatted = details.get("mean_accuracy_percent_formatted")
            lines.append(
                f"  Accuracy: {mean_accuracy:.4f} ({formatted or f'{mean_accuracy * 100:.3f}%'})"
            )

        weighted_accuracy = _coerce_float(details.get("weighted_accuracy"))
        if weighted_accuracy is not None:
            formatted = details.get("weighted_accuracy_percent_formatted")
            lines.append(
                f"  Accuracy gewichtet: {weighted_accuracy:.4f} ({formatted or f'{weighted_accuracy * 100:.3f}%'})"
            )

        mean_energy = _coerce_float(average.get("mean_dynamic_energy_joules"))
        if mean_energy is not None:
            lines.append(f"  Dynamische Energie: {mean_energy:.2f} J")

        mean_mj = _coerce_float(average.get("mean_millijoules_per_output_token"))
        if mean_mj is not None:
            lines.append(f"  Energie/Token: {mean_mj:.1f} mJ")

        mean_eq = _coerce_float(average.get("mean_eq_score"))
        if mean_eq is not None:
            lines.append(f"  EQ-Score: {mean_eq:.3f} Qualitaet/{EQ_SCORE_ENERGY_UNIT}")

        if sources:
            lines.append("")
            lines.append("Quell-Experimente:")
            for source in sources:
                lines.append(
                    f"  - {source.get('display_name', '?')} ({source.get('sample_count', '?')} Samples)"
                )

        return "\n".join(lines)

    def _populate_plots(self, data: dict, measurements: list[dict]) -> None:
        """Befüllt die plotext-Widgets mit Daten (Power + Accuracy)."""
        try:
            from textual_plotext import PlotextPlot
        except ImportError:
            return

        # Plot 1: Power über Zeit (aus Experiment-Verzeichnis oder JSON)
        try:
            power_plot = self.query_one("#plot-power", PlotextPlot)
            # Versuche power_samples.csv aus dem Experiment-Verzeichnis zu laden
            parent_dir = self._path.parent
            power_csv = (
                parent_dir / "power_samples.csv" if parent_dir.is_dir() else None
            )
            has_power = False
            if power_csv and power_csv.exists():
                from src.utils.data import load_power_samples

                power_df = load_power_samples(power_csv)
                if not power_df.empty:
                    has_power = True
                    plt = power_plot.plt
                    plt.clear_data()
                    plt.clear_figure()
                    if "measurement_id" in power_df.columns:
                        for measurement_id, group in power_df.groupby("measurement_id"):
                            plt.plot(
                                group["time_s"].tolist(),
                                group["power_watts"].tolist(),
                                label=f"ID {measurement_id}",
                            )
                    else:
                        plt.plot(
                            power_df["time_s"].tolist(),
                            power_df["power_watts"].tolist(),
                            label="Messung",
                        )
                    # P_idle als Referenz
                    idle_p = (
                        measurements[0].get("measurement_idle_power_watts", 0)
                        if measurements
                        else 0
                    )
                    if idle_p > 0:
                        plt.hline(idle_p, color="gray")
                    plt.title("Leistung (W)")
                    plt.xlabel("Zeit (s)")
                    plt.ylabel("Watt")
                    power_plot.refresh()
            if not has_power:
                power_plot.display = False
        except Exception:
            try:
                self.query_one("#plot-power").display = False
            except Exception:
                pass

        # Plot 2: Accuracy pro Messdatensatz (Balkendiagramm)
        try:
            acc_plot = self.query_one("#plot-accuracy", PlotextPlot)
            labels = []
            accuracies = []
            multiple_measurements = len(measurements) > 1
            for measurement in measurements:
                samples = measurement.get("samples", [])
                n = len(samples)
                if n == 0:
                    continue
                accuracy, _, _ = _get_accuracy_stats(samples)
                if accuracy is None:
                    continue
                if multiple_measurements:
                    labels.append(f"ID {measurement.get('measurement_id', '?')}")
                else:
                    labels.append("Messung")
                accuracies.append(accuracy * 100)
            if labels:
                plt = acc_plot.plt
                plt.clear_data()
                plt.clear_figure()
                plt.bar(labels, accuracies)
                plt.title("Accuracy (%)")
                plt.ylabel("%")
                acc_plot.refresh()
            else:
                acc_plot.display = False
        except Exception:
            try:
                self.query_one("#plot-accuracy").display = False
            except Exception:
                pass

    def _build_plain_summary(self, measurements: list[dict]) -> str:
        """Baut Klartext-Zusammenfassung für Clipboard."""
        lines = [f"Ergebnisse: {self._display_name}", "=" * 60]
        multiple_entries = len(measurements) > 1
        if multiple_entries:
            average_metrics = _build_average_metrics(measurements)
            lines.append("")
            lines.append(f"Durchschnitt über {len(measurements)} Messungen")
            lines.append(f"  Samples gesamt: {average_metrics['total_samples']}")
            mean_accuracy = average_metrics.get("mean_accuracy")
            if mean_accuracy is not None:
                lines.append(
                    f"  Accuracy: {mean_accuracy:.4f} ({mean_accuracy * 100:.3f}%)"
                )
            mean_energy = average_metrics.get("mean_dynamic_energy_joules")
            mean_energy_wh = average_metrics.get("mean_dynamic_energy_watthours")
            if mean_energy is not None:
                wh_str = (
                    f" / {mean_energy_wh * 1000:.3f} mWh"
                    if mean_energy_wh is not None
                    else ""
                )
                lines.append(f"  Dynamische Energie: {mean_energy:.2f} J{wh_str}")
            mean_mj = average_metrics.get("mean_millijoules_per_output_token")
            if mean_mj is not None:
                lines.append(f"  Energie/Token: {mean_mj:.1f} mJ")
            mean_eq = average_metrics.get("mean_eq_score")
            if mean_eq is not None:
                lines.append(
                    f"  EQ-Score: {mean_eq:.3f} Qualitaet/{EQ_SCORE_ENERGY_UNIT}"
                )
            mean_tps = average_metrics.get("mean_tokens_per_second")
            if mean_tps is not None:
                lines.append(f"  Tokens/s: {mean_tps:.1f}")
            mean_duration = average_metrics.get("mean_duration_seconds")
            if mean_duration is not None:
                lines.append(f"  Dauer: {mean_duration:.1f} s")
        for measurement in measurements:
            samples = measurement.get("samples", [])
            n = len(samples)
            sc = measurement.get("scenario", "?")
            measurement_id = measurement.get("measurement_id", "?")
            model_name = measurement.get("model") or (
                samples[0].get("model", "?") if samples else "?"
            )
            if multiple_entries:
                lines.append(f"\n{sc} - ID {measurement_id} ({n} Samples)")
            else:
                lines.append(f"\n{sc} ({n} Samples)")
            lines.append(f"  Modell: {model_name}")
            lines.append(f"  Benchmark: {measurement.get('benchmark', '?')}")
            # Routing
            if any(s.get("routing_decision") for s in samples):
                from collections import Counter

                route_counts = Counter(s.get("routing_decision", "?") for s in samples)
                failed = sum(1 for s in samples if s.get("routing_failed"))
                parts = [f"{k}: {v}" for k, v in sorted(route_counts.items())]
                lines.append(f"  Routing: {', '.join(parts)}")
                if failed:
                    lines.append(f"  Routing-Fehler: {failed}")
            accuracy, correct_count, evaluated_count = _get_accuracy_stats(samples)
            if accuracy is None:
                lines.append("  Accuracy: -")
            else:
                accuracy_text = f"  Accuracy: {correct_count}/{evaluated_count} ({accuracy * 100:.1f}%)"
                if evaluated_count < n:
                    accuracy_text += f" · {evaluated_count}/{n} ausgewertet"
                lines.append(accuracy_text)
            measurement_mj = measurement.get("measurement_millijoules_per_output_token")
            if measurement_mj is not None:
                lines.append(f"  Energie/Token: {measurement_mj:.1f} mJ")
            eq = compute_measurement_eq_score(measurement)
            if eq is not None:
                lines.append(f"  EQ-Score: {eq:.2f} Qualitaet/{EQ_SCORE_ENERGY_UNIT}")
            measurement_tps = measurement.get("measurement_tokens_per_second")
            if measurement_tps is not None:
                lines.append(f"  Tokens/s: {measurement_tps:.1f}")
            measurement_in = measurement.get("measurement_total_input_tokens")
            measurement_out = measurement.get("measurement_total_output_tokens")
            if measurement_in is not None or measurement_out is not None:
                lines.append(
                    f"  Tokens: {measurement_in or 0} ein / {measurement_out or 0} aus / {(measurement_in or 0) + (measurement_out or 0)} ges."
                )
            measurement_dyn_e = measurement.get("measurement_dynamic_energy_joules")
            measurement_dyn_e_wh = measurement.get(
                "measurement_dynamic_energy_watthours"
            )
            measurement_total_e_wh = measurement.get("measurement_energy_watthours")
            measurement_dyn_p = measurement.get("measurement_dynamic_power_watts")
            if measurement_dyn_e is not None:
                wh_str = (
                    f" / {measurement_dyn_e_wh * 1000:.3f} mWh"
                    if measurement_dyn_e_wh is not None
                    else ""
                )
                lines.append(f"  Dynamische Energie: {measurement_dyn_e:.2f} J{wh_str}")
            if measurement_total_e_wh is not None:
                lines.append(
                    f"  Gesamtenergie (inkl. Idle): {measurement_total_e_wh * 1000:.3f} mWh"
                )
            if measurement_dyn_p is not None:
                lines.append(f"  Dynamische Leistung: {measurement_dyn_p:.1f} W")
            measurement_avg_p = measurement.get("measurement_avg_power_watts")
            if measurement_avg_p is not None:
                lines.append(f"  Gesamtleistung: {measurement_avg_p:.1f} W")
            measurement_idle_p = measurement.get("measurement_idle_power_watts")
            if measurement_idle_p is not None:
                lines.append(f"  Idle-Leistung: {measurement_idle_p:.1f} W")
            measurement_min_p = measurement.get("measurement_min_power_watts")
            measurement_max_p = measurement.get("measurement_max_power_watts")
            if measurement_min_p is not None and measurement_max_p is not None:
                lines.append(
                    f"  Min/Max Leistung: {measurement_min_p:.1f} / {measurement_max_p:.1f} W"
                )
            measurement_dur = measurement.get("measurement_duration_seconds")
            if measurement_dur is not None:
                lines.append(f"  Dauer: {measurement_dur:.1f} s")
        return "\n".join(lines)

    @on(Button.Pressed, "#btn-copy")
    def on_copy(self) -> None:
        text = getattr(self, "_summary_text", "")
        if not text:
            self.notify("Keine Daten vorhanden.", severity="warning")
            return
        if _copy_to_clipboard(text):
            self.notify("Zusammenfassung kopiert.")
        else:
            tmp = Path("/tmp/llm_routing_results.txt")
            tmp.write_text(text, encoding="utf-8")
            self.notify(f"Gespeichert: {tmp}", severity="warning", timeout=8)

    @on(Button.Pressed, "#btn-reevaluate")
    def on_reevaluate(self) -> None:
        cmd = [sys.executable, "-m", "src.evaluation.reevaluate", str(self._path)]
        self.app.push_screen(
            ProcessScreen(cmd=cmd, title="Neu-Auswertung", cwd=str(PROJECT_ROOT))
        )

    @work(thread=True)
    def _do_replot(self) -> None:
        try:
            from src.plotting import generate_experiment_plots

            generate_experiment_plots(self._path.parent)
            self.app.call_from_thread(
                self.notify,
                "Plots wurden erfolgreich neu generiert.",
                severity="information",
            )
        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Fehler beim Erstellen der Plots: {e}", severity="error"
            )

    @on(Button.Pressed, "#btn-replot")
    def on_replot(self) -> None:
        self.notify("Erstelle Plots, bitte warten...", severity="information")
        self._do_replot()

    @on(Button.Pressed, "#btn-alias")
    def on_alias(self) -> None:
        self.app.push_screen(AliasEditScreen(self._path))

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─── Compare-Screen ──────────────────────────────────────────────────────


class CompareScreen(Screen):
    """Experimente vergleichen: Multi-Select und Vergleichsplots generieren."""

    BINDINGS = [
        ("escape", "go_back", "Zurück"),
        ("space", "toggle_selected", "Auswählen"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selected_result_paths: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Experimente vergleichen[/]\n"
                "[dim]↑↓ navigieren · Space oder Enter an-/abwählen · Esc Zurück[/]\n"
                "[dim]Mindestens 2 Experimente auswählen oder bestehende Comparisons öffnen.[/]",
                id="step",
            )
            self._file_list = _load_result_files()
            if len(self._file_list) < 2:
                yield Static(
                    "[red bold]Mindestens 2 Ergebnis-Dateien benötigt.[/]\n"
                    "[dim]Führe zuerst mehrere Experimente durch.[/]",
                    id="no-results",
                )
            else:
                yield Tree("Experimente", id="compare-tree")
                yield Static("0 Experimente ausgewählt", id="compare-status")
            with Horizontal(classes="button-row"):
                yield Button(
                    "Vergleich starten",
                    variant="success",
                    id="btn-compare",
                )
                yield Button(
                    "Durchschnitt bilden",
                    variant="primary",
                    id="btn-average",
                )
                yield Button(
                    "Comparisons öffnen",
                    variant="default",
                    id="btn-open-comparisons",
                )
        yield Footer()

    def on_mount(self) -> None:
        if self._file_list:
            tree = self.query_one("#compare-tree", Tree)
            _populate_result_tree(tree, self._file_list, self._selected_result_paths)
            self._update_status()

    def _update_status(self) -> None:
        try:
            self.query_one("#compare-status", Static).update(
                f"{len(self._selected_result_paths)} Experimente ausgewählt"
            )
        except Exception:
            pass

    def action_toggle_selected(self) -> None:
        try:
            tree = self.query_one("#compare-tree", Tree)
        except Exception:
            return
        self._toggle_tree_node(tree.cursor_node)

    @on(Tree.NodeSelected, "#compare-tree")
    def on_tree_selected(self, event: Tree.NodeSelected) -> None:
        self._toggle_tree_node(event.node)

    def _toggle_tree_node(self, node) -> None:
        data = getattr(node, "data", None)
        if not isinstance(data, dict) or "path" not in data:
            return

        path_str = data["path"]
        if path_str in self._selected_result_paths:
            self._selected_result_paths.remove(path_str)
        else:
            self._selected_result_paths.append(path_str)

        node.label = _format_result_leaf(
            Path(path_str),
            data.get("info", {}),
            path_str in self._selected_result_paths,
        )
        self._update_status()

    @on(Button.Pressed, "#btn-compare")
    def on_compare(self) -> None:
        selected = list(self._selected_result_paths)
        if len(selected) < 2:
            self.notify(
                "Mindestens 2 Experimente auswählen.",
                severity="warning",
            )
            return

        # Experiment-Verzeichnisse aus den Pfaden bestimmen
        exp_dirs = []
        for file_path_str in selected:
            p = Path(file_path_str)
            exp_dirs.append(str(p.parent))

        cmd = [
            sys.executable,
            "-m",
            "experiments.compare",
            *exp_dirs,
        ]
        self.app.push_screen(ProcessScreen(cmd=cmd, title="Experiment-Vergleich"))

    @on(Button.Pressed, "#btn-average")
    def on_average(self) -> None:
        selected = list(self._selected_result_paths)
        if len(selected) < 2:
            self.notify(
                "Mindestens 2 Experimente auswählen.",
                severity="warning",
            )
            return

        exp_dirs = []
        for file_path_str in selected:
            p = Path(file_path_str)
            exp_dirs.append(str(p.parent))

        cmd = [
            sys.executable,
            "-m",
            "experiments.average",
            *exp_dirs,
        ]
        self.app.push_screen(ProcessScreen(cmd=cmd, title="Durchschnitts-Run erzeugen"))

    @on(Button.Pressed, "#btn-open-comparisons")
    def on_open_comparisons(self) -> None:
        self.app.push_screen(ComparisonScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ComparisonScreen(Screen):
    """Bestehende Comparison-Verzeichnisse browsen."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Comparisons[/]\n"
                "[dim]↑↓ navigieren · Enter für Details · Esc Zurück[/]",
                id="step",
            )
            self._file_list = _load_comparison_files()
            if not self._file_list:
                yield Static(
                    "[red bold]Keine Comparison-Ordner gefunden.[/]\n"
                    "[dim]Erzeuge zuerst einen Vergleich oder prüfe das Manifest comparison_inputs.json.[/]",
                    id="no-comparisons",
                )
            else:
                options = []
                for manifest_path, info in self._file_list:
                    from datetime import datetime

                    display_name = info.get("display_name", manifest_path.parent.name)
                    folder_name = info.get(
                        "folder_name", _format_comparison_relative_dir(manifest_path)
                    )
                    mtime = info.get("mtime")
                    date_str = (
                        datetime.fromtimestamp(mtime).strftime("%d.%m. %H:%M")
                        if mtime
                        else ""
                    )
                    exp_count = info.get("experiment_count", 0)
                    plot_count = len(info.get("plot_files", []))
                    benchmarks = ", ".join(info.get("benchmarks", [])) or "?"
                    experiment_types = (
                        ", ".join(info.get("experiment_types", [])) or "?"
                    )
                    missing_count = info.get("missing_count", 0)
                    missing_str = f" · {missing_count} fehlend" if missing_count else ""
                    folder_part = (
                        f" · Ordner {folder_name}"
                        if display_name != folder_name
                        else ""
                    )
                    options.append(
                        Option(
                            Text.assemble(
                                (display_name, "bold"),
                                "\n",
                                (
                                    f"  {exp_count} Inputs · {benchmarks} · Typen {experiment_types} · {plot_count} PNGs · {date_str}{missing_str}{folder_part}",
                                    "dim",
                                ),
                            )
                        )
                    )
                yield OptionList(*options, id="comparison-list")
        yield Footer()

    @on(OptionList.OptionSelected, "#comparison-list")
    def on_select(self, event: OptionList.OptionSelected) -> None:
        if not self._file_list:
            return
        manifest_path, _ = self._file_list[event.option_index]
        self.app.push_screen(ComparisonDetailScreen(manifest_path))

    def action_go_back(self) -> None:
        self.app.pop_screen()


class ComparisonDetailScreen(Screen):
    """Zeigt Details eines bestehenden Comparison-Verzeichnisses."""

    BINDINGS = [("escape", "go_back", "Zurück")]

    def __init__(self, manifest_path: Path) -> None:
        super().__init__()
        self._manifest_path = manifest_path
        self._comparison_dir = _normalize_comparison_dir(manifest_path)
        self._display_name = self._comparison_dir.name

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                _compose_comparison_heading(
                    self._display_name,
                    self._comparison_dir,
                ),
                id="step",
            )
            yield RichLog(highlight=True, markup=True, id="detail-log")
            with Horizontal(classes="button-row"):
                yield Button("Kopieren", variant="primary", id="btn-copy")
                yield Button("Plots neu", variant="default", id="btn-replot")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#detail-log", RichLog)
        _, info = _build_comparison_info(
            self._manifest_path,
            _build_result_lookup(),
        )

        self._display_name = info.get("display_name", self._comparison_dir.name)
        self.query_one("#step", Static).update(
            _compose_comparison_heading(self._display_name, self._comparison_dir)
        )

        summary = Table(title="Comparison", expand=True, padding=(0, 1))
        summary.add_column("Metrik", style="dim")
        summary.add_column("Wert", style="bold")
        summary.add_row(
            "Ordner",
            info.get(
                "folder_name", _format_comparison_relative_dir(self._comparison_dir)
            ),
        )
        summary.add_row("Manifest", self._manifest_path.name)
        summary.add_row("Input-Experimente", str(info.get("experiment_count", 0)))

        benchmarks = info.get("benchmarks", [])
        if benchmarks:
            summary.add_row("Benchmarks", ", ".join(benchmarks))

        experiment_types = info.get("experiment_types", [])
        if experiment_types:
            summary.add_row("Typen", ", ".join(experiment_types))

        created_at = info.get("created_at")
        if created_at:
            summary.add_row("Erstellt", str(created_at))

        plot_files = info.get("plot_files", [])
        summary.add_row(
            "Plots",
            (
                f"{len(plot_files)} PNG-Dateien"
                if plot_files
                else "Keine PNG-Dateien gefunden"
            ),
        )

        missing_count = info.get("missing_count", 0)
        if missing_count:
            summary.add_row("Fehlende Inputs", str(missing_count))

        filters = info.get("filters", {})
        filter_parts = []
        if filters.get("types"):
            filter_parts.append(f"Typen: {', '.join(filters['types'])}")
        if filters.get("benchmarks"):
            filter_parts.append(f"Benchmarks: {', '.join(filters['benchmarks'])}")
        if filters.get("group") and filters["group"] != "all":
            filter_parts.append(f"Gruppe: {filters['group']}")
        if filter_parts:
            summary.add_row("Manifest-Filter", " | ".join(filter_parts))

        if info.get("manifest_error"):
            summary.add_row("Manifest-Fehler", info["manifest_error"])

        log.write(summary)

        source_infos = info.get("source_infos", [])
        if source_infos:
            log.write("")
            sources = Table(title="Input-Experimente", expand=True, padding=(0, 1))
            sources.add_column("Name", style="bold")
            sources.add_column("Typ")
            sources.add_column("Benchmark")
            sources.add_column("Status")
            sources.add_column("Pfad", style="dim")
            for source in source_infos:
                status = Text("OK", style="green")
                if not source["exists"]:
                    status = Text("fehlt", style="bold red")
                sources.add_row(
                    source["display_name"],
                    source["experiment_type"],
                    source["benchmark"],
                    status,
                    source["folder_name"],
                )
            log.write(sources)
        else:
            log.write("")
            log.write(
                Text(
                    "Keine Input-Experimente im Manifest gefunden.",
                    style="bold yellow",
                )
            )

        if plot_files:
            log.write("")
            plots = Table(title="Plot-Dateien", expand=True, padding=(0, 1))
            plots.add_column("Datei", style="bold")
            for plot_file in plot_files:
                plots.add_row(plot_file)
            log.write(plots)

        log.write("")
        log.write(Text(f"Comparison-Ordner: {self._comparison_dir}", style="dim"))
        log.write(Text(f"Manifest: {self._manifest_path}", style="dim"))

        self._summary_text = self._build_plain_summary(info)

    def _build_plain_summary(self, info: dict) -> str:
        """Baut eine Klartext-Zusammenfassung für Clipboard."""
        lines = [f"Comparison: {self._display_name}", "=" * 60]
        lines.append(
            f"Ordner: {info.get('folder_name', _format_comparison_relative_dir(self._comparison_dir))}"
        )
        lines.append(f"Manifest: {self._manifest_path.name}")
        lines.append(f"Input-Experimente: {info.get('experiment_count', 0)}")

        benchmarks = info.get("benchmarks", [])
        if benchmarks:
            lines.append(f"Benchmarks: {', '.join(benchmarks)}")

        experiment_types = info.get("experiment_types", [])
        if experiment_types:
            lines.append(f"Typen: {', '.join(experiment_types)}")

        created_at = info.get("created_at")
        if created_at:
            lines.append(f"Erstellt: {created_at}")

        filters = info.get("filters", {})
        if filters:
            lines.append(f"Manifest-Filter: {filters}")

        if info.get("manifest_error"):
            lines.append(f"Manifest-Fehler: {info['manifest_error']}")

        plot_files = info.get("plot_files", [])
        lines.append(
            f"Plots: {len(plot_files)} PNG-Dateien"
            if plot_files
            else "Plots: Keine PNG-Dateien gefunden"
        )

        source_infos = info.get("source_infos", [])
        if source_infos:
            lines.append("")
            lines.append("Input-Experimente:")
            for source in source_infos:
                status = "OK" if source["exists"] else "FEHLT"
                lines.append(
                    f"  - {source['display_name']} [{source['experiment_type']}] {source['benchmark']} · {status} · {source['folder_name']}"
                )

        if plot_files:
            lines.append("")
            lines.append("Plot-Dateien:")
            for plot_file in plot_files:
                lines.append(f"  - {plot_file}")

        return "\n".join(lines)

    @on(Button.Pressed, "#btn-copy")
    def on_copy(self) -> None:
        text = getattr(self, "_summary_text", "")
        if not text:
            self.notify("Keine Daten vorhanden.", severity="warning")
            return
        if _copy_to_clipboard(text):
            self.notify("Zusammenfassung kopiert.")
        else:
            tmp = Path("/tmp/llm_routing_comparison.txt")
            tmp.write_text(text, encoding="utf-8")
            self.notify(f"Gespeichert: {tmp}", severity="warning", timeout=8)

    @on(Button.Pressed, "#btn-replot")
    def on_replot(self) -> None:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "regenerate_plots.py"),
            "comparison",
            "--refresh-existing",
            "--comparison-dirs",
            str(self._comparison_dir),
        ]
        self.app.push_screen(
            ProcessScreen(
                cmd=cmd,
                title="Comparison-Plots neu",
                cwd=str(PROJECT_ROOT),
            )
        )

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─── Download-Screen ───────────────────────────────────────────────────────


class DownloadScreen(Screen):
    """Modell-Verwaltung: herunterladen und löschen."""

    BINDINGS = [("escape", "app.pop_screen", "Zurück")]

    def __init__(self) -> None:
        super().__init__()
        self._models_info: list[dict] = (
            []
        )  # {display_id, hf_name, vram_str, fits, downloaded}
        self._selected_idx: int | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Modell-Verwaltung[/]\n"
                "[dim]↑↓ navigieren · Enter auswählen · Esc zurück[/]",
                id="step",
            )
            yield OptionList(id="model-list")
            with Horizontal(id="model-buttons"):
                yield Button(
                    "⬇ Herunterladen",
                    id="btn-download-one",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "🗑 Löschen",
                    id="btn-delete-one",
                    variant="error",
                    disabled=True,
                )
                yield Button(
                    "Alle fehlenden ⬇",
                    id="btn-download-all",
                    variant="default",
                )
            yield RichLog(id="download-log", wrap=True, markup=True, max_lines=300)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        """Baut die Modell-Liste neu auf und aktualisiert die OptionList."""
        from src.config import get_config, get_models, get_baseline, get_hardware

        config = get_config()
        hw = get_hardware(config)
        models = get_models(config)
        baseline = get_baseline(config)
        router_name = config.get("router", {}).get("model", "")
        router_vram = config.get("router", {}).get("vram_gb", 2.5)
        available_single = hw.per_gpu_vram_gb - router_vram - 2.0

        self._models_info = []

        # Router
        self._models_info.append(
            {
                "display_id": "router",
                "hf_name": router_name,
                "vram_str": f"{router_vram:.1f} GB",
                "fits": True,
                "downloaded": self._is_downloaded(router_name),
            }
        )

        # Zielmodelle
        for m in models:
            fits = m.vram_gb <= available_single or (
                hw.num_gpus > 1 and m.vram_gb <= hw.total_vram_gb - router_vram - 2.0
            )
            self._models_info.append(
                {
                    "display_id": m.id,
                    "hf_name": m.name,
                    "vram_str": f"{m.vram_gb:.1f} GB",
                    "fits": fits,
                    "downloaded": self._is_downloaded(m.name),
                }
            )

        # Baseline
        self._models_info.append(
            {
                "display_id": f"baseline ({baseline.model_id})",
                "hf_name": baseline.model_name,
                "vram_str": f"{baseline.vram_gb:.1f} GB",
                "fits": True,
                "downloaded": self._is_downloaded(baseline.model_name),
            }
        )

        ol = self.query_one("#model-list", OptionList)
        ol.clear_options()
        for info in self._models_info:
            ok = info["downloaded"]
            status_text = ("✓ OK", "green") if ok else ("✗ Fehlt", "red")
            extra = (
                ("  [VRAM passt nicht]", "dim yellow") if not info["fits"] else ("", "")
            )
            ol.add_option(
                Option(
                    Text.assemble(
                        (f"{info['display_id']:<30}", "bold"),
                        (f"{info['vram_str']:>8}  ", "dim"),
                        status_text,
                        extra,
                    )
                )
            )

        # Buttons nach Refresh deaktivieren (Auswahl weg)
        self._selected_idx = None
        self.query_one("#btn-download-one", Button).disabled = True
        self.query_one("#btn-delete-one", Button).disabled = True

    @on(OptionList.OptionHighlighted, "#model-list")
    def on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        """Buttons aktivieren/deaktivieren je nach Download-Status des markierten Modells."""
        self._selected_idx = event.option_index
        info = self._models_info[self._selected_idx]
        self.query_one("#btn-download-one", Button).disabled = info["downloaded"]
        self.query_one("#btn-delete-one", Button).disabled = not info["downloaded"]

    @on(Button.Pressed, "#btn-download-one")
    @work(thread=True)
    def download_one(self) -> None:
        if self._selected_idx is None:
            return
        info = self._models_info[self._selected_idx]
        self._do_download(info["hf_name"])
        self.app.call_from_thread(self._refresh_list)

    @on(Button.Pressed, "#btn-delete-one")
    @work(thread=True)
    def delete_one(self) -> None:
        if self._selected_idx is None:
            return
        info = self._models_info[self._selected_idx]
        self._do_delete(info["hf_name"])
        self.app.call_from_thread(self._refresh_list)

    @on(Button.Pressed, "#btn-download-all")
    @work(thread=True)
    def download_all(self) -> None:
        log = self.query_one("#download-log", RichLog)
        missing = [m["hf_name"] for m in self._models_info if not m["downloaded"]]
        if not missing:
            log.write("[green]Alle Modelle bereits heruntergeladen.[/green]")
            return
        for name in missing:
            self._do_download(name)
        self.app.call_from_thread(self._refresh_list)

    def _do_download(self, hf_name: str) -> None:
        log = self.query_one("#download-log", RichLog)
        log.write(f"[cyan]⬇ Lade: {hf_name}...[/cyan]")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(hf_name)
            log.write(f"[green]✓ Fertig: {hf_name}[/green]")
        except Exception as e:
            log.write(f"[red]✗ Fehler ({hf_name}): {e}[/red]")

    def _do_delete(self, hf_name: str) -> None:
        log = self.query_one("#download-log", RichLog)
        log.write(f"[yellow]🗑 Lösche: {hf_name}...[/yellow]")
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            commit_hashes = [
                rev.commit_hash
                for repo in cache_info.repos
                if repo.repo_id == hf_name
                for rev in repo.revisions
            ]
            if commit_hashes:
                strategy = cache_info.delete_revisions(*commit_hashes)
                strategy.execute()
                log.write(f"[green]✓ Gelöscht: {hf_name}[/green]")
            else:
                log.write(f"[yellow]Nicht im Cache gefunden: {hf_name}[/yellow]")
        except Exception as e:
            log.write(f"[red]✗ Fehler beim Löschen ({hf_name}): {e}[/red]")

    def _is_downloaded(self, model_name: str) -> bool:
        try:
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_id == model_name:
                    return True
            return False
        except Exception:
            return False


# ─── Haupt-App ──────────────────────────────────────────────────────────────


class LLMRoutingApp(App):
    """LLM-Routing Energiemessung — Interaktive TUI."""

    TITLE = "LLM-Routing Energiemessung"
    SUB_TITLE = "Interaktive Experiment-Steuerung"

    CSS = """
    Screen {
        background: $surface;
    }
    #content {
        padding: 1 2;
        height: 1fr;
    }
    #step {
        text-align: center;
        padding: 1 0;
        margin-bottom: 1;
    }
    #options, #scenarios, #file-list, #file-tree, #compare-tree {
        height: 1fr;
        margin: 0 4;
    }
    #compare-status {
        height: auto;
        margin: 1 4 0 4;
        color: $text-muted;
    }
    #form {
        margin: 0 8;
        height: auto;
    }
    .label {
        margin: 1 0 0 0;
    }
    Checkbox {
        margin: 1 0 0 0;
    }
    .button-row {
        height: auto;
        margin: 2 0 0 0;
    }
    #confirm-buttons {
        align: center middle;
        height: auto;
        margin: 2 0 0 0;
    }
    Button {
        margin: 0 2;
    }
    #summary {
        margin: 1 4;
        height: auto;
    }
    #process-log {
        height: 1fr;
        margin: 0 2;
        border: solid $accent;
    }
    #download-log {
        height: 1fr;
        margin: 0 2;
        border: solid $accent;
    }
    #detail-log {
        height: 1fr;
        margin: 0 2;
        border: solid $accent;
    }
    #plot-row {
        height: 20;
        margin: 1 2 0 2;
    }
    #plot-power, #plot-accuracy {
        width: 1fr;
    }
    #model-list {
        height: 1fr;
        margin: 0 2;
        max-height: 14;
    }
    #model-buttons {
        height: auto;
        margin: 1 2;
        align: left middle;
    }
    #model-buttons Button {
        margin: 0 1;
    }
    #progress {
        height: auto;
        max-height: 1;
        margin: 0 2;
        color: $text-muted;
    }
    #status {
        text-align: center;
        height: auto;
        padding: 1 0;
    }
    #no-results {
        text-align: center;
        margin: 4 0;
    }
    """

    BINDINGS = [
        ("escape", "quit", "Beenden"),
        ("ctrl+q", "quit", "Beenden"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.wizard: dict = {}
        self.profiles: dict = {}
        self.benchmarks: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="content"):
            yield Static(
                "[bold cyan]Schritt 1[/] — Experiment wählen\n"
                "[dim]↑↓ navigieren · Enter auswählen · Esc beenden[/]",
                id="step",
            )
            yield OptionList(
                *[
                    Option(
                        Text.assemble(
                            (label, "bold"),
                            "\n",
                            (f"  {desc}", "dim italic"),
                        )
                    )
                    for _, label, desc in EXPERIMENTS
                ],
                id="experiment-options",
            )
        yield Footer()

    def on_mount(self) -> None:
        # Docker-Modus im Header anzeigen
        from src.config import is_docker

        if is_docker():
            self.sub_title = "Docker-Modus | Interaktive Experiment-Steuerung"
        try:
            self.profiles = load_profile_info()
        except Exception as e:
            self.notify(
                f"Fehler: Profile konnten nicht geladen werden: {e}", severity="error"
            )
        try:
            self.benchmarks = get_available_benchmarks()
        except Exception as e:
            self.notify(
                f"Fehler: Benchmarks konnten nicht geladen werden: {e}",
                severity="error",
            )

    @on(OptionList.OptionSelected, "#experiment-options")
    def on_experiment_select(self, event: OptionList.OptionSelected) -> None:
        key, label, _ = EXPERIMENTS[event.option_index]
        self.wizard = {"experiment": key, "experiment_label": label}

        if key == "test_energy":
            cmd = build_command(key)
            self.push_screen(ProcessScreen(cmd=cmd, title=label))
        elif key == "results":
            self.push_screen(ResultsScreen())
        elif key == "compare":
            self.push_screen(CompareScreen())
        elif key == "download":
            self.push_screen(DownloadScreen())
        else:
            self.push_screen(ProfileScreen())

    def go_home(self) -> None:
        """Alle Screens schliessen und Zurück zum Hauptmenü."""
        self.wizard = {}
        for _ in range(20):
            try:
                self.pop_screen()
            except Exception:
                break


def main():
    """Startet die TUI."""
    app = LLMRoutingApp()
    app.run()


if __name__ == "__main__":
    main()
