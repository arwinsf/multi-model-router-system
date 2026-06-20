from __future__ import annotations

"""Datenverarbeitung und -speicherung.

Experiment-Ergebnisse werden in einem Verzeichnis pro Experiment gespeichert:
    {output_dir}/{name}_{timestamp}/
        results.json            — Vollständige Ergebnisse (Config + Messung(en) + Samples)
        config.json             — Nur der Config-Block (für schnellen Zugriff)
        measurement_summary.csv — Flache Zeile(n) mit Messungsmetriken
        samples.csv             — Flache CSV aller Samples der Messung
        power_samples.csv       — Leistungszeitreihe der Messung
        scheduler_events.csv    — Optionale Scheduler-Ereignisse der Messung
        average_summary.json    — Optional: Metadaten fuer Average-Results
"""

from datetime import datetime
from pathlib import Path
import csv
import json
import shutil
from statistics import pstdev
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None


RESULTS_FILENAME = "results.json"
CONFIG_FILENAME = "config.json"
MEASUREMENT_SUMMARY_FILENAME = "measurement_summary.csv"
SAMPLES_FILENAME = "samples.csv"
POWER_SAMPLES_FILENAME = "power_samples.csv"
SCHEDULER_EVENTS_FILENAME = "scheduler_events.csv"
AVERAGE_SUMMARY_FILENAME = "average_summary.json"
EXPERIMENT_ALIAS_KEY = "experiment_alias"


def _require_pandas():
    if pd is None:
        raise ModuleNotFoundError(
            "pandas wird fuer diese CSV/DataFrame-Funktion benoetigt, ist aber nicht installiert."
        )
    return pd


def get_experiment_alias(config: dict[str, Any] | None) -> str | None:
    """Liest einen optionalen Alias aus dem Config-Block."""
    if not isinstance(config, dict):
        return None
    alias = config.get(EXPERIMENT_ALIAS_KEY)
    if not isinstance(alias, str):
        return None
    alias = alias.strip()
    return alias or None


def get_experiment_display_name(
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> str:
    """Gibt Alias oder Ordnernamen als Anzeigename zurück."""
    resolved = Path(path)
    if resolved.name == RESULTS_FILENAME:
        resolved = resolved.parent
    return get_experiment_alias(config) or resolved.name


def _normalize_measurement_dict(
    measurement: dict[str, Any],
    default_id: int = 1,
) -> dict[str, Any]:
    normalized = dict(measurement)
    normalized.setdefault("measurement_id", default_id)
    normalized.setdefault("samples", [])
    normalized.setdefault("scheduler_events", [])
    return normalized


def _scalar_items(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _build_experiment_context(
    config: dict[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    hardware = config.get("hardware", {}) if isinstance(config, dict) else {}
    return {
        "experiment_name": experiment_name,
        "experiment_alias": get_experiment_alias(config),
        "profile": config.get("profile") if isinstance(config, dict) else None,
        "experiment_type": (
            config.get("experiment_type") if isinstance(config, dict) else None
        ),
        "benchmark": config.get("benchmark") if isinstance(config, dict) else None,
        "batch_size": config.get("batch_size") if isinstance(config, dict) else None,
        "configured_num_samples": (
            config.get("num_samples") if isinstance(config, dict) else None
        ),
        "num_gpus": hardware.get("num_gpus") if isinstance(hardware, dict) else None,
        "per_gpu_vram_gb": (
            hardware.get("per_gpu_vram_gb") if isinstance(hardware, dict) else None
        ),
    }


def _count_scheduler_actions(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        action = event.get("action", "unknown")
        counts[action] = counts.get(action, 0) + 1
    return counts


def make_writable(path: str | Path) -> None:
    """Ändert die Berechtigungen, damit auch der Host-Nutzer (außerhalb von Docker) vollen Zugriff hat."""
    import subprocess

    p = Path(path)
    if not p.exists():
        return
    try:
        if p.is_dir():
            p.chmod(0o777)
        else:
            p.chmod(0o666)
    except Exception as e:
        # Versuche mit sudo falls chmod fehlschlägt
        try:
            if p.is_dir():
                subprocess.run(["sudo", "chmod", "777", str(p)], check=True)
            else:
                subprocess.run(["sudo", "chmod", "666", str(p)], check=True)
        except Exception as e2:
            print(f"[make_writable] chmod failed for {p}: {e} | sudo failed: {e2}")


def save_results(
    data: dict[str, Any],
    output_dir: str | Path,
    name: str,
    power_data: tuple[list[float], float] | None = None,
    log_file: Path | None = None,
    failed: bool = False,
    verbose: bool = True,
) -> Path:
    """Speichert Experiment-Ergebnisse als Verzeichnis mit JSON und CSV-Artefakten."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_writable(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_FAILED" if failed else ""
    experiment_dir = output_dir / f"{name}_{timestamp}{suffix}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    make_writable(experiment_dir)

    return write_results_artifacts(
        data,
        experiment_dir,
        power_data=power_data,
        log_file=log_file,
        failed=failed,
        verbose=verbose,
    )


def write_results_artifacts(
    data: dict[str, Any],
    experiment_dir: str | Path,
    power_data: tuple[list[float], float] | None = None,
    log_file: Path | None = None,
    failed: bool = False,
    verbose: bool = True,
) -> Path:
    """Schreibt canonical JSON und abgeleitete CSV-Artefakte in ein Experiment-Verzeichnis."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = experiment_dir.name

    def _emit(message: str) -> None:
        if verbose:
            print(message)

    payload = dict(data)
    if "schema_version" not in payload and (
        "measurement" in payload or "measurements" in payload
    ):
        payload["schema_version"] = 3 if "measurements" in payload else 2

    json_path = experiment_dir / RESULTS_FILENAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    make_writable(json_path)
    _emit(f"Gespeichert: {json_path}")

    if "config" in payload:
        config_path = experiment_dir / CONFIG_FILENAME
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload["config"], f, indent=2, ensure_ascii=False, default=str)
        make_writable(config_path)

    normalized = normalize_results(payload)
    context = _build_experiment_context(normalized.get("config", {}), experiment_name)

    try:
        summary_path = experiment_dir / MEASUREMENT_SUMMARY_FILENAME
        _write_measurement_summary_csv(
            normalized,
            summary_path,
            context,
            verbose=verbose,
        )
        make_writable(summary_path)
    except Exception as exc:
        if not failed:
            raise
        print(
            f"Warnung: measurement_summary.csv konnte nicht geschrieben werden: {exc}"
        )

    try:
        samples_path = experiment_dir / SAMPLES_FILENAME
        _write_samples_csv(
            normalized,
            samples_path,
            context,
            verbose=verbose,
        )
        make_writable(samples_path)
    except Exception as exc:
        if not failed:
            raise
        print(f"Warnung: samples.csv konnte nicht geschrieben werden: {exc}")

    if power_data:
        try:
            power_path = experiment_dir / POWER_SAMPLES_FILENAME
            _write_power_csv(
                power_data,
                power_path,
                experiment_name,
                verbose=verbose,
            )
            make_writable(power_path)
        except Exception as exc:
            if not failed:
                raise
            print(f"Warnung: power_samples.csv konnte nicht geschrieben werden: {exc}")

    try:
        sched_path = experiment_dir / SCHEDULER_EVENTS_FILENAME
        _write_scheduler_events_csv(
            normalized,
            sched_path,
            context,
            verbose=verbose,
        )
        make_writable(sched_path)
    except Exception as exc:
        if not failed:
            raise
        print(f"Warnung: scheduler_events.csv konnte nicht geschrieben werden: {exc}")

    try:
        average_path = experiment_dir / AVERAGE_SUMMARY_FILENAME
        _write_average_summary_json(
            normalized,
            average_path,
            verbose=verbose,
        )
        if average_path.exists():
            make_writable(average_path)
    except Exception as exc:
        if not failed:
            raise
        print(f"Warnung: average_summary.json konnte nicht geschrieben werden: {exc}")

    if log_file and Path(log_file).exists():
        log_dest = experiment_dir / "experiment.txt"
        shutil.copy2(log_file, log_dest)
        make_writable(log_dest)
        _emit(f"Gespeichert: {log_dest}")

    return experiment_dir


def set_experiment_alias(path: str | Path, alias: str | None) -> Path:
    """Setzt oder entfernt einen optionalen Alias fuer ein bestehendes Experiment."""
    path = Path(path)
    result_path = path / RESULTS_FILENAME if path.is_dir() else path
    experiment_dir = result_path.parent

    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.setdefault("config", {})
    cleaned_alias = (alias or "").strip()
    if cleaned_alias:
        config[EXPERIMENT_ALIAS_KEY] = cleaned_alias
    else:
        config.pop(EXPERIMENT_ALIAS_KEY, None)

    return write_results_artifacts(payload, experiment_dir, verbose=False)


def _write_measurement_summary_csv(
    data: dict[str, Any],
    csv_path: Path,
    context: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Schreibt flache Zeilen mit Messungsmetriken."""
    measurements = data.get("measurements", [])
    if not measurements:
        return

    rows = []
    for measurement in measurements:
        thinking_stats = measurement.get("thinking_stats", {})
        scheduler_events = measurement.get("scheduler_events", [])
        startup_events = sum(
            1 for event in scheduler_events if event.get("phase") == "startup_preload"
        )
        runtime_events = sum(
            1 for event in scheduler_events if event.get("phase") == "runtime"
        )

        rows.append(
            {
                **context,
                **_scalar_items(measurement),
                "benchmark": measurement.get("benchmark") or context.get("benchmark"),
                "thinking_count": thinking_stats.get("thinking"),
                "no_thinking_count": thinking_stats.get("no_thinking"),
                "routing_stats_json": (
                    json.dumps(measurement["routing_stats"], ensure_ascii=False)
                    if measurement.get("routing_stats")
                    else None
                ),
                "thinking_stats_json": (
                    json.dumps(thinking_stats, ensure_ascii=False)
                    if thinking_stats
                    else None
                ),
                "scheduler_event_count": len(scheduler_events),
                "scheduler_startup_event_count": startup_events,
                "scheduler_runtime_event_count": runtime_events,
                "scheduler_action_counts_json": (
                    json.dumps(
                        _count_scheduler_actions(scheduler_events),
                        ensure_ascii=False,
                    )
                    if scheduler_events
                    else None
                ),
            }
        )

    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"Gespeichert: {csv_path}")


def _write_samples_csv(
    data: dict[str, Any],
    csv_path: Path,
    context: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Schreibt alle Samples als flache CSV-Datei."""
    rows = []
    for measurement in data.get("measurements", []):
        measurement_id = measurement.get("measurement_id", 1)
        scenario = measurement.get("scenario", "")
        benchmark = measurement.get("benchmark") or context.get("benchmark", "")

        for sample in measurement.get("samples", []):
            rows.append(
                {
                    "experiment_name": context.get("experiment_name"),
                    "experiment_type": context.get("experiment_type"),
                    "profile": context.get("profile"),
                    "measurement_id": measurement_id,
                    "scenario": scenario,
                    "benchmark": benchmark,
                    "batch_size": context.get("batch_size"),
                    "configured_num_samples": context.get("configured_num_samples"),
                    **sample,
                }
            )

    if not rows:
        return

    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"Gespeichert: {csv_path}")


def _write_power_csv(
    power_data: tuple[list[float], float],
    csv_path: Path,
    experiment_name: str,
    verbose: bool = True,
) -> None:
    """Schreibt die Power-Zeitreihe als CSV."""
    samples, interval = power_data
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_name", "measurement_id", "time_s", "power_watts"])
        for i, power in enumerate(samples):
            writer.writerow(
                [experiment_name, 1, round(i * interval, 3), round(power, 2)]
            )

    if verbose:
        print(f"Gespeichert: {csv_path}")


def _write_scheduler_events_csv(
    data: dict[str, Any],
    csv_path: Path,
    context: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Schreibt Scheduler-Ereignisse als flache CSV-Datei."""
    rows = []
    for measurement in data.get("measurements", []):
        events = measurement.get("scheduler_events", [])
        for index, event in enumerate(events, start=1):
            rows.append(
                {
                    **context,
                    "measurement_id": measurement.get("measurement_id", 1),
                    "scenario": measurement.get("scenario"),
                    "event_index": index,
                    **event,
                }
            )

    if not rows:
        return

    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"Gespeichert: {csv_path}")


def _write_average_summary_json(
    data: dict[str, Any],
    json_path: Path,
    verbose: bool = True,
) -> None:
    """Schreibt Average-Metadaten als separates JSON-Artefakt."""
    if not any(key in data for key in ("average", "source_experiments", "averaging")):
        return

    payload = {
        key: data[key]
        for key in ("average", "source_experiments", "averaging")
        if key in data
    }
    extended = _build_average_details(data.get("measurements", []))
    if extended:
        payload["average_details"] = extended

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print(f"Gespeichert: {json_path}")


def _format_float(value: float | None, digits: int = 6) -> str | None:
    if value is None:
        return None
    return f"{value:.{digits}f}"


def _sample_accuracy_counts(measurement: dict[str, Any]) -> tuple[int, int]:
    samples = measurement.get("samples", [])
    evaluated = [sample for sample in samples if sample.get("is_correct") is not None]
    correct = sum(1 for sample in evaluated if sample.get("is_correct"))
    return correct, len(evaluated)


def _measurement_accuracy_value(measurement: dict[str, Any]) -> float | None:
    value = measurement.get("measurement_accuracy")
    if isinstance(value, (int, float)):
        return float(value)

    correct, evaluated = _sample_accuracy_counts(measurement)
    if evaluated == 0:
        return None
    return correct / evaluated


def _measurement_eq_score_value(measurement: dict[str, Any]) -> float | None:
    value = measurement.get("measurement_eq_score")
    if isinstance(value, (int, float)):
        return float(value)

    accuracy = _measurement_accuracy_value(measurement)
    energy = measurement.get("measurement_dynamic_energy_joules")
    if accuracy is None or not isinstance(energy, (int, float)) or energy <= 0:
        return None
    return accuracy / (float(energy) / 3_600_000.0)


def _build_average_details(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    if not measurements:
        return {}

    accuracies = [
        accuracy
        for measurement in measurements
        if (accuracy := _measurement_accuracy_value(measurement)) is not None
    ]
    eq_scores = [
        score
        for measurement in measurements
        if (score := _measurement_eq_score_value(measurement)) is not None
    ]
    counts = [_sample_accuracy_counts(measurement) for measurement in measurements]
    total_correct = sum(correct for correct, _ in counts)
    total_evaluated = sum(evaluated for _, evaluated in counts)
    weighted_accuracy = total_correct / total_evaluated if total_evaluated > 0 else None

    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    mean_eq_score = sum(eq_scores) / len(eq_scores) if eq_scores else None

    return {
        "evaluated_samples": total_evaluated,
        "correct_samples": total_correct,
        "mean_accuracy_formatted": _format_float(mean_accuracy),
        "mean_accuracy_percent": (
            mean_accuracy * 100 if mean_accuracy is not None else None
        ),
        "mean_accuracy_percent_formatted": (
            f"{mean_accuracy * 100:.3f}%" if mean_accuracy is not None else None
        ),
        "weighted_accuracy": weighted_accuracy,
        "weighted_accuracy_formatted": _format_float(weighted_accuracy),
        "weighted_accuracy_percent_formatted": (
            f"{weighted_accuracy * 100:.3f}%" if weighted_accuracy is not None else None
        ),
        "accuracy_values": accuracies,
        "accuracy_min": min(accuracies) if accuracies else None,
        "accuracy_max": max(accuracies) if accuracies else None,
        "accuracy_std": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_eq_score_formatted": _format_float(mean_eq_score),
        "eq_score_values": eq_scores,
        "eq_score_min": min(eq_scores) if eq_scores else None,
        "eq_score_max": max(eq_scores) if eq_scores else None,
        "eq_score_std": pstdev(eq_scores) if len(eq_scores) > 1 else 0.0,
    }


def load_results(path: str | Path) -> dict:
    """Lädt Ergebnisse aus einer canonical ``results.json`` oder einem Experiment-Verzeichnis."""
    path = Path(path)

    if path.is_dir():
        path = path / RESULTS_FILENAME

    if path.name != RESULTS_FILENAME:
        raise ValueError(
            f"Erwartet canonical {RESULTS_FILENAME}, erhalten: {path.name}"
        )
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return normalize_results(data)


def load_power_samples(path: str | Path) -> pd.DataFrame:
    """Lädt Power-Zeitreihen aus einer CSV-Datei."""
    pandas = _require_pandas()
    path = Path(path)
    if path.is_dir():
        path = path / POWER_SAMPLES_FILENAME

    if not path.exists():
        return pandas.DataFrame(
            columns=["experiment_name", "measurement_id", "time_s", "power_watts"]
        )

    df = pandas.read_csv(path)
    if "experiment_name" not in df.columns:
        df["experiment_name"] = path.parent.name
    if "measurement_id" not in df.columns:
        df["measurement_id"] = 1
    return df


def load_measurement_summary(path: str | Path) -> pd.DataFrame:
    """Lädt die flache Messungszusammenfassung eines Experiments."""
    pandas = _require_pandas()
    path = Path(path)
    if path.is_dir():
        path = path / MEASUREMENT_SUMMARY_FILENAME

    if not path.exists():
        return pandas.DataFrame()

    df = pandas.read_csv(path)
    if "experiment_name" not in df.columns:
        df["experiment_name"] = path.parent.name
    if "measurement_id" not in df.columns:
        df["measurement_id"] = 1
    return df


def load_scheduler_events(path: str | Path) -> pd.DataFrame:
    """Lädt flache Scheduler-Ereignisse eines Experiments."""
    pandas = _require_pandas()
    path = Path(path)
    if path.is_dir():
        path = path / SCHEDULER_EVENTS_FILENAME

    if not path.exists():
        return pandas.DataFrame()

    df = pandas.read_csv(path)
    if "experiment_name" not in df.columns:
        df["experiment_name"] = path.parent.name
    if "measurement_id" not in df.columns:
        df["measurement_id"] = 1
    return df


def normalize_results(data: dict[str, Any]) -> dict:
    """Normalisiert Ergebnisse auf das kanonische Messungslisten-Format."""
    if not isinstance(data, dict):
        raise TypeError("Erwartet Ergebnisdaten als Dictionary.")

    result: dict[str, Any] = {}
    for key in (
        "config",
        "error",
        "schema_version",
        "average",
        "averaging",
        "source_experiments",
    ):
        if key in data:
            result[key] = data[key]

    raw_measurements: list[Any] = []
    if "measurements" in data:
        measurements = data["measurements"]
        if not isinstance(measurements, list):
            raise TypeError("'measurements' muss eine Liste sein.")
        raw_measurements = measurements
    elif "measurement" in data:
        raw_measurements = [data["measurement"]]

    if not raw_measurements:
        result["measurements"] = []
        return result

    normalized_measurements = []
    for index, measurement_data in enumerate(raw_measurements, start=1):
        if not isinstance(measurement_data, dict):
            raise TypeError("Messungen müssen Dictionaries sein.")
        normalized_measurements.append(
            _normalize_measurement_dict(measurement_data, default_id=index)
        )

    result["measurement"] = normalized_measurements[0]
    result["measurements"] = normalized_measurements
    return result
