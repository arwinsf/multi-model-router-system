"""Average-Result aus mehreren kompatiblen Experiment-Runs erzeugen.

Der erzeugte Ordner enthaelt eine ``results.json`` mit mehreren Messungen.
Die bestehenden Comparison-Plots mitteln ueber diese Messungen und behandeln
den Ordner dadurch wie einen normalen, aber gemittelten Run.
"""

from __future__ import annotations

import copy
import csv
import importlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.plotting import generate_experiment_plots
from src.utils import setup_logging
from src.utils.data import (
    EXPERIMENT_ALIAS_KEY,
    POWER_SAMPLES_FILENAME,
    RESULTS_FILENAME,
    get_experiment_display_name,
    load_power_samples,
    load_results,
    make_writable,
    save_results,
)
from src.utils.metrics import compute_accuracy, compute_measurement_eq_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = SRC_ROOT / "results"


def _resolve_result_file(path: Path) -> Path:
    if path.is_dir():
        path = path / RESULTS_FILENAME
    if path.name != RESULTS_FILENAME:
        raise ValueError(
            f"Erwartet Experiment-Verzeichnis oder {RESULTS_FILENAME}: {path}"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _stored_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "run"


def _scenario_signature(measurements: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted({str(measurement.get("scenario", "?")) for measurement in measurements})
    )


def _compatibility_signature(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    measurements = payload.get("measurements", [])
    return {
        "benchmark": config.get("benchmark")
        or _first_measurement_value(measurements, "benchmark"),
        "experiment_type": config.get("experiment_type"),
        "batch_size": config.get("batch_size"),
        "scenarios": _scenario_signature(measurements),
    }


def _first_measurement_value(measurements: list[dict[str, Any]], key: str) -> Any:
    for measurement in measurements:
        value = measurement.get(key)
        if value is not None:
            return value
    return None


def _validate_compatible(
    payloads: list[dict[str, Any]],
    *,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    signatures = [_compatibility_signature(payload) for payload in payloads]
    reference = signatures[0]
    mismatches = []
    for index, signature in enumerate(signatures[1:], start=2):
        for key, reference_value in reference.items():
            if signature.get(key) != reference_value:
                mismatches.append(
                    f"Run {index}: {key}={signature.get(key)!r} statt {reference_value!r}"
                )

    if mismatches and not allow_mismatch:
        joined = "\n  - ".join(mismatches)
        raise ValueError(
            "Die ausgewählten Runs sind nicht kompatibel. "
            "Wähle Runs mit gleichem Benchmark, Typ, Batch und Szenario "
            "oder nutze --allow-mismatch.\n  - "
            f"{joined}"
        )

    return reference


def _numeric_mean(measurements: list[dict[str, Any]], key: str) -> float | None:
    values = [measurement.get(key) for measurement in measurements]
    numeric_values = [
        float(value) for value in values if isinstance(value, (int, float))
    ]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _measurement_accuracy(measurement: dict[str, Any]) -> float | None:
    stored = measurement.get("measurement_accuracy")
    if isinstance(stored, (int, float)):
        return float(stored)
    return compute_accuracy(measurement.get("samples", []))


def _average_summary(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [
        accuracy
        for measurement in measurements
        if (accuracy := _measurement_accuracy(measurement)) is not None
    ]
    eq_scores = [
        score
        for measurement in measurements
        if (score := compute_measurement_eq_score(measurement)) is not None
    ]
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    mean_eq_score = sum(eq_scores) / len(eq_scores) if eq_scores else None

    summary = {
        "run_count": len(measurements),
        "total_samples": sum(
            len(measurement.get("samples", [])) for measurement in measurements
        ),
        "mean_accuracy": mean_accuracy,
        "mean_dynamic_energy_joules": _numeric_mean(
            measurements,
            "measurement_dynamic_energy_joules",
        ),
        "mean_dynamic_energy_watthours": _numeric_mean(
            measurements,
            "measurement_dynamic_energy_watthours",
        ),
        "mean_duration_seconds": _numeric_mean(
            measurements,
            "measurement_duration_seconds",
        ),
        "mean_millijoules_per_output_token": _numeric_mean(
            measurements,
            "measurement_millijoules_per_output_token",
        ),
        "mean_tokens_per_second": _numeric_mean(
            measurements,
            "measurement_tokens_per_second",
        ),
        "mean_eq_score": mean_eq_score,
    }
    return summary


def _write_combined_power_samples(
    experiment_dir: Path,
    source_experiments: list[dict[str, Any]],
) -> Path | None:
    """Kopiert Power-Zeitreihen der Quellruns in das Average-Ergebnis."""
    rows: list[dict[str, Any]] = []
    for source in source_experiments:
        source_path = Path(source.get("path", ""))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        power_df = load_power_samples(source_path)
        if power_df.empty:
            continue

        source_index = source.get("source_index")
        source_name = source.get("display_name") or source_path.name
        for _, row in power_df.iterrows():
            rows.append(
                {
                    "experiment_name": experiment_dir.name,
                    "measurement_id": source_index,
                    "source_experiment": source_name,
                    "source_measurement_id": row.get("measurement_id", 1),
                    "time_s": row.get("time_s"),
                    "power_watts": row.get("power_watts"),
                }
            )

    if not rows:
        return None

    power_path = experiment_dir / POWER_SAMPLES_FILENAME
    fieldnames = [
        "experiment_name",
        "measurement_id",
        "source_experiment",
        "source_measurement_id",
        "time_s",
        "power_watts",
    ]
    with open(power_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    make_writable(power_path)
    return power_path


def _build_average_payload(
    result_files: list[Path],
    *,
    alias: str | None = None,
    allow_mismatch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = []
    for result_file in result_files:
        payload = load_results(result_file)
        if not payload.get("measurements"):
            raise ValueError(f"Keine Messungen in {result_file}")
        loaded.append((result_file, payload))

    signature = _validate_compatible(
        [payload for _, payload in loaded],
        allow_mismatch=allow_mismatch,
    )

    base_config = copy.deepcopy(loaded[0][1].get("config", {}))
    created_at = datetime.now(timezone.utc).isoformat()
    combined_measurements: list[dict[str, Any]] = []
    source_experiments = []

    next_measurement_id = 1
    for source_index, (result_file, payload) in enumerate(loaded, start=1):
        experiment_dir = result_file.parent
        source_measurements = payload.get("measurements", [])
        source_info = {
            "source_index": source_index,
            "path": _stored_path(experiment_dir),
            "result_file": _stored_path(result_file),
            "display_name": get_experiment_display_name(
                experiment_dir,
                payload.get("config", {}),
            ),
            "measurement_count": len(source_measurements),
            "sample_count": sum(
                len(measurement.get("samples", []))
                for measurement in source_measurements
            ),
        }
        source_experiments.append(source_info)

        for measurement in source_measurements:
            copied = copy.deepcopy(measurement)
            copied["source_experiment"] = experiment_dir.name
            copied["source_result_path"] = source_info["path"]
            copied["source_measurement_id"] = copied.get("measurement_id", 1)
            copied["measurement_id"] = next_measurement_id
            combined_measurements.append(copied)
            next_measurement_id += 1

    run_count = len(combined_measurements)
    benchmark = signature.get("benchmark") or base_config.get("benchmark") or "unknown"
    experiment_type = (
        signature.get("experiment_type")
        or base_config.get("experiment_type")
        or "experiment"
    )
    scenarios = signature.get("scenarios") or (experiment_type,)
    scenario_slug = _slugify("_".join(scenarios))

    if alias is None:
        alias = f"Average {run_count}x {experiment_type} {benchmark}"

    base_config["measurement_mode"] = "average"
    base_config["average_run_count"] = run_count
    base_config["average_source_result_count"] = len(source_experiments)
    base_config["average_created_at"] = created_at
    base_config["averaged_from"] = source_experiments
    base_config[EXPERIMENT_ALIAS_KEY] = alias

    average = _average_summary(combined_measurements)
    payload = {
        "schema_version": 3,
        "config": base_config,
        "measurements": combined_measurements,
        "average": average,
        "source_experiments": source_experiments,
        "averaging": {
            "created_at": created_at,
            "source_result_count": len(source_experiments),
            "measurement_count": run_count,
            "compatibility_signature": signature,
        },
    }
    naming = {
        "benchmark": benchmark,
        "experiment_type": experiment_type,
        "scenario_slug": scenario_slug,
        "run_count": run_count,
    }
    return payload, naming


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Mehrere kompatible Experiment-Runs zu einem Average-Result bündeln"
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Experiment-Verzeichnisse oder results.json-Dateien",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Zielverzeichnis (default: src/results)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Anzeigename fuer den erzeugten Average-Run",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Kompatibilitätsprüfung lockern (nur für bewusste Sonderfälle)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Experiment-Plots nach dem Schreiben nicht generieren",
    )
    args = parser.parse_args()

    logger = setup_logging()
    result_files = [_resolve_result_file(path) for path in args.experiments]

    if len(result_files) < 2:
        raise SystemExit(
            "Mindestens zwei Runs auswählen, um einen Durchschnitt zu bilden."
        )

    payload, naming = _build_average_payload(
        result_files,
        alias=args.alias.strip() if args.alias else None,
        allow_mismatch=args.allow_mismatch,
    )
    prefix = (
        f"average_{naming['benchmark']}_{naming['experiment_type']}_"
        f"{naming['scenario_slug']}_{naming['run_count']}runs"
    )

    logger.info("=" * 60)
    logger.info("Average-Result")
    logger.info("=" * 60)
    for result_file in result_files:
        logger.info("  %s", result_file.parent.name)

    average = payload["average"]
    accuracy = average.get("mean_accuracy")
    energy = average.get("mean_dynamic_energy_joules")
    mj_per_token = average.get("mean_millijoules_per_output_token")
    logger.info("Runs: %s", average["run_count"])
    logger.info("Samples gesamt: %s", average["total_samples"])
    if accuracy is not None:
        logger.info("Mean Accuracy: %.2f%%", accuracy * 100)
    if energy is not None:
        logger.info("Mean Dynamic Energy: %.2f J", energy)
    if mj_per_token is not None:
        logger.info("Mean mJ/Output-Token: %.2f", mj_per_token)

    experiment_dir = save_results(
        payload,
        args.output_dir,
        prefix,
    )

    power_path = _write_combined_power_samples(
        experiment_dir,
        payload["source_experiments"],
    )
    if power_path is not None:
        logger.info("Power-Samples gespeichert: %s", power_path)

    if not args.no_plots:
        try:
            generate_experiment_plots(experiment_dir)
        except Exception as exc:
            logger.warning("Plot-Generierung fehlgeschlagen: %s", exc)

    logger.info("Average-Result gespeichert in: %s", experiment_dir)


if __name__ == "__main__":
    main()
