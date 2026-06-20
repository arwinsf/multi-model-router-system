"""CLI fuer nachtraegliche Neu-Auswertung bestehender Resultate."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from src.config import get_config, is_docker, setup_docker_environment
from src.evaluation.postprocess import (
    benchmark_uses_lm_eval_backend,
    build_evaluation_config,
    reevaluate_result_bundle,
)
from src.utils import setup_logging
from src.utils.data import RESULTS_FILENAME, write_results_artifacts


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


def _bootstrap_docker_hf_environment(profile: str | None) -> None:
    """Initialisiert HF-Cache/Token fuer Docker-CLI-Pfade ohne run.py."""
    if not is_docker():
        return

    try:
        setup_docker_environment(get_config(profile))
    except Exception:
        setup_docker_environment()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bestehende results.json-Dateien ohne Re-Inferenz neu auswerten"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Experiment-Verzeichnisse oder direkte Pfade auf results.json",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--extractor",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Plots nach der Neu-Auswertung nicht neu erzeugen",
    )
    args = parser.parse_args()

    _bootstrap_docker_hf_environment(args.profile)

    logger = setup_logging()

    for input_path in args.paths:
        result_file = _resolve_result_file(input_path)
        experiment_dir = result_file.parent

        with open(result_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        config_snapshot = payload.setdefault("config", {})
        benchmark_name = config_snapshot.get("benchmark")

        if args.profile is not None or args.extractor is not None:
            logger.warning(
                "--extractor wird ignoriert; --profile beeinflusst nur das Docker-HF-Setup. Die Neu-Auswertung nutzt immer den gespeicherten Replay-/Postprocess-Pfad."
            )

        logger.info("=" * 60)
        logger.info("Neu-Auswertung: %s", experiment_dir.name)
        logger.info("Benchmark: %s", benchmark_name)
        if benchmark_uses_lm_eval_backend(benchmark_name):
            logger.info("Evaluations-Backend: lm-eval Replay")

        summary = reevaluate_result_bundle(
            payload,
            benchmark_name=benchmark_name,
            log=logger,
        )

        evaluation_cfg = build_evaluation_config({}, benchmark_name)
        evaluation_cfg["last_evaluation_method"] = (
            "lm_eval_replay" if summary["used_lm_eval"] else "postprocess"
        )
        evaluation_cfg["last_evaluated_samples"] = summary["evaluated_samples"]
        evaluation_cfg["last_changed_extractions"] = summary["changed_extractions"]
        evaluation_cfg["last_changed_correctness"] = summary["changed_correctness"]
        evaluation_cfg["last_recomputed_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        if summary.get("backend_version") and summary.get("used_lm_eval"):
            evaluation_cfg["backend_version"] = summary["backend_version"]
        config_snapshot["evaluation"] = evaluation_cfg

        write_results_artifacts(payload, experiment_dir, verbose=False)

        if not args.no_plots:
            try:
                from src.plotting import generate_experiment_plots

                generate_experiment_plots(experiment_dir)
            except Exception as exc:
                logger.warning("Plot-Generierung fehlgeschlagen: %s", exc)

        # Accuracy aus dem aktualisierten Payload lesen
        accuracy = None
        m = payload.get("measurement")
        if isinstance(m, dict):
            accuracy = m.get("measurement_accuracy")
        elif isinstance(payload.get("measurements"), list):
            accs = [
                ms.get("measurement_accuracy")
                for ms in payload["measurements"]
                if isinstance(ms, dict) and ms.get("measurement_accuracy") is not None
            ]
            if accs:
                accuracy = sum(accs) / len(accs)

        logger.info(
            "Samples: %s | bewertet: %s | Extraktionen geaendert: %s | Correctness geaendert: %s",
            summary["total_samples"],
            summary["evaluated_samples"],
            summary["changed_extractions"],
            summary["changed_correctness"],
        )
        if summary.get("backend_version") and summary.get("used_lm_eval"):
            logger.info("lm-eval Version: %s", summary["backend_version"])

        acc_str = f"{accuracy:.1%}" if accuracy is not None else "n/a"
        logger.info(
            "Neu-Auswertung abgeschlossen: %s — Accuracy: %s",
            experiment_dir.name,
            acc_str,
        )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
