"""Cross-Experiment-Vergleich: Vergleichsplots über mehrere Experimente.

Lädt Ergebnisse aus mehreren Experiment-Verzeichnissen und erzeugt
Vergleichsplots (Energie, Genauigkeit, Qualität vs. Energie, Power-Overlay).

Nutzung:
    python -m experiments.compare results/router_mmlu-pro_20260412/ results/baseline_mmlu-pro_20260412/
    python -m experiments.compare results/router_* --output results/comparison/
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.plotting import generate_comparison_plots
from src.utils.data import get_experiment_display_name, load_results
from src.utils import setup_logging

COMPARISON_MANIFEST_NAME = "comparison_inputs.json"


def _write_comparison_manifest(output_dir: Path, experiment_dirs: list[Path]) -> Path:
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.parent.parent
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_dirs": [],
    }

    for exp_dir in experiment_dirs:
        resolved = Path(exp_dir).resolve()
        try:
            stored = str(resolved.relative_to(project_root))
        except ValueError:
            stored = str(resolved)
        payload["experiment_dirs"].append(stored)

    manifest_path = output_dir / COMPARISON_MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return manifest_path


def main():
    """Hauptfunktion für Cross-Experiment-Vergleich."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Cross-Experiment-Vergleich: Vergleichsplots aus mehreren Experimenten"
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Pfade zu Experiment-Verzeichnissen (mit results.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Ausgabeverzeichnis für Vergleichsplots (default: results/comparison_{timestamp}/)",
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Experiment-Verzeichnisse validieren
    valid_dirs = []
    for exp_dir in args.experiments:
        if exp_dir.is_dir():
            if (exp_dir / "results.json").exists():
                valid_dirs.append(exp_dir)
            else:
                logger.warning(f"Kein results.json in {exp_dir}, übersprungen.")
        elif exp_dir.is_file() and exp_dir.name == "results.json":
            valid_dirs.append(exp_dir.parent)
        else:
            logger.warning(f"Ungültiger Pfad: {exp_dir}, übersprungen.")

    if len(valid_dirs) < 2:
        logger.error(
            f"Mindestens 2 gültige Experiment-Verzeichnisse benötigt, "
            f"nur {len(valid_dirs)} gefunden."
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Cross-Experiment-Vergleich")
    logger.info("=" * 60)
    for d in valid_dirs:
        logger.info(f"  {d.name}")

    # Ausgabeverzeichnis
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"comparison_{timestamp}"

    # Zusammenfassung auf stdout
    logger.info(f"\nExperiment-Zusammenfassung:")
    logger.info("-" * 60)
    for d in valid_dirs:
        try:
            data = load_results(d)
            measurements = data.get("measurements", [])
            config = data.get("config", {})
            display_name = get_experiment_display_name(d, config)
            exp_type = config.get("experiment_type", "?")
            benchmark = config.get("benchmark", "?")
            num_measurements = len(measurements)

            # Aggregate
            total_samples = sum(len(m.get("samples", [])) for m in measurements)
            accs = []
            energies = []
            for measurement in measurements:
                samples = measurement.get("samples", [])
                n = len(samples)
                if n > 0:
                    accs.append(
                        sum(1 for s in samples if s.get("is_correct")) / n * 100
                    )
                dyn_e = measurement.get("measurement_dynamic_energy_joules")
                if dyn_e is not None:
                    energies.append(dyn_e)

            avg_acc = sum(accs) / len(accs) if accs else 0
            avg_energy = sum(energies) / len(energies) if energies else 0

            logger.info(
                f"  {display_name}: [{exp_type}] {benchmark} · "
                f"{num_measurements} Messung{'en' if num_measurements != 1 else ''} · {total_samples} Samples · "
                f"Acc {avg_acc:.1f}% · {avg_energy:.1f} J"
            )
        except Exception as e:
            logger.warning(f"  {d.name}: Fehler beim Laden: {e}")

    # Vergleichsplots generieren
    logger.info(f"\nErzeuge Vergleichsplots in: {output_dir}")
    generated = generate_comparison_plots(valid_dirs, output_dir)
    manifest_path = _write_comparison_manifest(output_dir, valid_dirs)
    logger.info(f"Vergleichs-Metadaten gespeichert: {manifest_path}")

    if generated:
        logger.info(f"\n{len(generated)} Plots erzeugt:")
        for p in generated:
            logger.info(f"  {p}")
    else:
        logger.warning("Keine Plots erzeugt.")

    logger.info("\nVergleich abgeschlossen!")


if __name__ == "__main__":
    main()
