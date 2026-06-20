#!/usr/bin/env python3
"""Regeneriert Plot-Artefakte in bestehenden Results-Verzeichnissen.

Unterstützt zwei Modi:
    - experiment: per-Experiment-Plots in bestehenden Ergebnisordnern neu erzeugen
    - comparison: Vergleichsplots neu erzeugen oder bestehende Comparison-Ordner
      anhand eines gespeicherten Input-Manifests aktualisieren

Das Skript liegt absichtlich auf Root-Ebene neben ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


COMPARISON_MANIFEST_NAME = "comparison_inputs.json"
EXPERIMENT_TYPES = ("baseline", "single", "router", "random")


@dataclass(frozen=True)
class ExperimentRef:
    path: Path
    kind: str
    benchmark: str | None


def _load_plotting_api():
    from src.plotting import generate_comparison_plots, generate_experiment_plots

    return generate_experiment_plots, generate_comparison_plots


def _resolve_results_root(raw_path: str | None) -> Path:
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate.resolve()

    candidates = [PROJECT_ROOT / "src" / "results", PROJECT_ROOT / "results"]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[0]
    if len(existing) == 1:
        return existing[0]

    scored = []
    for candidate in existing:
        try:
            count = sum(1 for child in candidate.iterdir() if child.is_dir())
        except OSError:
            count = 0
        prefers_src = candidate == PROJECT_ROOT / "src" / "results"
        scored.append((count, prefers_src, candidate))
    return max(scored)[2]


def _expand_types(values: list[str]) -> list[str]:
    normalized = [value.lower() for value in values]
    if "all" in normalized:
        return list(EXPERIMENT_TYPES)
    expanded = []
    for value in normalized:
        mapped = "random" if value == "random_routing" else value
        if mapped in EXPERIMENT_TYPES and mapped not in expanded:
            expanded.append(mapped)
    return expanded


def _infer_kind_from_name(name: str) -> str | None:
    normalized = name.lower()
    if normalized.startswith("comparison_"):
        return "comparison"
    if normalized.startswith("baseline_"):
        return "baseline"
    if normalized.startswith("single_"):
        return "single"
    if normalized.startswith("router_"):
        return "router"
    if normalized.startswith("random_routing_") or normalized.startswith("random_"):
        return "random"
    return None


def _read_small_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_experiment_config(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        return _read_small_json(config_path)

    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        return {}

    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("config", {})


def _classify_experiment_dir(experiment_dir: Path) -> str | None:
    kind = _infer_kind_from_name(experiment_dir.name)
    if kind and kind != "comparison":
        return kind

    config = _read_experiment_config(experiment_dir)
    experiment_type = str(config.get("experiment_type", "")).lower()
    if experiment_type == "router":
        return "router"
    if experiment_type == "random_routing":
        return "random"
    if experiment_type == "baseline":
        return "baseline"
    return kind


def _resolve_experiment_arg(raw_value: str, results_root: Path) -> Path:
    candidate = Path(raw_value)
    search_order = []

    if candidate.is_absolute():
        search_order.append(candidate)
    else:
        search_order.append(results_root / candidate)
        search_order.append(PROJECT_ROOT / candidate)

    for path in search_order:
        resolved = path.resolve()
        if resolved.is_file() and resolved.name == "results.json":
            resolved = resolved.parent
        if resolved.is_dir() and (resolved / "results.json").exists():
            return resolved

    raise FileNotFoundError(
        f"Kein gültiges Experiment-Verzeichnis gefunden: {raw_value}"
    )


def _resolve_dir_arg(raw_value: str, results_root: Path) -> Path:
    candidate = Path(raw_value)
    search_order = []

    if candidate.is_absolute():
        search_order.append(candidate)
    else:
        search_order.append(results_root / candidate)
        search_order.append(PROJECT_ROOT / candidate)

    for path in search_order:
        resolved = path.resolve()
        if resolved.is_dir():
            return resolved

    raise FileNotFoundError(f"Kein Verzeichnis gefunden: {raw_value}")


def _resolve_output_dir(raw_value: str | None, results_root: Path) -> Path:
    if raw_value is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return results_root / f"comparison_{timestamp}"

    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    if len(candidate.parts) == 1:
        return results_root / candidate
    return (PROJECT_ROOT / candidate).resolve()


def _discover_experiments(
    results_root: Path,
    kinds: list[str],
    benchmarks: set[str] | None,
) -> list[ExperimentRef]:
    if not results_root.exists():
        return []

    discovered: list[ExperimentRef] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir() or not (child / "results.json").exists():
            continue

        kind = _classify_experiment_dir(child)
        if kind is None or kind not in kinds:
            continue

        config = _read_experiment_config(child)
        benchmark = config.get("benchmark")
        if benchmarks and benchmark not in benchmarks:
            continue

        discovered.append(ExperimentRef(path=child, kind=kind, benchmark=benchmark))

    return discovered


def _load_explicit_experiments(
    raw_values: list[str],
    results_root: Path,
    allowed_kinds: list[str] | None,
    benchmarks: set[str] | None,
) -> list[ExperimentRef]:
    resolved_refs: list[ExperimentRef] = []
    for raw_value in raw_values:
        experiment_dir = _resolve_experiment_arg(raw_value, results_root)
        kind = _classify_experiment_dir(experiment_dir)
        config = _read_experiment_config(experiment_dir)
        benchmark = config.get("benchmark")

        if allowed_kinds and kind not in allowed_kinds:
            continue
        if benchmarks and benchmark not in benchmarks:
            continue

        resolved_refs.append(
            ExperimentRef(
                path=experiment_dir, kind=kind or "unknown", benchmark=benchmark
            )
        )

    unique = {ref.path.resolve(): ref for ref in resolved_refs}
    return [unique[path] for path in sorted(unique)]


def _discover_comparison_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return [
        child
        for child in sorted(results_root.iterdir())
        if child.is_dir() and _infer_kind_from_name(child.name) == "comparison"
    ]


def _clean_experiment_plots(experiment_dir: Path) -> None:
    plots_dir = experiment_dir / "plots"
    if plots_dir.exists():
        shutil.rmtree(plots_dir)


def _clean_comparison_plots(comparison_dir: Path) -> None:
    for png_path in comparison_dir.glob("*.png"):
        png_path.unlink()


def _path_for_manifest(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_comparison_manifest(
    output_dir: Path,
    experiment_dirs: list[Path],
    results_root: Path,
    filters: dict | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results_root": _path_for_manifest(results_root),
        "experiment_dirs": [_path_for_manifest(path) for path in experiment_dirs],
        "filters": filters or {},
    }
    manifest_path = output_dir / COMPARISON_MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return manifest_path


def _load_comparison_manifest(comparison_dir: Path) -> list[Path]:
    manifest_path = comparison_dir / COMPARISON_MANIFEST_NAME
    payload = _read_small_json(manifest_path)
    experiment_dirs = []
    for raw_path in payload.get("experiment_dirs", []):
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.name == "results.json":
            resolved = resolved.parent
        experiment_dirs.append(resolved)
    return experiment_dirs


def _print_experiment_summary(experiments: list[ExperimentRef]) -> None:
    print(f"Gefundene Experimente: {len(experiments)}")
    for experiment in experiments:
        benchmark = experiment.benchmark or "?"
        print(f"  [{experiment.kind}] {experiment.path.name} ({benchmark})")


def _group_experiments_by_benchmark(
    experiments: list[ExperimentRef],
) -> dict[str, list[ExperimentRef]]:
    grouped: dict[str, list[ExperimentRef]] = {}
    for experiment in experiments:
        benchmark = experiment.benchmark or "unknown"
        grouped.setdefault(benchmark, []).append(experiment)
    return grouped


def _run_experiment_regeneration(args: argparse.Namespace) -> int:
    results_root = _resolve_results_root(args.results_dir)
    kinds = _expand_types(args.types)
    benchmarks = set(args.benchmarks or []) or None

    if args.inputs:
        experiments = _load_explicit_experiments(
            args.inputs,
            results_root,
            allowed_kinds=kinds,
            benchmarks=benchmarks,
        )
    else:
        experiments = _discover_experiments(results_root, kinds, benchmarks)

    if not experiments:
        print("Keine passenden Experiment-Verzeichnisse gefunden.")
        return 1

    print(f"Results-Root: {results_root}")
    _print_experiment_summary(experiments)

    if args.dry_run:
        return 0

    generate_experiment_plots, _ = _load_plotting_api()

    regenerated = 0
    for experiment in experiments:
        print(f"\nRegeneriere Plots für {experiment.path.name} ...")
        if args.clean:
            _clean_experiment_plots(experiment.path)
        generated = generate_experiment_plots(experiment.path)
        regenerated += 1
        print(f"  Erzeugte Dateien: {len(generated)}")

    print(f"\nFertig: {regenerated} Experiment(e) aktualisiert.")
    return 0


def _run_existing_comparison_regeneration(
    args: argparse.Namespace,
    results_root: Path,
) -> int:
    generate_comparison_plots = None
    if not args.dry_run:
        _, generate_comparison_plots = _load_plotting_api()

    if args.comparison_dirs:
        comparison_dirs = [
            _resolve_dir_arg(raw_value, results_root)
            for raw_value in args.comparison_dirs
        ]
    else:
        comparison_dirs = _discover_comparison_dirs(results_root)

    if not comparison_dirs:
        print("Keine Comparison-Verzeichnisse gefunden.")
        return 1

    refreshed = 0
    skipped = 0
    for comparison_dir in comparison_dirs:
        manifest_path = comparison_dir / COMPARISON_MANIFEST_NAME
        if not manifest_path.exists():
            print(
                f"[SKIP] {comparison_dir.name}: kein {COMPARISON_MANIFEST_NAME} vorhanden"
            )
            skipped += 1
            continue

        experiment_dirs = [
            path
            for path in _load_comparison_manifest(comparison_dir)
            if (path / "results.json").exists()
        ]
        if len(experiment_dirs) < 2:
            print(
                f"[SKIP] {comparison_dir.name}: weniger als 2 gültige Input-Experimente im Manifest"
            )
            skipped += 1
            continue

        print(f"\nRegeneriere Comparison {comparison_dir.name} ...")
        for experiment_dir in experiment_dirs:
            print(f"  - {experiment_dir.name}")

        if args.dry_run:
            refreshed += 1
            continue

        if args.clean:
            _clean_comparison_plots(comparison_dir)
        generated = generate_comparison_plots(experiment_dirs, comparison_dir)
        refreshed += 1
        print(f"  Erzeugte Dateien: {len(generated)}")

    print(f"\nFertig: {refreshed} Comparison(s) aktualisiert, {skipped} übersprungen.")
    return 0 if refreshed > 0 else 1


def _run_new_comparison_generation(args: argparse.Namespace) -> int:
    results_root = _resolve_results_root(args.results_dir)
    kinds = _expand_types(args.types)
    benchmarks = set(args.benchmarks or []) or None

    if args.refresh_existing:
        return _run_existing_comparison_regeneration(args, results_root)

    if args.inputs:
        experiments = _load_explicit_experiments(
            args.inputs,
            results_root,
            allowed_kinds=kinds,
            benchmarks=benchmarks,
        )
    else:
        experiments = _discover_experiments(results_root, kinds, benchmarks)

    if len(experiments) < 2:
        print("Für einen Comparison-Plot werden mindestens 2 Experimente benötigt.")
        return 1

    print(f"Results-Root: {results_root}")
    _print_experiment_summary(experiments)

    if args.group_by_benchmark:
        groups = _group_experiments_by_benchmark(experiments)
    else:
        groups = {"all": experiments}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = (
        _resolve_output_dir(args.output, results_root) if args.output else None
    )

    if args.dry_run:
        for group_name, group_experiments in groups.items():
            if group_name == "all":
                preview_output = output_base or (
                    results_root / f"comparison_{timestamp}"
                )
            else:
                if output_base:
                    preview_output = (
                        output_base / f"comparison_{group_name}_{timestamp}"
                    )
                else:
                    preview_output = (
                        results_root / f"comparison_{group_name}_{timestamp}"
                    )
            print(f"\nComparison-Ziel: {preview_output}")
            for experiment in group_experiments:
                print(f"  - {experiment.path.name}")
        return 0

    _, generate_comparison_plots = _load_plotting_api()

    generated_dirs = 0
    for group_name, group_experiments in groups.items():
        if len(group_experiments) < 2:
            print(f"[SKIP] Gruppe {group_name}: weniger als 2 Experimente")
            continue

        if group_name == "all":
            output_dir = output_base or (results_root / f"comparison_{timestamp}")
        else:
            if output_base:
                output_dir = output_base / f"comparison_{group_name}_{timestamp}"
            else:
                output_dir = results_root / f"comparison_{group_name}_{timestamp}"

        print(f"\nErzeuge Comparison in {output_dir} ...")
        if args.clean and output_dir.exists():
            _clean_comparison_plots(output_dir)

        experiment_dirs = [experiment.path for experiment in group_experiments]
        generated = generate_comparison_plots(experiment_dirs, output_dir)
        manifest_path = _write_comparison_manifest(
            output_dir,
            experiment_dirs,
            results_root,
            filters={
                "types": kinds,
                "benchmarks": sorted(
                    {exp.benchmark for exp in group_experiments if exp.benchmark}
                ),
                "group": group_name,
            },
        )
        generated_dirs += 1
        print(f"  Erzeugte Dateien: {len(generated)}")
        print(f"  Manifest: {manifest_path}")

    if generated_dirs == 0:
        print("Keine Comparison-Verzeichnisse erzeugt.")
        return 1

    print(
        f"\nFertig: {generated_dirs} Comparison-Verzeichnis(se) erzeugt/aktualisiert."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plots in bestehenden Results-Verzeichnissen neu generieren"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Plots für bestehende Experiment-Verzeichnisse neu erzeugen",
    )
    experiment_parser.add_argument(
        "--types",
        nargs="+",
        default=["baseline"],
        choices=[*EXPERIMENT_TYPES, "all", "random_routing"],
        help="Zu berücksichtigende Ergebnis-Typen (default: baseline)",
    )
    experiment_parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optionaler Benchmark-Filter, z.B. gpqa livebench-da",
    )
    experiment_parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Explizite Experiment-Verzeichnisse oder results.json-Dateien",
    )
    experiment_parser.add_argument(
        "--results-dir",
        default=None,
        help="Pfad zum Results-Ordner (default: automatisch src/results oder results)",
    )
    experiment_parser.add_argument(
        "--clean",
        action="store_true",
        help="Vor der Neugenerierung das plots/-Verzeichnis löschen",
    )
    experiment_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, welche Ordner verarbeitet würden",
    )

    comparison_parser = subparsers.add_parser(
        "comparison",
        help="Comparison-Plots neu erzeugen oder bestehende Comparison-Ordner aktualisieren",
    )
    comparison_parser.add_argument(
        "--types",
        nargs="+",
        default=["baseline", "single", "router"],
        choices=[*EXPERIMENT_TYPES, "all", "random_routing"],
        help="Input-Experiment-Typen für neue Comparison-Plots",
    )
    comparison_parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optionaler Benchmark-Filter für neue Comparison-Plots",
    )
    comparison_parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Explizite Experiment-Verzeichnisse oder results.json-Dateien für neue Comparison-Plots",
    )
    comparison_parser.add_argument(
        "--results-dir",
        default=None,
        help="Pfad zum Results-Ordner (default: automatisch src/results oder results)",
    )
    comparison_parser.add_argument(
        "--output",
        default=None,
        help="Ausgabeordner für neue Comparison-Plots; bei --group-by-benchmark als Basisordner verwendet",
    )
    comparison_parser.add_argument(
        "--group-by-benchmark",
        action="store_true",
        help="Erzeuge getrennte Comparison-Ordner pro Benchmark",
    )
    comparison_parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Bestehende comparison_* Ordner mit comparison_inputs.json neu rendern",
    )
    comparison_parser.add_argument(
        "--comparison-dirs",
        nargs="+",
        default=None,
        help="Explizite bestehende Comparison-Ordner für --refresh-existing",
    )
    comparison_parser.add_argument(
        "--clean",
        action="store_true",
        help="Vor der Neugenerierung vorhandene PNGs im Zielordner löschen",
    )
    comparison_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, welche Ordner verarbeitet würden",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "experiment":
        return _run_experiment_regeneration(args)
    if args.command == "comparison":
        return _run_new_comparison_generation(args)

    parser.error(f"Unbekanntes Kommando: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
