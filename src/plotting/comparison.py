"""Cross-Experiment Vergleichsplots.

Laedt Ergebnisse aus mehreren Experiment-Verzeichnissen und erzeugt
Vergleichsplots fuer Energie, Genauigkeit und Effizienz.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.utils.data import (
    get_experiment_display_name,
    load_power_samples,
    load_results,
    make_writable,
)
from src.utils.metrics import EQ_SCORE_ENERGY_UNIT, compute_measurement_eq_score

# --- Globaler Stil (konsistent mit experiment.py) ----------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

_COLORS = plt.cm.tab10.colors

# Quadrant-Farben (Artificial-Analysis-Stil)
_QUADRANT_BEST_COLOR = "#e8f5e9"  # Leichtes Grün
_QUADRANT_GOOD_COLOR = "#f5f5f5"  # Sehr helles Grau
_QUADRANT_NEUTRAL_COLOR = "#fafafa"  # Fast Weiß
_QUADRANT_WORST_COLOR = "#fff3e0"  # Leichtes Orange

# Quadrant-Labels (DE)
_QUADRANT_LABELS = {
    "best": "Ideal",
    "good": "Gut",
    "neutral": "",
    "worst": "Ungünstig",
}


def _draw_quadrant_backgrounds(
    ax,
    x_values: list[float],
    y_values: list[float],
    x_higher_is_better: bool,
    y_higher_is_better: bool,
) -> None:
    """Zeichnet Quadranten-Hintergrund im Artificial-Analysis-Stil.

    Teilt den Plot in 4 Quadranten am Median der Daten und färbt
    den besten Quadranten grünlich ein. Gegenüber wird leicht
    orange eingefärbt.

    Args:
        ax: Matplotlib Axes.
        x_values: X-Datenwerte (für Median-Berechnung).
        y_values: Y-Datenwerte (für Median-Berechnung).
        x_higher_is_better: True wenn hohe X-Werte besser sind.
        y_higher_is_better: True wenn hohe Y-Werte besser sind.
    """
    if len(x_values) < 2 or len(y_values) < 2:
        return

    x_mid = float(np.median(x_values))
    y_mid = float(np.median(y_values))

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Quadranten: (x_range, y_range) → Qualität
    # best = gute X + gute Y, worst = schlechte X + schlechte Y
    quadrants = [
        # (x_start, x_end, y_start, y_end, x_is_good_side, y_is_good_side)
        (x_min, x_mid, y_mid, y_max, not x_higher_is_better, y_higher_is_better),
        (x_mid, x_max, y_mid, y_max, x_higher_is_better, y_higher_is_better),
        (x_min, x_mid, y_min, y_mid, not x_higher_is_better, not y_higher_is_better),
        (x_mid, x_max, y_min, y_mid, x_higher_is_better, not y_higher_is_better),
    ]

    for x0, x1, y0, y1, x_good, y_good in quadrants:
        if x_good and y_good:
            color = _QUADRANT_BEST_COLOR
            label = _QUADRANT_LABELS["best"]
        elif not x_good and not y_good:
            color = _QUADRANT_WORST_COLOR
            label = _QUADRANT_LABELS["worst"]
        else:
            color = _QUADRANT_GOOD_COLOR
            label = _QUADRANT_LABELS["good"]

        ax.axvspan(
            x0,
            x1,
            ymin=(y0 - y_min) / (y_max - y_min),
            ymax=(y1 - y_min) / (y_max - y_min),
            color=color,
            zorder=0,
        )

        if label:
            # Label klein in der Ecke des Quadranten platzieren
            lx = x0 + (x1 - x0) * 0.05
            ly = y1 - (y1 - y0) * 0.08
            ax.text(
                lx,
                ly,
                label,
                fontsize=7,
                color="#888888",
                fontstyle="italic",
                zorder=1,
                va="top",
                ha="left",
            )

    # Quadranten-Kreuz zeichnen
    ax.axvline(x_mid, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(y_mid, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # Achsenlimits wiederherstellen (axvspan kann sie verändern)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _coerce_thinking_flag(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "t", "yes", "thinking"}:
            return True
        if normalized in {"false", "0", "f", "no", "non-thinking", "nothinking"}:
            return False
    return bool(value)


def _normalize_thinking_column(samples_df: pd.DataFrame) -> pd.DataFrame:
    if "enable_thinking" not in samples_df.columns:
        return samples_df

    normalized = samples_df.copy()
    normalized["enable_thinking"] = normalized["enable_thinking"].apply(
        _coerce_thinking_flag
    )
    return normalized.dropna(subset=["enable_thinking"])


def _get_model_order(config: dict, observed_model_ids: list[str]) -> list[str]:
    ordered: list[str] = []
    for model in sorted(config.get("models", []), key=lambda item: item.get("tier", 0)):
        model_id = model.get("id")
        if model_id in observed_model_ids and model_id not in ordered:
            ordered.append(model_id)
    for model_id in sorted(observed_model_ids):
        if model_id not in ordered:
            ordered.append(model_id)
    return ordered


def _resolve_model_order(
    model_order: list[str] | None,
    observed_model_ids: list[str],
) -> list[str]:
    ordered: list[str] = []
    for model_id in model_order or []:
        if model_id in observed_model_ids and model_id not in ordered:
            ordered.append(model_id)
    for model_id in observed_model_ids:
        if model_id not in ordered:
            ordered.append(model_id)
    return ordered


def _is_routing_experiment(experiment: dict) -> bool:
    scenario = experiment.get("scenario", "")
    if "router" in scenario or "random" in scenario:
        return True

    samples_df = experiment.get("samples_df", pd.DataFrame())
    return not samples_df.empty and "routing_decision" in samples_df.columns


# =============================================================================
# Hilfsfunktionen
# =============================================================================


def _load_experiment(experiment_dir: Path) -> dict:
    """Laedt ein Experiment und reichert es mit Metadaten an.

    Returns:
        Dict mit ``name``, ``config``, ``measurements`` und ``power_df``.
    """
    data = load_results(experiment_dir)
    power_df = load_power_samples(experiment_dir)
    name = get_experiment_display_name(experiment_dir, data.get("config", {}))
    measurements = data.get("measurements", [])
    primary_measurement = data.get("measurement") or (
        measurements[0] if measurements else None
    )

    samples_rows = []
    for measurement in measurements:
        measurement_id = measurement.get("measurement_id", 1)
        for sample in measurement.get("samples", []):
            samples_rows.append({"measurement_id": measurement_id, **sample})

    return {
        "name": name,
        "dir": experiment_dir,
        "config": data.get("config", {}),
        "measurements": measurements,
        "samples_df": pd.DataFrame(samples_rows) if samples_rows else pd.DataFrame(),
        "scenario": (
            primary_measurement.get("scenario", "") if primary_measurement else ""
        ),
        "power_df": power_df,
    }


def _compute_measurement_aggregates(experiment: dict) -> dict:
    """Berechnet Aggregate pro Experiment ueber alle Messungen.

    Returns:
        Dict mit ``mean_accuracy``, ``mean_dynamic_energy``,
        ``mean_duration``, ``total_samples``.
    """
    measurements = experiment["measurements"]
    if not measurements:
        return {
            "mean_accuracy": None,
            "mean_dynamic_energy": None,
            "mean_duration": None,
            "mean_latency": None,
            "mean_tokens_per_second": None,
            "mean_thinking_share": None,
            "mean_mj_per_token": None,
            "total_samples": 0,
        }

    accuracies = []
    energies = []
    durations = []
    latencies = []
    throughputs = []
    thinking_shares = []
    eq_scores = []
    mj_per_tokens = []
    total_samples = 0

    for measurement in measurements:
        dyn_e = measurement.get("measurement_dynamic_energy_joules", 0)
        dur = measurement.get("measurement_duration_seconds", 0)
        latency = measurement.get("measurement_avg_latency_seconds")
        throughput = measurement.get("measurement_tokens_per_second")
        thinking_share = measurement.get("measurement_thinking_share")
        mj_pt = measurement.get("measurement_millijoules_per_output_token")
        eq_score = compute_measurement_eq_score(measurement)

        if dyn_e is not None:
            energies.append(dyn_e)
        if dur is not None:
            durations.append(dur)
        if latency is not None:
            latencies.append(latency)
        if throughput is not None:
            throughputs.append(throughput)
        if thinking_share is not None:
            thinking_shares.append(thinking_share)
        if eq_score is not None:
            eq_scores.append(eq_score)
        if mj_pt is not None:
            mj_per_tokens.append(mj_pt)

        samples = measurement.get("samples", [])
        total_samples += len(samples)
        correct_vals = [
            s.get("is_correct") for s in samples if s.get("is_correct") is not None
        ]
        if correct_vals:
            accuracies.append(sum(1 for v in correct_vals if v) / len(correct_vals))

    return {
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
        "mean_dynamic_energy": float(np.mean(energies)) if energies else None,
        "mean_duration": float(np.mean(durations)) if durations else None,
        "mean_latency": float(np.mean(latencies)) if latencies else None,
        "mean_tokens_per_second": (
            float(np.mean(throughputs)) if throughputs else None
        ),
        "mean_thinking_share": (
            float(np.mean(thinking_shares)) if thinking_shares else None
        ),
        "mean_mj_per_token": (float(np.mean(mj_per_tokens)) if mj_per_tokens else None),
        "mean_eq_score": float(np.mean(eq_scores)) if eq_scores else None,
        "total_samples": total_samples,
    }


# =============================================================================
# Hauptfunktion
# =============================================================================


def generate_comparison_plots(
    experiment_dirs: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Erzeugt Vergleichsplots ueber mehrere Experimente.

    Args:
        experiment_dirs: Liste von Experiment-Verzeichnissen.
        output_dir: Zielverzeichnis fuer die Vergleichsplots.

    Returns:
        Liste der erzeugten Plot-Dateien.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_writable(output_dir)

    # Experimente laden
    experiments = []
    for d in experiment_dirs:
        try:
            experiments.append(_load_experiment(Path(d)))
        except Exception as exc:
            print(f"[WARN] Experiment {d} konnte nicht geladen werden: {exc}")

    if not experiments:
        print("[WARN] Keine Experimente geladen, keine Plots erzeugt.")
        return []

    generated: list[Path] = []

    # --- 1. Energie-Vergleich ---
    try:
        out = output_dir / "energy_comparison.png"
        plot_energy_comparison(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_energy_comparison fehlgeschlagen: {exc}")

    # --- 2. Genauigkeits-Vergleich ---
    try:
        out = output_dir / "accuracy_comparison.png"
        plot_accuracy_comparison(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_accuracy_comparison fehlgeschlagen: {exc}")

    # --- 2.5 EQ Score-Vergleich ---
    try:
        out = output_dir / "eq_score_comparison.png"
        plot_eq_score_comparison(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_eq_score_comparison fehlgeschlagen: {exc}")

    # --- 2.6 Routing-Entscheidungen inkl. Score ---
    try:
        out = output_dir / "routing_decision_comparison.png"
        plot_routing_decision_comparison(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_routing_decision_comparison fehlgeschlagen: {exc}")

    # --- 3. Qualitaet vs. Energie (Scatter + Pareto) ---
    try:
        out = output_dir / "quality_vs_energy.png"
        plot_quality_vs_energy(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_quality_vs_energy fehlgeschlagen: {exc}")

    # --- 4. Power-Overlay ---
    try:
        dirs_with_power = [e["dir"] for e in experiments if not e["power_df"].empty]
        if dirs_with_power:
            out = output_dir / "power_overlay.png"
            plot_power_overlay(dirs_with_power, out)
            generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_power_overlay fehlgeschlagen: {exc}")

    # --- 5. Latenz vs. Energie ---
    try:
        out = output_dir / "latency_vs_energy.png"
        plot_latency_vs_energy(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_latency_vs_energy fehlgeschlagen: {exc}")

    # --- 6. Durchsatz vs. Genauigkeit ---
    try:
        out = output_dir / "throughput_vs_accuracy.png"
        plot_throughput_vs_accuracy(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_throughput_vs_accuracy fehlgeschlagen: {exc}")

    # --- 7. Thinking-Anteil vs. Energie ---
    try:
        out = output_dir / "thinking_share_vs_energy.png"
        plot_thinking_share_vs_energy(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_thinking_share_vs_energy fehlgeschlagen: {exc}")

    # --- 8. mJ pro Token Vergleich ---
    try:
        out = output_dir / "mj_per_token_comparison.png"
        plot_mj_per_token_comparison(experiments, out)
        generated.append(out)
    except Exception as exc:
        print(f"[WARN] plot_mj_per_token_comparison fehlgeschlagen: {exc}")

    # Zusammenfassung
    print(f"Vergleichsplots erzeugt ({len(generated)}):")
    for p in generated:
        print(f"  {p}")

    return generated


# =============================================================================
# Einzelne Plot-Funktionen
# =============================================================================


def plot_energy_comparison(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Gruppierte Balken: Dynamische Energie (J) pro Experiment.

    Falls Experimente unterschiedliche Benchmarks enthalten, werden
    die Balken nach Benchmark gruppiert.

    Args:
        experiments: Liste von Experiment-Dicts.
        output_path: Ziel-PNG-Pfad.
    """
    # Aggregate pro Experiment berechnen
    names = []
    energies = []
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        energy = agg["mean_dynamic_energy"]
        if energy is None:
            continue
        names.append(exp["name"])
        energies.append(energy)

    if not energies or all(e == 0 for e in energies):
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(names))]
    bars = ax.bar(x, energies, color=colors, edgecolor="white", width=0.6)

    # Wert-Labels
    for bar, val in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(energies) * 0.01,
            f"{val:.1f} J",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Dynamische Energie (J)")
    ax.set_title("Energievergleich")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_accuracy_comparison(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Gruppierte Balken: Genauigkeit pro Experiment.

    Args:
        experiments: Liste von Experiment-Dicts.
        output_path: Ziel-PNG-Pfad.
    """
    names = []
    accuracies = []
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        accuracy = agg["mean_accuracy"]
        if accuracy is None:
            continue
        names.append(exp["name"])
        accuracies.append(accuracy)

    if not accuracies:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(names))]
    bars = ax.bar(x, accuracies, color=colors, edgecolor="white", width=0.6)

    # Prozent-Labels
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Genauigkeit")
    ax.set_title("Genauigkeitsvergleich")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_eq_score_comparison(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Gruppierte Balken: EQ Score pro Experiment.

    Args:
        experiments: Liste von Experiment-Dicts.
        output_path: Ziel-PNG-Pfad.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from src.utils.metrics import compute_measurement_eq_score, EQ_SCORE_ENERGY_UNIT
    from src.utils.data import make_writable

    # Using the same module-level variables
    _COLORS = plt.cm.tab10.colors

    names = []
    eq_scores = []

    for exp in experiments:
        measurements = exp.get("measurements", [])
        m_scores = []
        for m in measurements:
            score = compute_measurement_eq_score(m)
            if score is not None:
                m_scores.append(score)

        if m_scores:
            names.append(exp["name"])
            eq_scores.append(float(np.mean(m_scores)))

    if not eq_scores:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(names))]
    bars = ax.bar(x, eq_scores, color=colors, edgecolor="white", width=0.6)

    for bar, score in zip(bars, eq_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(eq_scores) * 0.01 if max(eq_scores) > 0 else 0.1),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_xlabel("Experiment")
    ax.set_ylabel(f"EQ Score (Qualität / {EQ_SCORE_ENERGY_UNIT})")
    ax.set_title("Energy-Quality (EQ) Score Vergleich")

    # Extend ylim to fit text
    max_score = max(eq_scores) if eq_scores else 1.0
    ax.set_ylim(0, max_score * 1.15)

    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_routing_decision_comparison(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Vergleicht Routing-Entscheidungen pro Experiment mit Score-Annotationen."""
    routing_experiments: list[tuple[dict, pd.DataFrame]] = []
    model_ids: list[str] = []

    for experiment in experiments:
        if not _is_routing_experiment(experiment):
            continue

        samples_df = experiment.get("samples_df", pd.DataFrame())
        if samples_df.empty or "model_id" not in samples_df.columns:
            continue

        valid = samples_df.dropna(subset=["model_id"]).copy()
        if valid.empty:
            continue

        if "enable_thinking" not in valid.columns:
            valid["enable_thinking"] = False
        valid = _normalize_thinking_column(valid)
        if valid.empty:
            continue

        routing_experiments.append((experiment, valid))
        model_ids.extend(valid["model_id"].dropna().unique().tolist())

    if not routing_experiments:
        return

    combined_model_order: list[str] = []
    for experiment, _ in routing_experiments:
        for model_id in _get_model_order(experiment.get("config", {}), model_ids):
            if model_id not in combined_model_order:
                combined_model_order.append(model_id)
    combined_model_order = _resolve_model_order(
        combined_model_order,
        list(dict.fromkeys(model_ids)),
    )
    if not combined_model_order:
        return

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(4.5, len(routing_experiments) * 0.9 + len(combined_model_order) * 0.2),
        )
    )

    names = [experiment["name"] for experiment, _ in routing_experiments]
    y_pos = np.arange(len(routing_experiments))
    left = np.zeros(len(routing_experiments))
    base_colors = {
        model_id: _COLORS[idx % len(_COLORS)]
        for idx, model_id in enumerate(combined_model_order)
    }

    for model_id in combined_model_order:
        for mode, alpha, hatch, label_suffix in [
            (False, 0.55, None, "N"),
            (True, 0.9, "//", "T"),
        ]:
            shares = []
            for _, samples_df in routing_experiments:
                total = len(samples_df)
                count = len(
                    samples_df[
                        (samples_df["model_id"] == model_id)
                        & (samples_df["enable_thinking"] == mode)
                    ]
                )
                shares.append((count / total) * 100 if total > 0 else 0.0)

            if not any(share > 0 for share in shares):
                continue

            ax.barh(
                y_pos,
                shares,
                left=left,
                color=base_colors[model_id],
                alpha=alpha,
                edgecolor="white",
                height=0.7,
                hatch=hatch,
                label=f"{model_id} {label_suffix}",
            )
            left += np.array(shares)

    for idx, (experiment, _) in enumerate(routing_experiments):
        aggregates = _compute_measurement_aggregates(experiment)
        score_parts = []
        if aggregates["mean_accuracy"] is not None:
            score_parts.append(f"Acc {aggregates['mean_accuracy']:.1%}")
        if aggregates["mean_eq_score"] is not None:
            score_parts.append(f"EQ {aggregates['mean_eq_score']:.3f}")
        if score_parts:
            ax.text(101.0, idx, " | ".join(score_parts), va="center", fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    deduped: dict[str, Patch] = {}
    for handle, label in zip(handles, labels):
        if label not in deduped:
            deduped[label] = handle

    ax.set_xlim(0, 118)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Anteil der Prompts (%)")
    ax.set_ylabel("Experiment")
    ax.set_title("Routing-Entscheidungen im Vergleich")
    ax.invert_yaxis()
    if deduped:
        ax.legend(
            deduped.values(),
            deduped.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=3,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_quality_vs_energy(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Scatterplot: Genauigkeit vs. Energie mit Pareto-Frontlinie.

    X-Achse = dynamische Energie (J), Y-Achse = Genauigkeit.
    Jeder Punkt ist ein Experiment, beschriftet mit dem Namen.
    Die Pareto-Front verbindet die dominierenden Punkte
    (hoechste Genauigkeit bei niedrigster Energie).

    Args:
        experiments: Liste von Experiment-Dicts.
        output_path: Ziel-PNG-Pfad.
    """
    points = []  # (energy, accuracy, name)
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        e = agg["mean_dynamic_energy"]
        a = agg["mean_accuracy"]
        if e is not None and a is not None and e > 0:
            points.append((e, a, exp["name"]))

    if len(points) < 1:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    energies = [p[0] for p in points]
    accuracies = [p[1] for p in points]
    labels = [p[2] for p in points]

    # Punkte plotten
    for i, (e, a, label) in enumerate(points):
        color = _COLORS[i % len(_COLORS)]
        ax.scatter(e, a, s=80, color=color, zorder=3, edgecolors="white", linewidth=0.5)
        ax.annotate(
            label,
            (e, a),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            color=color,
        )

    # Quadranten-Hintergrund (Artificial-Analysis-Stil)
    # Niedrige Energie + Hohe Genauigkeit = Ideal (oben-links)
    _draw_quadrant_backgrounds(
        ax,
        energies,
        accuracies,
        x_higher_is_better=False,
        y_higher_is_better=True,
    )

    # Pareto-Front berechnen und zeichnen
    # Pareto-optimal: kein anderer Punkt hat WENIGER Energie UND MEHR Genauigkeit
    pareto_indices = _compute_pareto_front(energies, accuracies)
    if len(pareto_indices) > 1:
        pareto_e = [energies[i] for i in pareto_indices]
        pareto_a = [accuracies[i] for i in pareto_indices]
        # Nach Energie sortieren fuer Linienplot
        sorted_pairs = sorted(zip(pareto_e, pareto_a))
        ax.plot(
            [p[0] for p in sorted_pairs],
            [p[1] for p in sorted_pairs],
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.6,
            label="Pareto-Front",
        )
        ax.legend(loc="lower right")

    ax.set_xlabel("Dynamische Energie (J)")
    ax.set_ylabel("Genauigkeit")
    ax.set_title("Qualität vs. Energieverbrauch")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def _compute_pareto_front(
    energies: list[float],
    accuracies: list[float],
) -> list[int]:
    """Bestimmt die Indizes der Pareto-optimalen Punkte.

    Ein Punkt ist Pareto-optimal, wenn kein anderer Punkt sowohl
    weniger Energie verbraucht als auch eine hoehere Genauigkeit hat.

    Args:
        energies: Liste der Energie-Werte (kleiner = besser).
        accuracies: Liste der Genauigkeits-Werte (groesser = besser).

    Returns:
        Liste der Indizes der Pareto-optimalen Punkte.
    """
    n = len(energies)
    pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominiert i, wenn j weniger Energie UND mehr Genauigkeit hat
            if energies[j] <= energies[i] and accuracies[j] >= accuracies[i]:
                if energies[j] < energies[i] or accuracies[j] > accuracies[i]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)
    return pareto


def plot_power_overlay(
    experiment_dirs: list[Path],
    output_path: Path,
) -> None:
    """Ueberlagerte Leistungskurven aus mehreren Experimenten.

    Zeigt die Leistung ueber Zeit fuer mehrere Experimente
    uebereinander (erste Messung jedes Experiments).

    Args:
        experiment_dirs: Liste von Experiment-Verzeichnissen
            (mit ``power_samples.csv``).
        output_path: Ziel-PNG-Pfad.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, exp_dir in enumerate(experiment_dirs):
        exp_dir = Path(exp_dir)
        power_df = load_power_samples(exp_dir)
        if power_df.empty:
            continue

        color = _COLORS[i % len(_COLORS)]
        result_data = load_results(exp_dir)
        label = get_experiment_display_name(exp_dir, result_data.get("config", {}))
        config = result_data.get("config", {})

        if (
            config.get("measurement_mode") == "average"
            and "measurement_id" in power_df.columns
            and power_df["measurement_id"].nunique() > 1
        ):
            plot_df = power_df.copy()
            # Align source runs by elapsed time and average equal time bins.
            plot_df["time_s"] = plot_df["time_s"].round(1)
            plot_df = (
                plot_df.groupby("time_s", as_index=False)["power_watts"]
                .mean()
                .sort_values("time_s")
            )
            label = f"{label} (Mean)"
        else:
            # Nur die erste Messung verwenden (fuer uebersichtliche Darstellung)
            plot_df = power_df
            if "measurement_id" in plot_df.columns:
                first_measurement = plot_df["measurement_id"].min()
                plot_df = plot_df[plot_df["measurement_id"] == first_measurement]

        ax.plot(
            plot_df["time_s"],
            plot_df["power_watts"],
            linewidth=0.8,
            color=color,
            label=label,
            alpha=0.8,
        )

    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Leistung (W)")
    ax.set_title("Leistungsvergleich über Zeit")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_latency_vs_energy(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Scatterplot: mittlere Anfrage-Latenz vs. dynamische Energie."""
    points = []
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        latency = agg["mean_latency"]
        energy = agg["mean_dynamic_energy"]
        if latency is not None and energy is not None:
            points.append((latency, energy, exp["name"]))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (latency, energy, label) in enumerate(points):
        color = _COLORS[i % len(_COLORS)]
        ax.scatter(
            latency,
            energy,
            s=80,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            label,
            (latency, energy),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            zorder=4,
        )

    # Quadranten-Hintergrund (Artificial-Analysis-Stil)
    # Niedrige Latenz + Niedrige Energie = Ideal (unten-links)
    _draw_quadrant_backgrounds(
        ax,
        [p[0] for p in points],
        [p[1] for p in points],
        x_higher_is_better=False,
        y_higher_is_better=False,
    )

    ax.set_xlabel("Mittlere Anfrage-Latenz (s)")
    ax.set_ylabel("Dynamische Energie (J)")
    ax.set_title("Latenz vs. Energie")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_throughput_vs_accuracy(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Scatterplot: Token-Durchsatz vs. Genauigkeit."""
    points = []
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        throughput = agg["mean_tokens_per_second"]
        accuracy = agg["mean_accuracy"]
        if throughput is not None and accuracy is not None:
            points.append((throughput, accuracy, exp["name"]))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (throughput, accuracy, label) in enumerate(points):
        color = _COLORS[i % len(_COLORS)]
        ax.scatter(
            throughput,
            accuracy,
            s=80,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            label,
            (throughput, accuracy),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            zorder=4,
        )

    # Quadranten-Hintergrund (Artificial-Analysis-Stil)
    # Hoher Durchsatz + Hohe Genauigkeit = Ideal (oben-rechts)
    _draw_quadrant_backgrounds(
        ax,
        [p[0] for p in points],
        [p[1] for p in points],
        x_higher_is_better=True,
        y_higher_is_better=True,
    )

    ax.set_xlabel("Output-Tokens pro Sekunde")
    ax.set_ylabel("Genauigkeit")
    ax.set_title("Durchsatz vs. Genauigkeit")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_thinking_share_vs_energy(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Scatterplot: Thinking-Anteil vs. dynamische Energie."""
    points = []
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        thinking_share = agg["mean_thinking_share"]
        energy = agg["mean_dynamic_energy"]
        if thinking_share is not None and energy is not None:
            points.append((thinking_share, energy, exp["name"]))

    if not points:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (thinking_share, energy, label) in enumerate(points):
        color = _COLORS[i % len(_COLORS)]
        x_value = thinking_share * 100
        ax.scatter(
            x_value,
            energy,
            s=80,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            label,
            (x_value, energy),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            zorder=4,
        )

    # Quadranten-Hintergrund (Artificial-Analysis-Stil)
    # Niedriger Thinking-Anteil + Niedrige Energie = Ideal (unten-links)
    _draw_quadrant_backgrounds(
        ax,
        [p[0] * 100 for p in points],
        [p[1] for p in points],
        x_higher_is_better=False,
        y_higher_is_better=False,
    )

    ax.set_xlabel("Thinking-Anteil (%)")
    ax.set_ylabel("Dynamische Energie (J)")
    ax.set_title("Thinking-Anteil vs. Energie")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_mj_per_token_comparison(
    experiments: list[dict],
    output_path: Path,
) -> None:
    """Gruppierte Balken: Millijoule pro Output-Token pro Experiment.

    Args:
        experiments: Liste von Experiment-Dicts.
        output_path: Ziel-PNG-Pfad.
    """
    names = []
    mj_vals = []
    for exp in experiments:
        agg = _compute_measurement_aggregates(exp)
        mj = agg["mean_mj_per_token"]
        if mj is None:
            continue
        names.append(exp["name"])
        mj_vals.append(mj)

    if not mj_vals:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(names))]
    bars = ax.bar(x, mj_vals, color=colors, edgecolor="white", width=0.6)

    # Wert-Labels
    for bar, val in zip(bars, mj_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mj_vals) * 0.01,
            f"{val:.1f} mJ",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Energie pro Token (mJ/Token)")
    ax.set_title("Effizienzvergleich: Millijoule pro Output-Token")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)
