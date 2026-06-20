"""Per-Experiment Plots: Leistung, Routing, Energie, Genauigkeit.

Erzeugt PNG-Plots fuer ein einzelnes Experiment-Verzeichnis.
Alle Plots werden in einem Unterverzeichnis ``plots/`` abgelegt.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.utils.data import load_results, load_power_samples, make_writable
from src.utils.metrics import EQ_SCORE_ENERGY_UNIT, compute_measurement_eq_score

# --- Globaler Stil -----------------------------------------------------------

# Akademisch sauber: dezentes Raster, serifenlose Schrift
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

# Standard-Farbpalette (tab10)
_COLORS = plt.cm.tab10.colors
_NON_THINKING_COLOR = _COLORS[0]
_THINKING_COLOR = _COLORS[1]
_OVERALL_COLOR = _COLORS[2]


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
    if not observed_model_ids:
        return []

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


# =============================================================================
# Hauptfunktion
# =============================================================================


def generate_experiment_plots(experiment_dir: Path) -> list[Path]:
    """Erzeugt alle anwendbaren Plots fuer ein Experiment-Verzeichnis.

    Liest ``results.json`` und optional ``power_samples.csv``, erstellt
    Plots in ``{experiment_dir}/plots/`` und gibt die erzeugten Pfade
    zurueck.

    Args:
        experiment_dir: Pfad zum Experiment-Verzeichnis
            (enthaelt ``results.json``).

    Returns:
        Liste der erzeugten Plot-Dateien.
    """
    experiment_dir = Path(experiment_dir)
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    make_writable(plots_dir)

    data = load_results(experiment_dir)
    config = data.get("config", {})
    measurements = data.get("measurements", [])
    primary_measurement = data.get("measurement") or (
        measurements[0] if measurements else None
    )

    # Power-Zeitreihe (optional)
    power_csv = experiment_dir / "power_samples.csv"
    power_df = load_power_samples(power_csv) if power_csv.exists() else pd.DataFrame()

    # Samples als flaches DataFrame (ueber alle Messungen)
    samples_rows = []
    for measurement in measurements:
        measurement_id = measurement.get("measurement_id", 1)
        for sample in measurement.get("samples", []):
            samples_rows.append({"measurement_id": measurement_id, **sample})
    samples_df = pd.DataFrame(samples_rows) if samples_rows else pd.DataFrame()
    model_order = _get_model_order(
        config,
        (
            samples_df["model_id"].dropna().unique().tolist()
            if not samples_df.empty and "model_id" in samples_df.columns
            else []
        ),
    )

    # Szenario-Typ bestimmen (baseline / router_scheduler / random_routing)
    scenario = primary_measurement.get("scenario", "") if primary_measurement else ""
    is_routing = "router" in scenario or "random" in scenario

    generated: list[Path] = []

    # --- 1. Leistung ueber Zeit ---
    if not power_df.empty:
        try:
            idle_power = (
                primary_measurement.get("measurement_idle_power_watts", 0.0)
                if primary_measurement
                else 0.0
            )
            out = plots_dir / "power_over_time.png"
            plot_power_over_time(power_df, idle_power, out)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_power_over_time fehlgeschlagen: {exc}")

    # --- 2. Routing-Verteilung (nur Router/Random) ---
    if is_routing and primary_measurement:
        try:
            routing_stats = primary_measurement.get("routing_stats", {})
            thinking_stats = primary_measurement.get("thinking_stats", {})
            out = plots_dir / "routing_distribution.png"
            plot_routing_distribution(
                samples_df,
                routing_stats,
                thinking_stats,
                out,
                model_order=model_order,
            )
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_routing_distribution fehlgeschlagen: {exc}")

    # --- 3. Arbeitslast pro Modell ---
    if not samples_df.empty and "model_id" in samples_df.columns:
        try:
            out = plots_dir / "workload_by_model.png"
            plot_workload_by_model(samples_df, out, model_order=model_order)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_workload_by_model fehlgeschlagen: {exc}")

    # --- 4. Genauigkeit pro Modell ---
    if not samples_df.empty and "model_id" in samples_df.columns:
        try:
            out = plots_dir / "accuracy_per_model.png"
            plot_accuracy_per_model(samples_df, out, model_order=model_order)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_accuracy_per_model fehlgeschlagen: {exc}")

    # --- 5. Latenz pro Modell ---
    if not samples_df.empty and "latency_seconds" in samples_df.columns:
        try:
            out = plots_dir / "latency_by_model.png"
            plot_latency_by_model(samples_df, out, model_order=model_order)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_latency_by_model fehlgeschlagen: {exc}")

    # --- 6. Thinking-Overhead ---
    if is_routing and not samples_df.empty and "enable_thinking" in samples_df.columns:
        try:
            out = plots_dir / "thinking_overhead.png"
            plot_thinking_overhead(samples_df, out)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_thinking_overhead fehlgeschlagen: {exc}")

    # --- 7. Token-Verteilung ---
    if not samples_df.empty and "output_tokens" in samples_df.columns:
        try:
            out = plots_dir / "token_distribution.png"
            plot_token_distribution(samples_df, out, is_routing=is_routing)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_token_distribution fehlgeschlagen: {exc}")

    # --- 8. Scheduler-Ereignisse ---
    if (
        is_routing
        and primary_measurement
        and primary_measurement.get("scheduler_events")
    ):
        try:
            out = plots_dir / "scheduler_event_counts.png"
            plot_scheduler_event_counts(primary_measurement, out)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_scheduler_event_counts fehlgeschlagen: {exc}")

    # --- 9. EQ-Score pro Messung ---
    if measurements:
        try:
            out = plots_dir / "eq_score_summary.png"
            plot_eq_score_summary(measurements, out)
            generated.append(out)
        except Exception as exc:
            print(f"[WARN] plot_eq_score_summary fehlgeschlagen: {exc}")

    # Zusammenfassung
    print(f"Plots erzeugt ({len(generated)}):")
    for p in generated:
        print(f"  {p}")

    return generated


# =============================================================================
# Einzelne Plot-Funktionen
# =============================================================================


def plot_power_over_time(
    power_df: pd.DataFrame,
    idle_power: float,
    output_path: Path,
) -> None:
    """Liniendiagramm: Leistung (W) ueber Zeit (s).

    Zeigt die gemessene Leistungskurve und eine horizontale
    gestrichelte Linie fuer P_idle.

    Args:
        power_df: DataFrame mit Spalten ``time_s``, ``power_watts``,
            optional ``measurement_id``.
        idle_power: Idle-Leistung in Watt (P_idle).
        output_path: Ziel-PNG-Pfad.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Falls mehrere Messungen vorliegen: pro Messung eine Linie
    if "measurement_id" in power_df.columns:
        for i, (mid, group) in enumerate(power_df.groupby("measurement_id")):
            color = _COLORS[i % len(_COLORS)]
            ax.plot(
                group["time_s"],
                group["power_watts"],
                linewidth=0.8,
                color=color,
                label=f"Messung {mid}",
                alpha=0.85,
            )
    else:
        ax.plot(
            power_df["time_s"],
            power_df["power_watts"],
            linewidth=0.8,
            color=_COLORS[0],
            label="Gemessen",
        )

    # P_idle Referenzlinie
    if idle_power > 0:
        ax.axhline(
            idle_power,
            linestyle="--",
            linewidth=1.0,
            color="gray",
            label=f"P_idle ({idle_power:.0f} W)",
        )

    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Leistung (W)")
    ax.set_title("GPU-Leistung über Zeit")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_routing_distribution(
    samples_df: pd.DataFrame,
    routing_stats: dict[str, int],
    thinking_stats: dict[str, int],
    output_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Horizontales Balkendiagramm: Anzahl Prompts pro Modell.

    Farbkodiert nach Thinking/Non-Thinking-Anteil.

    Args:
        samples_df: Flaches Sample-DataFrame mit ``model_id`` und optional
            ``enable_thinking``.
        routing_stats: ``{model_id: count}``.
        thinking_stats: ``{"thinking": n, "no_thinking": m}``.
        output_path: Ziel-PNG-Pfad.
    """
    if not samples_df.empty and {"model_id", "enable_thinking"}.issubset(
        samples_df.columns
    ):
        valid = _normalize_thinking_column(samples_df.dropna(subset=["model_id"]))
        if not valid.empty:
            grouped = (
                valid.groupby(["model_id", "enable_thinking"])
                .size()
                .unstack(fill_value=0)
            )
            grouped = grouped.reindex(columns=[False, True], fill_value=0)
            models = _resolve_model_order(model_order, grouped.index.tolist())
            grouped = grouped.reindex(models, fill_value=0)

            non_thinking = grouped[False].to_numpy()
            thinking = grouped[True].to_numpy()
            totals = non_thinking + thinking
            if totals.sum() == 0:
                return

            fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.8)))
            ax.barh(
                models,
                non_thinking,
                color=_NON_THINKING_COLOR,
                edgecolor="white",
                height=0.6,
                label="Non-Thinking",
            )
            ax.barh(
                models,
                thinking,
                left=non_thinking,
                color=_THINKING_COLOR,
                edgecolor="white",
                height=0.6,
                label="Thinking",
            )

            label_offset = max(totals) * 0.01 if max(totals) > 0 else 0.5
            for model, total in zip(models, totals):
                ax.text(
                    total + label_offset,
                    model,
                    str(int(total)),
                    va="center",
                    fontsize=9,
                )

            total_thinking = int(thinking.sum())
            total = int(totals.sum())
            pct = (100 * total_thinking / total) if total > 0 else 0.0
            ax.set_title(
                f"Routing-Verteilung (Thinking: {pct:.0f}%, Non-Thinking: {100 - pct:.0f}%)"
            )
            ax.set_xlabel("Anzahl Prompts")
            ax.set_ylabel("Modell")
            if total_thinking > 0 and int(non_thinking.sum()) > 0:
                ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(output_path)
            make_writable(output_path)
            plt.close(fig)
            return

    if not routing_stats:
        return

    models = _resolve_model_order(model_order, list(routing_stats.keys()))
    counts = [routing_stats[m] for m in models]

    fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.8)))

    bars = ax.barh(models, counts, color=_COLORS[0], edgecolor="white", height=0.6)

    # Anzahl-Labels rechts an den Balken
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

    # Thinking-Anteil als Annotation
    total_thinking = thinking_stats.get("thinking", 0)
    total_no_thinking = thinking_stats.get("no_thinking", 0)
    total = total_thinking + total_no_thinking
    if total > 0:
        pct = 100 * total_thinking / total
        ax.set_title(
            f"Routing-Verteilung (Thinking: {pct:.0f}%, "
            f"Non-Thinking: {100 - pct:.0f}%)"
        )
    else:
        ax.set_title("Routing-Verteilung")

    ax.set_xlabel("Anzahl Prompts")
    ax.set_ylabel("Modell")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_workload_by_model(
    samples_df: pd.DataFrame,
    output_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Zwei Balkendiagramme: Anfragezahl und Output-Tokens pro Modell.

    Args:
        samples_df: Flaches DataFrame mit ``model_id``,
            ``output_tokens`` Spalten.
        output_path: Ziel-PNG-Pfad.
    """
    if (
        "model_id" not in samples_df.columns
        or "output_tokens" not in samples_df.columns
    ):
        return

    valid = samples_df.dropna(subset=["model_id", "output_tokens"]).copy()
    if valid.empty:
        return

    if "enable_thinking" in valid.columns:
        valid = _normalize_thinking_column(valid)

    if "enable_thinking" in valid.columns and not valid.empty:
        count_by_mode = (
            valid.groupby(["model_id", "enable_thinking"]).size().unstack(fill_value=0)
        )
        count_by_mode = count_by_mode.reindex(columns=[False, True], fill_value=0)
        tokens_by_mode = (
            valid.groupby(["model_id", "enable_thinking"])["output_tokens"]
            .sum()
            .unstack(fill_value=0)
        )
        tokens_by_mode = tokens_by_mode.reindex(columns=[False, True], fill_value=0)
        models = _resolve_model_order(model_order, count_by_mode.index.tolist())
        count_by_mode = count_by_mode.reindex(models, fill_value=0)
        tokens_by_mode = tokens_by_mode.reindex(models, fill_value=0)

        prompt_counts = count_by_mode.sum(axis=1).to_numpy()
        total_output_tokens = tokens_by_mode.sum(axis=1).to_numpy()
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

        axes[0].bar(
            models,
            count_by_mode[False].to_numpy(),
            color=_NON_THINKING_COLOR,
            edgecolor="white",
            width=0.6,
            label="Non-Thinking",
        )
        axes[0].bar(
            models,
            count_by_mode[True].to_numpy(),
            bottom=count_by_mode[False].to_numpy(),
            color=_THINKING_COLOR,
            edgecolor="white",
            width=0.6,
            label="Thinking",
        )
        axes[1].bar(
            models,
            tokens_by_mode[False].to_numpy(),
            color=_NON_THINKING_COLOR,
            edgecolor="white",
            width=0.6,
        )
        axes[1].bar(
            models,
            tokens_by_mode[True].to_numpy(),
            bottom=tokens_by_mode[False].to_numpy(),
            color=_THINKING_COLOR,
            edgecolor="white",
            width=0.6,
        )

        prompt_offset = (
            max(prompt_counts) * 0.01
            if len(prompt_counts) and max(prompt_counts) > 0
            else 0.1
        )
        token_offset = (
            max(total_output_tokens) * 0.01
            if len(total_output_tokens) and max(total_output_tokens) > 0
            else 1.0
        )
        for idx, total in enumerate(prompt_counts):
            axes[0].text(
                idx,
                total + prompt_offset,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for idx, total in enumerate(total_output_tokens):
            axes[1].text(
                idx,
                total + token_offset,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        axes[0].set_ylabel("Anfragen")
        axes[0].set_title("Arbeitslast pro Modell")
        axes[0].legend(loc="upper left")
        axes[1].set_xlabel("Modell")
        axes[1].set_ylabel("Output-Tokens")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(output_path)
        make_writable(output_path)
        plt.close(fig)
        return

    grouped = valid.groupby("model_id")["output_tokens"].agg(["sum", "mean", "count"])
    models = _resolve_model_order(model_order, grouped.index.tolist())
    grouped = grouped.reindex(models)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    prompt_counts = grouped["count"].values
    total_output_tokens = grouped["sum"].values
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(models))]

    prompt_bars = axes[0].bar(
        models, prompt_counts, color=colors, edgecolor="white", width=0.6
    )
    token_bars = axes[1].bar(
        models, total_output_tokens, color=colors, edgecolor="white", width=0.6
    )

    for bar, val in zip(prompt_bars, prompt_counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(prompt_counts) * 0.01,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(token_bars, total_output_tokens):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_output_tokens) * 0.01,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0].set_ylabel("Anfragen")
    axes[0].set_title("Arbeitslast pro Modell")
    axes[1].set_xlabel("Modell")
    axes[1].set_ylabel("Output-Tokens")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_accuracy_per_model(
    samples_df: pd.DataFrame,
    output_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Balkendiagramm: Genauigkeit (is_correct Mittelwert) pro Modell.

    Args:
        samples_df: Flaches DataFrame mit ``model_id``,
            ``is_correct`` Spalten.
        output_path: Ziel-PNG-Pfad.
    """
    if "model_id" not in samples_df.columns or "is_correct" not in samples_df.columns:
        return

    # Nur Zeilen mit valider Bewertung
    valid = samples_df.dropna(subset=["is_correct"])
    if valid.empty:
        return

    grouped = valid.groupby("model_id")["is_correct"].mean()
    models = _resolve_model_order(model_order, grouped.index.tolist())
    grouped = grouped.reindex(models)

    fig, ax = plt.subplots(figsize=(8, 5))

    if "enable_thinking" in valid.columns:
        valid = _normalize_thinking_column(valid)

    series_data: list[tuple[str, pd.Series, tuple[float, float, float]]] = [
        ("Gesamt", grouped, _OVERALL_COLOR)
    ]
    title = "Genauigkeit pro Modell"

    if (
        "enable_thinking" in valid.columns
        and not valid.empty
        and valid["enable_thinking"].nunique() > 1
    ):
        mode_grouped = (
            valid.groupby(["model_id", "enable_thinking"])["is_correct"]
            .mean()
            .unstack()
        )
        mode_grouped = mode_grouped.reindex(models)
        non_thinking = (
            mode_grouped[False]
            if False in mode_grouped.columns
            else pd.Series(index=models, dtype=float)
        )
        thinking = (
            mode_grouped[True]
            if True in mode_grouped.columns
            else pd.Series(index=models, dtype=float)
        )
        if not non_thinking.dropna().empty:
            series_data.append(("Non-Thinking", non_thinking, _NON_THINKING_COLOR))
        if not thinking.dropna().empty:
            series_data.append(("Thinking", thinking, _THINKING_COLOR))
        if len(series_data) > 1:
            title = "Genauigkeit pro Modell und Modus"

    x = np.arange(len(models))
    width = 0.75 / max(len(series_data), 1)
    max_accuracy = 0.0

    for idx, (label, series, color) in enumerate(series_data):
        offset = (idx - (len(series_data) - 1) / 2) * width
        values = [series.get(model, np.nan) for model in models]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="white",
            label=label,
        )
        for bar, acc in zip(bars, values):
            if pd.notna(acc):
                max_accuracy = max(max_accuracy, float(acc))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(acc) + 0.01,
                    f"{float(acc):.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_xlabel("Modell")
    ax.set_ylabel("Genauigkeit")
    ax.set_title(title)
    ax.set_ylim(0, max(1.1, max_accuracy + 0.12))
    if len(series_data) > 1:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_token_distribution(
    samples_df: pd.DataFrame,
    output_path: Path,
    is_routing: bool = False,
) -> None:
    """Histogramm: Verteilung der Output-Tokens.

    Falls ``model_id`` vorhanden, wird nach Modell aufgeteilt
    (gestapeltes Histogramm). Bei Router-Experimenten
    (``is_routing=True``) und vorhandener ``enable_thinking``-Spalte
    wird zusaetzlich zwischen Thinking- und Non-Thinking-Anfragen
    unterschieden: ein Subplot pro Modell mit zwei ueberlagerten
    Histogrammen.

    Args:
        samples_df: Flaches DataFrame mit ``output_tokens``,
            optional ``model_id``, optional ``enable_thinking``.
        output_path: Ziel-PNG-Pfad.
        is_routing: Falls True und ``enable_thinking`` vorhanden, wird
            nach Thinking-Modus pro Modell aufgeteilt.
    """
    if "output_tokens" not in samples_df.columns:
        return

    # Router-Experiment: Thinking vs. Non-Thinking pro Modell
    if (
        is_routing
        and "model_id" in samples_df.columns
        and "enable_thinking" in samples_df.columns
    ):
        valid = samples_df.dropna(subset=["output_tokens", "model_id"]).copy()
        valid = _normalize_thinking_column(valid)
        if valid.empty:
            return

        models = sorted(valid["model_id"].unique())
        n_models = len(models)
        if n_models == 0:
            return

        ncols = min(n_models, 3)
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )

        has_thinking_split = valid["enable_thinking"].nunique() > 1

        for idx, model in enumerate(models):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            model_data = valid[valid["model_id"] == model]

            if has_thinking_split:
                non_t = model_data[model_data["enable_thinking"] == False][
                    "output_tokens"
                ].values
                t = model_data[model_data["enable_thinking"] == True][
                    "output_tokens"
                ].values
                all_vals = (
                    np.concatenate([v for v in [non_t, t] if len(v) > 0])
                    if (len(non_t) + len(t)) > 0
                    else np.array([])
                )
                if len(all_vals) == 0:
                    ax.set_visible(False)
                    continue
                bins = np.linspace(all_vals.min(), all_vals.max() + 1, 31)
                if len(non_t):
                    ax.hist(
                        non_t,
                        bins=bins,
                        color=_NON_THINKING_COLOR,
                        alpha=0.65,
                        label=f"Non-Thinking (n={len(non_t)})",
                        edgecolor="white",
                        linewidth=0.4,
                    )
                if len(t):
                    ax.hist(
                        t,
                        bins=bins,
                        color=_THINKING_COLOR,
                        alpha=0.65,
                        label=f"Thinking (n={len(t)})",
                        edgecolor="white",
                        linewidth=0.4,
                    )
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.hist(
                    model_data["output_tokens"].values,
                    bins=30,
                    color=_COLORS[idx % len(_COLORS)],
                    edgecolor="white",
                    linewidth=0.4,
                )

            ax.set_title(model, fontsize=9)
            ax.set_xlabel("Output-Tokens")
            ax.set_ylabel("Anfragen")

        # Unbenutzte Subplots ausblenden
        for idx in range(n_models, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            "Verteilung der Output-Tokens nach Modell und Thinking-Modus",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        make_writable(output_path)
        plt.close(fig)
        return

    # Fallback: Gestapeltes Histogramm nach Modell (oder einfaches Histogramm)
    fig, ax = plt.subplots(figsize=(8, 5))

    if "model_id" in samples_df.columns:
        models = sorted(samples_df["model_id"].unique())
        data = [
            samples_df[samples_df["model_id"] == m]["output_tokens"].values
            for m in models
        ]
        colors = [_COLORS[i % len(_COLORS)] for i in range(len(models))]
        ax.hist(
            data,
            bins=30,
            stacked=True,
            color=colors,
            label=models,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.legend(loc="upper right")
    else:
        ax.hist(
            samples_df["output_tokens"],
            bins=30,
            color=_COLORS[0],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Output-Tokens")
    ax.set_ylabel("Anzahl Anfragen")
    ax.set_title("Verteilung der Output-Tokens")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_latency_by_model(
    samples_df: pd.DataFrame,
    output_path: Path,
    model_order: list[str] | None = None,
) -> None:
    """Boxplot: Anfrage-Latenzen pro Modell."""
    if "latency_seconds" not in samples_df.columns:
        return

    valid = samples_df.dropna(subset=["latency_seconds"])
    if valid.empty:
        return

    if "enable_thinking" in valid.columns:
        valid = _normalize_thinking_column(valid)

    if (
        {"model_id", "enable_thinking"}.issubset(valid.columns)
        and not valid.empty
        and valid["enable_thinking"].nunique() > 1
    ):
        models = _resolve_model_order(
            model_order, valid["model_id"].dropna().unique().tolist()
        )
        positions: list[float] = []
        data: list[list[float]] = []
        colors: list[tuple[float, float, float]] = []
        present_modes = [
            mode for mode in [False, True] if (valid["enable_thinking"] == mode).any()
        ]
        offsets = (
            np.linspace(-0.18, 0.18, num=len(present_modes)) if present_modes else []
        )

        for model_index, model in enumerate(models, start=1):
            model_rows = valid[valid["model_id"] == model]
            for offset, mode in zip(offsets, present_modes):
                mode_values = model_rows[model_rows["enable_thinking"] == mode][
                    "latency_seconds"
                ].tolist()
                if not mode_values:
                    continue
                data.append(mode_values)
                positions.append(model_index + float(offset))
                colors.append(_THINKING_COLOR if mode else _NON_THINKING_COLOR)

        if data:
            fig, ax = plt.subplots(figsize=(8, 5))
            box = ax.boxplot(data, patch_artist=True, positions=positions, widths=0.28)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(range(1, len(models) + 1))
            ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_xlabel("Modell")
            ax.set_ylabel("Latenz (s)")
            ax.set_title("Latenz pro Modell und Modus")
            ax.legend(
                handles=[
                    Patch(
                        facecolor=_NON_THINKING_COLOR, alpha=0.7, label="Non-Thinking"
                    ),
                    Patch(facecolor=_THINKING_COLOR, alpha=0.7, label="Thinking"),
                ],
                loc="upper left",
            )
            fig.tight_layout()
            fig.savefig(output_path)
            make_writable(output_path)
            plt.close(fig)
            return

    if "model_id" in valid.columns:
        grouped = valid.groupby("model_id")
        labels = _resolve_model_order(model_order, list(grouped.groups.keys()))
        data = [group["latency_seconds"].tolist() for _, group in grouped]
        data = [
            valid[valid["model_id"] == label]["latency_seconds"].tolist()
            for label in labels
        ]
    else:
        labels = ["Messung"]
        data = [valid["latency_seconds"].tolist()]

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(
        box["boxes"],
        [_COLORS[i % len(_COLORS)] for i in range(len(labels))],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Modell")
    ax.set_ylabel("Latenz (s)")
    ax.set_title("Latenz pro Modell")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_thinking_overhead(
    samples_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Vergleicht Thinking und Non-Thinking fuer Latenz und Output-Tokens."""
    required_columns = {"enable_thinking", "latency_seconds", "output_tokens"}
    if not required_columns.issubset(samples_df.columns):
        return

    valid = samples_df.dropna(
        subset=["enable_thinking", "latency_seconds", "output_tokens"]
    )
    if valid.empty or valid["enable_thinking"].nunique() < 2:
        return

    latency_groups = [
        valid[valid["enable_thinking"] == False]["latency_seconds"].tolist(),
        valid[valid["enable_thinking"] == True]["latency_seconds"].tolist(),
    ]
    token_groups = [
        valid[valid["enable_thinking"] == False]["output_tokens"].tolist(),
        valid[valid["enable_thinking"] == True]["output_tokens"].tolist(),
    ]
    if not latency_groups[0] or not latency_groups[1]:
        return

    labels = ["Non-Thinking", "Thinking"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    latency_box = axes[0].boxplot(latency_groups, patch_artist=True, tick_labels=labels)
    token_box = axes[1].boxplot(token_groups, patch_artist=True, tick_labels=labels)
    for patch, color in zip(latency_box["boxes"], [_COLORS[0], _COLORS[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for patch, color in zip(token_box["boxes"], [_COLORS[0], _COLORS[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_ylabel("Latenz (s)")
    axes[0].set_title("Latenz nach Thinking-Modus")
    axes[1].set_ylabel("Output-Tokens")
    axes[1].set_title("Output-Tokens nach Thinking-Modus")

    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_scheduler_event_counts(
    measurement_data: dict,
    output_path: Path,
) -> None:
    """Gestapeltes Balkendiagramm: Scheduler-Ereignisse nach Aktion und Phase."""
    events = measurement_data.get("scheduler_events", [])
    if not events:
        return

    events_df = pd.DataFrame(events)
    if events_df.empty or "action" not in events_df.columns:
        return

    if "phase" not in events_df.columns:
        events_df["phase"] = "unknown"

    counts = pd.crosstab(events_df["action"], events_df["phase"])
    if counts.empty:
        return

    counts = counts.sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[_COLORS[i % len(_COLORS)] for i in range(len(counts.columns))],
    )

    ax.set_xlabel("Aktion")
    ax.set_ylabel("Anzahl Ereignisse")
    ax.set_title("Scheduler-Ereignisse nach Phase")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)


def plot_eq_score_summary(
    measurements_data: list[dict],
    output_path: Path,
) -> None:
    """Balkendiagramm: EQ-Score pro Messung.

    EQ-Score = Qualitaet / dynamische Energie (Kilowattstunden).
    Hoeherer Wert = effizienter.

    Args:
        measurements_data: Liste von Mess-Dicts mit EQ-relevanten Feldern.
        output_path: Ziel-PNG-Pfad.
    """
    labels = []
    eq_scores = []

    multiple_measurements = len(measurements_data) > 1
    for measurement in measurements_data:
        measurement_id = measurement.get("measurement_id", "?")
        eq = compute_measurement_eq_score(measurement)
        if eq is None:
            continue

        if multiple_measurements:
            labels.append(f"Messung {measurement_id}")
        else:
            labels.append("Messung")
        eq_scores.append(eq)

    if not eq_scores:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [_COLORS[i % len(_COLORS)] for i in range(len(labels))]
    bars = ax.bar(labels, eq_scores, color=colors, edgecolor="white", width=0.5)
    mean_score = float(np.mean(eq_scores))

    # Wert-Labels
    for bar, score in zip(bars, eq_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(eq_scores) * 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if len(eq_scores) > 1:
        ax.axhline(
            mean_score,
            color="#444444",
            linestyle="--",
            linewidth=1.1,
            label=f"Mean {mean_score:.3f}",
        )
        ax.text(
            len(labels) - 0.5,
            mean_score + max(eq_scores) * 0.015,
            f"Mean {mean_score:.3f}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Messung")
    ax.set_ylabel(f"EQ-Score (Qualitaet / {EQ_SCORE_ENERGY_UNIT})")
    ax.set_title("Effizienz-Qualitäts-Score (EQ) pro Messung")
    ax.set_ylim(0, max(eq_scores + [mean_score]) * 1.18)
    fig.tight_layout()
    fig.savefig(output_path)
    make_writable(output_path)
    plt.close(fig)
