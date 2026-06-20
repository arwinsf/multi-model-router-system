"""Gemeinsame Metriken fuer Qualität, Energie und Verteilungen.

Der EQ-Score ist in dieser Arbeit kanonisch als

    Qualitaet / dynamische Messenergie in Kilowattstunden

definiert. Fuer Exact-Match Benchmarks entspricht Qualitaet der Accuracy als
Anteil im Bereich [0, 1]. Die kWh-Skalierung ist rein eine Darstellungsfrage,
damit die Werte groessenordnungsmaessig gut lesbar bleiben und gleichzeitig an
eine uebliche Energieeinheit anschliesst.
"""

from __future__ import annotations

from math import ceil, floor


EQ_SCORE_ENERGY_UNIT = "kWh"
EQ_SCORE_ENERGY_SCALE_JOULES = 3_600_000.0
EQ_SCORE_UNIT_KEY = "quality_per_kilowatthour"


def compute_accuracy(samples: list[dict]) -> float | None:
    """Berechnet Accuracy als Anteil korrekter Antworten.

    Samples ohne ``is_correct`` werden ignoriert. Wenn kein Sample eine
    auswertbare Genauigkeit enthaelt, wird ``None`` zurueckgegeben.
    """
    correct_values = [
        sample.get("is_correct")
        for sample in samples
        if sample.get("is_correct") is not None
    ]
    if not correct_values:
        return None

    return sum(1 for is_correct in correct_values if is_correct) / len(correct_values)


def compute_eq_score(
    quality_score: float | None,
    dynamic_energy_joules: float | None,
) -> float | None:
    """Berechnet den EQ-Score als Qualitaet pro Kilowattstunde."""
    if quality_score is None or dynamic_energy_joules is None:
        return None
    if dynamic_energy_joules <= 0:
        return None

    dynamic_energy_in_eq_unit = dynamic_energy_joules / EQ_SCORE_ENERGY_SCALE_JOULES
    return quality_score / dynamic_energy_in_eq_unit


def compute_percentile(values: list[float], percentile: float) -> float | None:
    """Berechnet ein lineares Perzentil fuer numerische Werte."""
    if not values:
        return None

    sorted_values = sorted(values)
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]

    rank = (len(sorted_values) - 1) * (percentile / 100)
    lower_idx = floor(rank)
    upper_idx = ceil(rank)
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]

    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    weight = rank - lower_idx
    return lower + (upper - lower) * weight


def compute_measurement_accuracy(measurement: dict) -> float | None:
    """Liest eine gespeicherte Accuracy oder berechnet sie aus Samples."""
    measurement_accuracy = measurement.get("measurement_accuracy")
    if measurement_accuracy is not None:
        return measurement_accuracy

    return compute_accuracy(measurement.get("samples", []))


def compute_measurement_eq_score(measurement: dict) -> float | None:
    """Berechnet den EQ-Score konsistent aus den Rohdaten einer Messung."""
    dynamic_energy_joules = measurement.get("measurement_dynamic_energy_joules")
    measurement_accuracy = compute_measurement_accuracy(measurement)
    if dynamic_energy_joules is not None and measurement_accuracy is not None:
        return compute_eq_score(measurement_accuracy, dynamic_energy_joules)

    stored_eq_score = measurement.get("measurement_eq_score")
    stored_eq_unit = measurement.get("measurement_eq_score_unit")
    if stored_eq_score is not None and stored_eq_unit == EQ_SCORE_UNIT_KEY:
        return stored_eq_score

    return None
