"""Energiemessungsmodul für NVIDIA GPUs via NVML."""

from .base import EnergyMonitor, EnergyMeasurement
from .nvidia import NvidiaEnergyMonitor

__all__ = [
    "EnergyMonitor",
    "EnergyMeasurement",
    "NvidiaEnergyMonitor",
]


def create_energy_monitor(
    sampling_interval: float = 0.1,
) -> NvidiaEnergyMonitor:
    """Erstellt einen EnergyMonitor für NVIDIA GPUs.

    Alle verfügbaren GPUs werden automatisch erkannt.

    Args:
        sampling_interval: Messintervall in Sekunden.

    Returns:
        NvidiaEnergyMonitor Instanz.
    """
    return NvidiaEnergyMonitor(
        sampling_interval=sampling_interval,
    )
