"""Schnelltest für die Energiemessung.

Dieses Skript testet, ob die NVIDIA GPU-Energiemessung funktioniert,
ohne ein LLM zu laden.
"""

import sys
import time
from pathlib import Path

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.energy import create_energy_monitor


def test_energy_monitor():
    """Testet den EnergyMonitor."""
    config = get_config()
    gpu_config = config["gpu"]

    print("=" * 60)
    print("NVIDIA GPU Energiemessung Test")
    print("=" * 60)
    print(f"Sampling-Intervall: {gpu_config['sampling_interval']} s")
    print()

    # Monitor erstellen und initialisieren
    with create_energy_monitor(
        sampling_interval=gpu_config["sampling_interval"],
    ) as monitor:

        # GPU-Info anzeigen
        print("GPU-Informationen:")
        info = monitor.get_gpu_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

        # Einzelne Power-Messung
        print("Aktuelle Leistung:")
        for i in range(5):
            power = monitor.get_power_watts()
            print(f"  Messung {i+1}: {power:.1f} W")
            time.sleep(0.5)
        print()

        # Idle-Power messen (Baseline für Dynamic Power)
        monitor.measure_idle_power(duration=5.0)

        # Kontinuierliche Messung
        print("Kontinuierliche Messung (5 Sekunden):")
        monitor.start_measurement()

        # Simuliere Last (einfache Berechnung)
        start = time.time()
        while time.time() - start < 5:
            _ = sum(i**2 for i in range(100000))

        measurement = monitor.stop_measurement()

        print(f"  {measurement}")
        print(f"  Samples: {measurement.num_samples}")
        print(f"  Min Power: {measurement.min_power_watts:.1f} W")
        print(f"  Max Power: {measurement.max_power_watts:.1f} W")
        print()

    print("Test erfolgreich!")


if __name__ == "__main__":
    test_energy_monitor()
