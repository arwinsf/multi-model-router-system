"""Abstrakte Basisklasse für GPU-Energiemessung."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, TypeVar
import statistics
import threading
import time

T = TypeVar("T")


@dataclass
class EnergyMeasurement:
    """Ergebnis einer Energiemessung.

    Leistungsmodell:
        P_GPU(t) = P_idle + P_dyn(t)

        wobei:
        - P_idle: Grundverbrauch der GPU(s) (kein Modell geladen)
        - P_dyn(t): Zusätzliche Leistung durch aktive Berechnung
                    (inkl. Modell-Overhead im VRAM + Inferenz + Scheduling)

        P_idle wird einheitlich für beide Szenarien (Baseline und Router)
        als Baseline verwendet. Damit werden alle Kosten oberhalb des
        Hardware-Grundverbrauchs erfasst -- ein fairer Vergleich beider Szenarien.

        Bei Multi-GPU-Setups werden die Leistungswerte aller GPUs summiert.
    """

    # Gesamte gemessene Energie in Joule
    energy_joules: float

    # Dauer in Sekunden
    duration_seconds: float

    # Durchschnittliche Leistung in Watt (Summe aller GPUs)
    avg_power_watts: float

    # Maximale Leistung in Watt
    max_power_watts: float

    # Minimale Leistung in Watt
    min_power_watts: float

    # Anzahl der Samples
    num_samples: int

    # Idle-Power in Watt (GPU(s) ohne Last, kein Modell)
    idle_power_watts: float = 0.0

    # Dynamic Power = avg_power - P_idle (der relevante Messwert!)
    dynamic_power_watts: float = 0.0

    # Dynamic Energy = dynamic_power * duration (der relevante Messwert!)
    dynamic_energy_joules: float = 0.0

    # Alle gemessenen Leistungswerte (für Debugging)
    power_samples: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Dynamic Energy: {self.dynamic_energy_joules:.2f} J | "
            f"Dauer: {self.duration_seconds:.2f} s | "
            f"Dynamic Power: {self.dynamic_power_watts:.1f} W "
            f"(Total: {self.avg_power_watts:.1f} W, Idle: {self.idle_power_watts:.1f} W)"
        )


class EnergyMonitor(ABC):
    """Abstrakte Basisklasse für GPU-Energiemonitoring.

    Erkennt automatisch alle verfügbaren GPUs und summiert deren
    Leistungswerte. Bei Multi-GPU werden die Leistungswerte aller
    erkannten GPUs summiert.

    WICHTIG - Idle-Power-Korrektur:
        GPU-Energiemessung via NVIDIA NVML misst immer die
        GESAMTE GPU-Leistung, nicht nur die eines bestimmten Prozesses.
        Daher wird vor jedem Experiment die Idle-Power gemessen und
        von den Messwerten abgezogen.

        Während der Experimente sollten keine anderen GPU-lastigen
        Anwendungen (Streaming, Gaming, andere ML-Tasks) laufen!
    """

    # Warnung bei zu hoher Idle-Power (deutet auf Hintergrundlast hin)
    IDLE_POWER_WARNING_THRESHOLD = 50.0  # Watt

    def __init__(
        self,
        sampling_interval: float = 0.1,
    ):
        """Initialisiert den Energiemonitor.

        Args:
            sampling_interval: Intervall zwischen Messungen in Sekunden.
        """
        self.device_indices: list[int] = []  # Wird in initialize() gesetzt
        self.sampling_interval = sampling_interval
        self._is_measuring = False
        self._power_samples: list[float] = []
        self._measurement_thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._idle_power: float = 0.0  # Gemessene Idle-Power (kein Modell)

    @abstractmethod
    def initialize(self) -> None:
        """Initialisiert die GPU-Bibliothek."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Beendet die GPU-Bibliothek sauber."""
        pass

    @abstractmethod
    def get_power_watts(self) -> float:
        """Liest die aktuelle GPU-Leistung in Watt.

        Returns:
            Aktuelle Leistungsaufnahme in Watt.
        """
        pass

    @abstractmethod
    def get_gpu_info(self) -> dict:
        """Gibt Informationen über die GPU zurück.

        Returns:
            Dictionary mit GPU-Informationen (Name, VRAM, etc.).
        """
        pass

    def measure_idle_power(self, duration: float = 5.0, quiet: bool = False) -> float:
        """Misst die Idle-Power der GPU für die Baseline-Korrektur.

        Diese Methode sollte VOR jedem Experiment aufgerufen werden,
        um den Basis-Stromverbrauch der GPU zu ermitteln. Dieser wird
        dann von den Messwerten abgezogen, um die "Dynamic Power"
        (tatsächlicher Mehrverbrauch durch Inferenz) zu berechnen.

        Args:
            duration: Messdauer in Sekunden (Standard: 5s für stabilen Median).
            quiet: Wenn True, keine Ausgaben auf der Konsole.

        Returns:
            Median der Idle-Power in Watt.

        Raises:
            RuntimeWarning: Wenn die Idle-Power ungewöhnlich hoch ist.
        """
        if not quiet:
            print(f"\n{'='*60}")
            print("⚠️  IDLE-POWER-MESSUNG")
            print("=" * 60)
            print("Bitte stelle sicher, dass auf den GPUs nichts anderes läuft!")
            print(f"Die Messung dauert {duration:.0f} Sekunden - bitte warten...")
            print()

        samples = []
        start = time.time()

        while time.time() - start < duration:
            try:
                power = self.get_power_watts()
                samples.append(power)
            except Exception as e:
                if not quiet:
                    print(f"  Warnung: Messfehler: {e}")
            time.sleep(self.sampling_interval)

        if not samples:
            raise RuntimeError("Keine Idle-Power-Samples gesammelt!")

        # Median ist robuster gegen Ausreißer als Mittelwert
        idle_power = statistics.median(samples)
        self._idle_power = idle_power

        if not quiet:
            print(f"✅ Idle-Power gemessen: {idle_power:.1f} W")
            print(
                f"   (Min: {min(samples):.1f} W, Max: {max(samples):.1f} W, "
                f"Samples: {len(samples)})"
            )

            if idle_power > self.IDLE_POWER_WARNING_THRESHOLD * len(
                self.device_indices
            ):
                print(
                    f"\n⚠️  WARNUNG: Idle-Power ist hoch ({idle_power:.1f} W > "
                    f"{self.IDLE_POWER_WARNING_THRESHOLD:.0f} W)!"
                )
                print("   Möglicherweise läuft eine Hintergrund-Anwendung auf der GPU.")
                print("   Die Messergebnisse könnten verfälscht sein.")
            print("=" * 60 + "\n")

        return idle_power

    def get_idle_power(self) -> float:
        """Gibt die zuletzt gemessene Idle-Power zurück.

        Returns:
            Idle-Power in Watt, oder 0.0 wenn noch nicht gemessen.
        """
        return self._idle_power

    def _sample_power(self) -> None:
        """Interne Methode: Sammelt Leistungsmessungen in einem Thread."""
        while self._is_measuring:
            try:
                power = self.get_power_watts()
                self._power_samples.append(power)
            except Exception as e:
                print(f"Warnung: Fehler beim Messen der Leistung: {e}")
            time.sleep(self.sampling_interval)

    def start_measurement(self) -> None:
        """Startet die kontinuierliche Energiemessung."""
        if self._is_measuring:
            raise RuntimeError("Messung läuft bereits!")

        self._power_samples = []
        self._is_measuring = True
        self._start_time = time.perf_counter()

        self._measurement_thread = threading.Thread(
            target=self._sample_power, daemon=True
        )
        self._measurement_thread.start()

    def stop_measurement(self) -> EnergyMeasurement:
        """Stoppt die Messung und berechnet die Ergebnisse.

        Die Dynamic Power (tatsächlicher Mehrverbrauch) wird berechnet,
        indem die Idle-Power von der gemessenen Power abgezogen wird.

        Returns:
            EnergyMeasurement mit allen berechneten Werten inkl. Dynamic Power.
        """
        if not self._is_measuring:
            raise RuntimeError("Keine Messung aktiv!")

        self._is_measuring = False
        end_time = time.perf_counter()

        if self._measurement_thread:
            self._measurement_thread.join(timeout=1.0)

        duration = end_time - self._start_time

        if not self._power_samples:
            return EnergyMeasurement(
                energy_joules=0.0,
                duration_seconds=duration,
                avg_power_watts=0.0,
                max_power_watts=0.0,
                min_power_watts=0.0,
                num_samples=0,
                idle_power_watts=self._idle_power,
                dynamic_power_watts=0.0,
                dynamic_energy_joules=0.0,
                power_samples=[],
            )

        avg_power = sum(self._power_samples) / len(self._power_samples)
        energy = avg_power * duration

        # Dynamic Power: P_idle als einheitliche Baseline
        dynamic_power = max(0.0, avg_power - self._idle_power)
        dynamic_energy = dynamic_power * duration

        return EnergyMeasurement(
            energy_joules=energy,
            duration_seconds=duration,
            avg_power_watts=avg_power,
            max_power_watts=max(self._power_samples),
            min_power_watts=min(self._power_samples),
            num_samples=len(self._power_samples),
            idle_power_watts=self._idle_power,
            dynamic_power_watts=dynamic_power,
            dynamic_energy_joules=dynamic_energy,
            power_samples=self._power_samples.copy(),
        )

    def measure(
        self, func: Callable[..., T], *args, **kwargs
    ) -> tuple[T, EnergyMeasurement]:
        """Misst den Energieverbrauch während der Ausführung einer Funktion.

        Args:
            func: Die auszuführende Funktion.
            *args: Positionale Argumente für die Funktion.
            **kwargs: Keyword-Argumente für die Funktion.

        Returns:
            Tuple aus (Funktionsergebnis, EnergyMeasurement).
        """
        self.start_measurement()
        try:
            result = func(*args, **kwargs)
        finally:
            measurement = self.stop_measurement()

        return result, measurement

    def __enter__(self) -> "EnergyMonitor":
        """Context-Manager: Startet die Messung."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context-Manager: Beendet die Bibliothek."""
        self.shutdown()
