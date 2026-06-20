"""NVIDIA GPU Energiemessung via nvidia-ml-py (NVML).

Optimiert für NVIDIA GPUs: RTX 3090, A100, H100, etc.
Erkennt automatisch alle verfügbaren GPUs und summiert deren Leistungswerte.

Features:
    - Automatische Erkennung aller NVIDIA GPUs
    - Echtzeit Power-Messung (Watts), bei Multi-GPU summiert
    - GPU-Informationen (Name, VRAM, Treiber, CUDA Compute Capability)
    - Temperatur und Auslastung

Installation:
    pip install nvidia-ml-py

Getestet mit:
    - Ubuntu 24.04 LTS
    - NVIDIA Driver 580+
    - RTX 3090, A100
"""

from .base import EnergyMonitor


class NvidiaEnergyMonitor(EnergyMonitor):
    """Energiemonitor für NVIDIA GPUs.

    Nutzt nvidia-ml-py (offizielle Python-Bindings für NVML)
    zum Auslesen der GPU-Leistung und Energiemessung.

    Erkennt automatisch alle verfügbaren GPUs und summiert deren
    Leistungsaufnahme. Kein manuelles Konfigurieren von GPU-Indices nötig.

    Unterstützte Funktionen:
        - Echtzeit Power-Messung (Watts), bei Multi-GPU summiert
        - Automatische Erkennung aller verfügbaren GPUs
        - GPU-Informationen (Name, VRAM, Treiber)
        - Temperatur und Auslastung
        - Persistenz-Modus Prüfung

    Beispiel:
        >>> monitor = NvidiaEnergyMonitor()
        >>> monitor.initialize()  # erkennt alle GPUs automatisch
        >>> print(f"Power: {monitor.get_power_watts()} W")
        >>> monitor.shutdown()
    """

    def __init__(self, sampling_interval: float = 0.1):
        super().__init__(sampling_interval=sampling_interval)
        self._nvml = None
        self._handle = None  # Primäre GPU (backward-compat)
        self._handles: dict[int, object] = {}  # Alle GPU-Handles
        self._nvlink_available: bool = False  # NVLink-Status zwischen GPUs

    def initialize(self) -> None:
        """Initialisiert NVML und erkennt automatisch alle verfügbaren GPUs.

        Alle gefundenen NVIDIA GPUs werden registriert und deren
        Leistungsaufnahme wird bei Messungen summiert.

        Raises:
            RuntimeError: Wenn pynvml nicht installiert oder keine GPU gefunden wird.
        """
        try:
            import pynvml

            self._nvml = pynvml
            pynvml.nvmlInit()

            # Alle verfügbaren GPUs automatisch erkennen
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                raise RuntimeError(
                    "Keine NVIDIA GPUs gefunden. Stelle sicher dass:\n"
                    "  1. Eine NVIDIA GPU installiert ist\n"
                    "  2. Der NVIDIA Treiber korrekt installiert ist\n"
                    "Prüfe mit: nvidia-smi"
                )

            # Handles für ALLE erkannten GPUs erstellen
            self.device_indices = list(range(device_count))
            for idx in self.device_indices:
                self._handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)

            # Primäre GPU (Index 0)
            self._handle = self._handles[0]

            # GPU-Info ausgeben (primäre GPU)
            info = self.get_gpu_info()
            print(f"NVIDIA GPU initialisiert: {info['name']}")
            print(f"  VRAM: {info['vram_total_gb']:.1f} GB")
            print(f"  Treiber: {info['driver_version']}")
            print(f"  CUDA: {info['cuda_version']}")

            if device_count > 1:
                print(
                    f"  Multi-GPU: {device_count} GPUs erkannt "
                    f"(Indices: {self.device_indices})"
                )
                for idx in self.device_indices:
                    handle = self._handles[idx]
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    print(f"    GPU {idx}: {name} ({mem.total / (1024**3):.1f} GB)")

                # NVLink-Erkennung: prüfe ob GPUs via NVLink verbunden sind
                self._nvlink_available = self._detect_nvlink()
                if self._nvlink_available:
                    print("  NVLink: aktiv (GPU-zu-GPU Verbindung erkannt)")
                else:
                    print("  NVLink: nicht verfügbar (PCIe-Verbindung)")

            # Persistenz-Modus prüfen (wichtig für genaue Messungen)
            try:
                persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self._handle)
                if persistence_mode == 0:
                    print("  WARNUNG: Persistenz-Modus deaktiviert!")
                    print("           Für genauere Messungen: sudo nvidia-smi -pm 1")
            except Exception:
                pass

        except ImportError:
            raise RuntimeError(
                "nvidia-ml-py nicht installiert.\n"
                "Installiere mit: pip install nvidia-ml-py"
            )
        except Exception as e:
            raise RuntimeError(f"NVML Initialisierung fehlgeschlagen: {e}")

    def _detect_nvlink(self) -> bool:
        """Erkennt ob NVLink zwischen GPUs aktiv ist.

        Prüft sequentiell die Link-Ports von GPU 0. A100 besitzt 12 NVLink-Ports;
        sobald einer aktiv ist (NVML_FEATURE_ENABLED), ist NVLink verfügbar.

        Returns:
            True wenn mindestens ein NVLink-Port aktiv ist, sonst False.
        """
        if not self._nvml or len(self._handles) < 2:
            return False
        try:
            # A100: 12 NVLink-Ports (Link-IDs 0–11); Ampere-Karten mit weniger Links
            # brechen beim ersten fehlenden Link ab – daher try/except pro Link.
            for link_id in range(12):
                try:
                    state = self._nvml.nvmlDeviceGetNvLinkState(
                        self._handles[0], link_id
                    )
                    if state:  # 1 = NVML_FEATURE_ENABLED
                        return True
                except self._nvml.NVMLError:
                    break  # Link nicht vorhanden, höhere IDs ebenfalls nicht
            return False
        except Exception:
            return False

    def shutdown(self) -> None:
        """Beendet NVML sauber."""
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            finally:
                self._nvml = None
                self._handle = None
                self._handles = {}

    def get_power_watts(self) -> float:
        """Liest die aktuelle GPU-Leistung in Watt.

        Bei Multi-GPU werden die Leistungswerte aller GPUs summiert.

        Returns:
            Aktuelle GPU-Leistung in Watt (Summe aller konfigurierten GPUs).

        Raises:
            RuntimeError: Wenn NVML nicht initialisiert ist.
        """
        if not self._nvml or not self._handles:
            raise RuntimeError("NVML nicht initialisiert! Rufe initialize() auf.")

        total_power = 0.0
        for handle in self._handles.values():
            # nvmlDeviceGetPowerUsage gibt Milliwatt zurück
            power_mw = self._nvml.nvmlDeviceGetPowerUsage(handle)
            total_power += power_mw / 1000.0
        return total_power

    def get_gpu_info(self) -> dict:
        """Gibt umfassende GPU-Informationen zurück (primäre GPU).

        Returns:
            Dictionary mit GPU-Informationen.
        """
        if not self._nvml or not self._handle:
            raise RuntimeError("NVML nicht initialisiert!")

        name = self._nvml.nvmlDeviceGetName(self._handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        memory_info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        driver_version = self._nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")

        # CUDA Version
        cuda_version = self._nvml.nvmlSystemGetCudaDriverVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10

        # Compute Capability
        try:
            major, minor = self._nvml.nvmlDeviceGetCudaComputeCapability(self._handle)
            compute_capability = f"{major}.{minor}"
        except Exception:
            compute_capability = "N/A"

        # Power Limit
        try:
            power_limit = (
                self._nvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
            )
        except Exception:
            power_limit = None

        # Temperatur
        try:
            temperature = self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temperature = None

        return {
            "name": name,
            "vram_total_gb": memory_info.total / (1024**3),
            "vram_used_gb": memory_info.used / (1024**3),
            "vram_free_gb": memory_info.free / (1024**3),
            "backend": "nvidia",
            "driver_version": driver_version,
            "cuda_version": f"{cuda_major}.{cuda_minor}",
            "compute_capability": compute_capability,
            "power_limit_watts": power_limit,
            "temperature_celsius": temperature,
            "num_gpus": len(self.device_indices),
            "nvlink_available": self._nvlink_available,
        }

    def get_temperature(self) -> float:
        """Liest die aktuelle GPU-Temperatur in Celsius (primäre GPU).

        Returns:
            GPU-Temperatur in Celsius.
        """
        if not self._nvml or not self._handle:
            raise RuntimeError("NVML nicht initialisiert!")

        return self._nvml.nvmlDeviceGetTemperature(
            self._handle, self._nvml.NVML_TEMPERATURE_GPU
        )

    def get_utilization(self) -> dict:
        """Liest die GPU- und Memory-Auslastung (primäre GPU).

        Returns:
            Dictionary mit GPU- und Memory-Auslastung in Prozent.
        """
        if not self._nvml or not self._handle:
            raise RuntimeError("NVML nicht initialisiert!")

        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        return {
            "gpu_percent": util.gpu,
            "memory_percent": util.memory,
        }
