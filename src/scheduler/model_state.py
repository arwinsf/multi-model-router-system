"""Modell-Zustandsverwaltung für den Scheduler.

Definiert die möglichen Zustände eines verwalteten Modells,
GPU-Zustandstracking und die Übergangsregeln.

State Machine:
    STOPPED  ---start_server--->  RUNNING
    RUNNING  ---sleep(level=1)--> SLEEPING
    SLEEPING ---wake_up---------> RUNNING
    RUNNING  ---stop_server-----> STOPPED
    SLEEPING ---stop_server-----> STOPPED
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ModelState(Enum):
    """Zustand eines verwalteten Modells."""

    STOPPED = "stopped"
    """Kein Prozess, kein VRAM, kein RAM belegt."""

    SLEEPING = "sleeping"
    """Prozess lebt, Weights in CPU-RAM, VRAM freigegeben (vLLM Sleep Level 1)."""

    RUNNING = "running"
    """Prozess lebt, Weights in VRAM, bereit für Inferenz."""


@dataclass
class GPUState:
    """Zustand einer einzelnen physischen GPU.

    Trackt welche Modelle auf dieser GPU laufen oder schlafen,
    sowie den permanent reservierten VRAM (z.B. Router auf GPU 0).
    """

    gpu_id: int
    vram_total_gb: float
    reserved_gb: float = 0.0  # Permanent reserviert (z.B. Router auf GPU 0)
    running_models: set[str] = field(default_factory=set)
    sleeping_models: set[str] = field(default_factory=set)


@dataclass
class ManagedModel:
    """Laufzeitinformationen eines vom Scheduler verwalteten Modells.

    Wird zur Laufzeit geführt (nicht persistent). Enthält sowohl
    statische Informationen aus dem Modellkatalog als auch dynamischen
    Zustand des zugehörigen vLLM-Servers.
    """

    # Statisch (aus Config)
    id: str
    name: str  # HuggingFace-Pfad
    vram_gb: float  # Gesamt-VRAM-Bedarf
    tier: int  # 1 (einfach) bis 5 (komplex)
    port: int
    sampling: dict = field(default_factory=dict)  # Sampling-Parameter (Thinking-Modus)
    sampling_nothinking: dict = field(
        default_factory=dict
    )  # Sampling-Parameter (Non-Thinking-Modus)

    # Vom Scheduler berechnet (Auto-TP)
    tensor_parallel_size: int = 1  # Aktueller TP-Grad
    vram_per_gpu_gb: float = 0.0  # VRAM-Bedarf pro GPU (= vram_gb / tp)
    preferred_gpu: int | None = None  # Aus Config, None = dynamisch (Best-Fit)
    gpu_assignment: list[int] = field(default_factory=list)

    # Dynamisch (Laufzeit)
    state: ModelState = ModelState.STOPPED
    last_used: datetime = field(default_factory=datetime.now)
    observed_vram_per_gpu_gb: float | None = None
