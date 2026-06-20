"""Intelligenter Modell-Scheduler für Multi-Modell LLM-Inferenz.

Verwaltet den Lebenszyklus von Modellen (RUNNING/SLEEPING/STOPPED)
innerhalb eines begrenzten VRAM- und RAM-Budgets mit Per-GPU-Tracking.

Kernfunktionen:
    - Per-GPU VRAM-Tracking: Jede GPU wird einzeln verwaltet
    - Auto-TP: Tensor Parallelism automatisch aus Modell-VRAM vs GPU-VRAM
    - Dynamisches TP: Pro Batch angepasst (1 Modell → alle GPUs, 2+ → TP=1)
    - GPU-Zuweisung: TP=1 auf GPU mit meistem freien VRAM, TP>1 über alle GPUs
    - Smart Preload: Lädt kleinste Modelle verteilt über GPUs
    - LRU Eviction: Offloaded/Stoppt am längsten unbenutzte Modelle per GPU
    - Execution Planning: Optimiert Ausführungsreihenfolge (RUNNING > SLEEPING > STOPPED)

Referenz: Scheduler-Design inspiriert von OS-Schedulern (LRU-Eviction)
und dem Avengers Paper (arXiv:2505.19797) für Multi-Modell-Orchestrierung.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import subprocess

from src.config import (
    get_hardware,
    get_loadable_models,
    get_router_runtime_config,
    get_scheduler_config,
    get_server_config,
)
from src.inference.client import VLLMClient
from src.inference.server import VLLMServerManager
from src.scheduler.model_state import GPUState, ManagedModel, ModelState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Schwellwert für Auto-TP: Modell braucht TP>1 wenn es mehr als
# 90% des VRAM einer einzelnen GPU belegt
AUTO_TP_THRESHOLD = 0.9

# Zusätzlicher Headroom für den Start eines neuen vLLM-Servers.
# vLLM benötigt beim Engine-Init mehr Speicher als im stabilen RUNNING-Zustand,
# z.B. für CUDA-Kontexte, Profiling und temporäre Workspace-Allokationen.
DENSE_STARTUP_HEADROOM_GB = 10.0
MOE_STARTUP_HEADROOM_GB = 14.0
STARTUP_HEADROOM_REFERENCE_MAX_MODEL_LEN = 32768
DENSE_MIN_STARTUP_HEADROOM_GB = 0.25
MOE_MIN_STARTUP_HEADROOM_GB = 1.0


@dataclass
class SchedulerEvent:
    """Protokolliert eine Scheduler-Aktion (für Logging und Analyse)."""

    timestamp: datetime
    action: str  # "start", "sleep", "wake", "stop"
    model_id: str
    details: str = ""


@dataclass
class ExecutionGroup:
    """Eine Gruppe von Prompts die das gleiche Modell + Thinking-Modus nutzen."""

    model_id: str
    enable_thinking: bool
    indices: list[int]  # Original-Indices im Batch
    priority: int = 0  # 0=RUNNING, 1=SLEEPING, 2=STOPPED
    gpu_ids: list[int] = field(default_factory=list)


class ModelScheduler:
    """Smart VRAM/RAM-Manager für Multi-Modell-Inferenz.

    Verwaltet den Lebenszyklus (RUNNING/SLEEPING/STOPPED) aller
    Zielmodelle innerhalb konfigurierbarer Hardware-Grenzen.
    Trackt VRAM pro GPU und weist Modelle intelligent zu.

    Nicht zuständig für den Router -- der läuft separat als
    offline vllm.LLM und wird nicht vom Scheduler verwaltet.
    """

    def __init__(
        self,
        config: dict,
        server_manager: VLLMServerManager,
        router_vram_override: float | None = None,
        nvlink_available: bool = False,
    ):
        self.config = config
        self.server_manager = server_manager
        self.nvlink_available = nvlink_available

        self.hw = get_hardware(config)
        self.scheduler_cfg = get_scheduler_config(config)
        self.server_cfg = get_server_config(config)
        self._kv_cache_reserve_gb = self.scheduler_cfg.kv_cache_reserve_gb

        # Router-VRAM abziehen (immer belegt, auf GPU 0).
        # Der Offline-Router nutzt eine eigene Budget-Herleitung, damit die
        # Scheduler-Reservation mit dem tatsächlichen vLLM-Budget übereinstimmt.
        # Bei router_vram_override=0 (z.B. Random-Routing) wird kein VRAM reserviert.
        if router_vram_override is not None:
            self.router_vram_gb = router_vram_override
        else:
            self.router_vram_gb = get_router_runtime_config(config).budget_gb

        # Inference-Einstellungen für Clients
        self._vllm_cfg = config.get("inference", {}).get("vllm", {})

        # Per-GPU State aufbauen
        self.gpus: list[GPUState] = []
        for i in range(self.hw.num_gpus):
            reserved = self.router_vram_gb if i == 0 else 0.0
            self.gpus.append(
                GPUState(
                    gpu_id=i,
                    vram_total_gb=self.hw.per_gpu_vram_gb,
                    reserved_gb=reserved,
                )
            )

        # Verwaltete Modelle initialisieren
        self.models: dict[str, ManagedModel] = {}
        self._clients: dict[str, VLLMClient] = {}
        self._events: list[SchedulerEvent] = []
        self._next_port = self.server_cfg.base_port

        # Ladbare Modelle aus Config aufbauen + Auto-TP (Schwelle 90% pro GPU)
        loadable = get_loadable_models(config)

        for model_info in loadable:
            port = self._next_port
            self._next_port += 1

            tp = 1
            while (
                model_info.vram_gb / tp
            ) > self.hw.per_gpu_vram_gb * AUTO_TP_THRESHOLD:
                tp *= 2
                if tp > self.hw.num_gpus:
                    tp = self.hw.num_gpus
                    break

            vram_per_gpu = model_info.vram_gb / tp

            self.models[model_info.id] = ManagedModel(
                id=model_info.id,
                name=model_info.name,
                vram_gb=model_info.vram_gb,
                tier=model_info.tier,
                port=port,
                sampling=model_info.sampling,
                sampling_nothinking=model_info.sampling_nothinking,
                tensor_parallel_size=tp,
                vram_per_gpu_gb=vram_per_gpu,
                preferred_gpu=model_info.gpu,
            )

        logger.info(
            f"Scheduler initialisiert: {len(self.models)} ladbare Modelle, "
            f"{self.hw.num_gpus} GPU(s) a {self.hw.per_gpu_vram_gb}GB"
        )
        for m in self.models.values():
            logger.info(
                f"  {m.id}: {m.vram_gb}GB (TP={m.tensor_parallel_size}, "
                f"{m.vram_per_gpu_gb:.1f}GB/GPU)"
            )

    # =========================================================================
    # VRAM/RAM Budget (Per-GPU)
    # =========================================================================

    def available_vram_on_gpu(self, gpu_id: int) -> float:
        """Berechnet aktuell freien VRAM auf einer bestimmten GPU."""
        gpu = self.gpus[gpu_id]
        used = gpu.reserved_gb
        for mid in gpu.running_models:
            m = self.models[mid]
            used += self._running_footprint_per_gpu(m)
        return max(0.0, gpu.vram_total_gb - used)

    def available_vram(self) -> float:
        """Berechnet aktuell freien VRAM über alle GPUs (für Logging)."""
        return sum(self.available_vram_on_gpu(i) for i in range(self.hw.num_gpus))

    def available_ram(self) -> float:
        """Berechnet aktuell verfügbaren RAM für Offloading."""
        used = sum(
            m.vram_gb for m in self.models.values() if m.state == ModelState.SLEEPING
        )
        return max(0.0, self.hw.ram_for_offloading_gb - used)

    def _vram_needed_per_gpu(self, model: ManagedModel) -> float:
        """VRAM-Bedarf eines Modells pro GPU: Weights + KV-Cache + CUDA-Context."""
        kv_per_gpu = self._kv_cache_reserve_gb / model.tensor_parallel_size
        return (
            model.vram_per_gpu_gb
            + kv_per_gpu
            + self.scheduler_cfg.cuda_context_overhead_gb
        )

    def _running_footprint_per_gpu(self, model: ManagedModel) -> float:
        """Aktuell angenommener Live-Footprint eines laufenden Modells pro GPU.

        Sobald ein Modell mindestens einmal erfolgreich gestartet wurde, nutzen wir
        die per ``nvidia-smi`` beobachtete Delta-Belegung. Bis dahin dient die
        statische Schätzung (Weights + KV + CUDA) als Fallback.
        """
        if model.observed_vram_per_gpu_gb is not None:
            return model.observed_vram_per_gpu_gb
        return self._vram_needed_per_gpu(model)

    def _query_live_gpu_usage_gb(self) -> dict[int, float] | None:
        """Liest die aktuell belegte VRAM-Menge pro physischer GPU via nvidia-smi."""
        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None

        if smi.returncode != 0:
            return None

        used_by_gpu: dict[int, float] = {}
        for line in smi.stdout.strip().splitlines():
            try:
                gpu_str, used_str = [
                    part.strip() for part in line.split(",", maxsplit=1)
                ]
                used_by_gpu[int(gpu_str)] = float(used_str) / 1024.0
            except ValueError:
                continue

        return used_by_gpu or None

    def _live_used_vram_on_gpu(self, gpu_id: int) -> float | None:
        """Gibt die aktuell belegte VRAM-Menge einer GPU in GB zurück."""
        usage = self._query_live_gpu_usage_gb()
        if usage is None:
            return None
        return usage.get(gpu_id)

    def _startup_headroom_gb(self, model: ManagedModel) -> float:
        """Zusätzlicher Headroom für sicheren Server-Start und Wake.

        Ein kürzeres max_model_len reduziert den Init-Peak typischerweise,
        aber ein konservativer Sockel für Fragmentierung, CUDA-Workspaces und
        Profiling bleibt erhalten.
        """
        headroom_override = self.config.get("scheduler", {}).get("startup_headroom_gb")
        if headroom_override is not None:
            headroom_gb = float(headroom_override)
            if headroom_gb < 0:
                raise ValueError("scheduler.startup_headroom_gb muss >= 0 sein.")
            return headroom_gb

        max_model_len = int(
            self._vllm_cfg.get(
                "max_model_len",
                STARTUP_HEADROOM_REFERENCE_MAX_MODEL_LEN,
            )
            or STARTUP_HEADROOM_REFERENCE_MAX_MODEL_LEN
        )
        context_scale = min(
            1.0,
            max_model_len / STARTUP_HEADROOM_REFERENCE_MAX_MODEL_LEN,
        )

        model_name = model.name.lower()
        model_id = model.id.lower()
        if "moe" in model_id or "a3b" in model_name or "a10b" in model_name:
            return max(
                MOE_MIN_STARTUP_HEADROOM_GB,
                MOE_STARTUP_HEADROOM_GB * context_scale,
            )
        return max(
            DENSE_MIN_STARTUP_HEADROOM_GB,
            DENSE_STARTUP_HEADROOM_GB * context_scale,
        )

    def _startup_budget_per_gpu(self, model: ManagedModel) -> float:
        """Geschätzter VRAM-Bedarf pro GPU während Init oder Wake."""
        return self._model_alloc_gb(model)

    def _capture_gpu_usage(self, gpu_ids: list[int]) -> dict[int, float] | None:
        """Liefert eine VRAM-Snapshot-Ansicht für die angegebenen GPUs."""
        usage = self._query_live_gpu_usage_gb()
        if usage is None:
            return None
        return {gpu_id: usage.get(gpu_id, 0.0) for gpu_id in gpu_ids}

    def _record_observed_footprint(
        self,
        model: ManagedModel,
        before_usage: dict[int, float] | None,
        after_usage: dict[int, float] | None,
    ) -> None:
        """Speichert den beobachteten Live-Footprint eines erfolgreich gestarteten Modells.

        Gemessen wird die Delta-Belegung der dem Modell zugewiesenen GPU(s) vor und
        nach dem erfolgreichen Serverstart. Das verbessert alle späteren Pack- und
        Eviction-Entscheidungen, weil nicht nur mit Gewichtsgrößen geplant wird.
        """
        if before_usage is None or after_usage is None:
            return

        deltas: list[float] = []
        for gpu_id in model.gpu_assignment:
            before = before_usage.get(gpu_id)
            after = after_usage.get(gpu_id)
            if before is None or after is None:
                continue
            delta = after - before
            if delta > 0.25:
                deltas.append(delta)

        if not deltas:
            return

        observed = max(deltas)
        if model.observed_vram_per_gpu_gb is None:
            model.observed_vram_per_gpu_gb = observed
        else:
            model.observed_vram_per_gpu_gb = max(
                model.observed_vram_per_gpu_gb,
                observed,
            )

        logger.info(
            f"  beobachteter Live-Footprint: {model.observed_vram_per_gpu_gb:.2f} GB/GPU"
        )

    def _effective_max_num_seqs(self) -> int | None:
        """Leitet max_num_seqs aus Batch-Größe und HTTP-Fan-out ab."""
        batch_size = self.config.get("experiment", {}).get("batch_size")
        max_concurrent = self.config.get("inference", {}).get(
            "max_concurrent_requests_per_model"
        )

        limits = []
        if isinstance(batch_size, int) and batch_size > 0:
            limits.append(batch_size)
        if isinstance(max_concurrent, int) and max_concurrent > 0:
            limits.append(max_concurrent)

        if not limits:
            return None
        return min(limits)

    # =========================================================================
    # GPU-Zuweisung
    # =========================================================================

    def _assign_gpu(self, model: ManagedModel) -> list[int]:
        """Bestimmt GPU-Zuweisung für ein Modell.

        TP>1: Die top-N GPUs mit meistem freien VRAM (N = tensor_parallel_size).
        TP=1 mit preferred_gpu: Bevorzugte GPU (Eviction macht ggf. Platz).
        TP=1 ohne preferred_gpu: GPU mit meistem freien VRAM (Best-Fit Decreasing).
        """
        if model.tensor_parallel_size > 1:
            # Sortiere GPUs nach freiem VRAM absteigend, nimm die besten N
            sorted_gpus = sorted(
                range(self.hw.num_gpus),
                key=lambda g: self.available_vram_on_gpu(g),
                reverse=True,
            )
            return sorted(sorted_gpus[: model.tensor_parallel_size])

        # TP=1: Bevorzugte GPU nutzen (Eviction macht Platz in _start_model)
        if model.preferred_gpu is not None:
            return [model.preferred_gpu]

        # Kein Pinning: Best-Fit Decreasing
        best_gpu = max(
            range(self.hw.num_gpus),
            key=lambda g: self.available_vram_on_gpu(g),
        )
        return [best_gpu]

    # =========================================================================
    # GPU-Memory-Utilization
    # =========================================================================

    def _model_alloc_gb(self, model: ManagedModel) -> float:
        """Budget eines einzelnen vLLM-Servers pro GPU inklusive Start-Headroom."""
        budget_gb = (
            model.vram_per_gpu_gb
            + self._kv_cache_reserve_gb / model.tensor_parallel_size
            + self.scheduler_cfg.cuda_context_overhead_gb
            + self._startup_headroom_gb(model)
        )
        return budget_gb

    def _used_by_others_on_gpu(self, gpu_id: int, model: ManagedModel) -> float:
        """Gebuchter VRAM auf einer GPU ohne das betrachtete Modell."""
        gpu = self.gpus[gpu_id]
        return gpu.reserved_gb + sum(
            self._running_footprint_per_gpu(self.models[mid])
            for mid in gpu.running_models
            if mid != model.id
        )

    def _compute_gpu_mem_util(self, model: ManagedModel) -> float:
        """Berechnet das vLLM-Prozessbudget für den Modellstart.

        vLLM interpretiert ``gpu_memory_utilization`` pro Prozess als
        ``total_vram * gpu_memory_utilization`` und verlangt, dass dieser Betrag
        beim Start frei ist. Bereits geladene Prozesse dürfen deshalb nicht in
        diesen Wert eingerechnet werden. Die kumulative Pack-Prüfung passiert
        separat in ``_ensure_room_for_alloc`` und der Preload-Planung.
        """
        max_gpu_util = float(self._vllm_cfg.get("gpu_memory_utilization", 0.95))
        startup_budget = self._startup_budget_per_gpu(model)
        return min(max_gpu_util, startup_budget / self.hw.per_gpu_vram_gb)

    def _create_client(self, model: ManagedModel) -> None:
        """Erstellt einen VLLMClient für ein laufendes Modell."""
        sampling = model.sampling
        snt = model.sampling_nothinking
        request_timeout_seconds = float(
            self.config.get("inference", {}).get("request_timeout_seconds", 600.0)
        )
        max_concurrent_requests = int(
            self.config.get("inference", {}).get(
                "max_concurrent_requests_per_model",
                32,
            )
        )
        server_info = self.server_manager.get_server_info(model.id)
        self._clients[model.id] = VLLMClient(
            base_url=f"http://{self.server_cfg.host}:{model.port}",
            model_name=model.name,
            temperature=sampling.get("temperature", 0.0),
            top_p=sampling.get("top_p", 0.95),
            top_k=sampling.get("top_k", 20),
            min_p=sampling.get("min_p", 0.0),
            presence_penalty=sampling.get("presence_penalty", 1.5),
            repetition_penalty=sampling.get("repetition_penalty", 1.0),
            temperature_nothinking=snt.get(
                "temperature", sampling.get("temperature", 0.0)
            ),
            top_p_nothinking=snt.get("top_p", sampling.get("top_p", 1.0)),
            presence_penalty_nothinking=snt.get(
                "presence_penalty", sampling.get("presence_penalty", 2.0)
            ),
            request_timeout_seconds=request_timeout_seconds,
            server_log_path=server_info.stderr_path if server_info else None,
            max_concurrent_requests=max_concurrent_requests,
        )

    # =========================================================================
    # Modell-Lifecycle
    # =========================================================================

    def _start_model(self, model_id: str) -> None:
        """Startet einen vLLM-Server für ein Modell (STOPPED -> RUNNING)."""
        model = self.models[model_id]
        if model.state == ModelState.RUNNING:
            return

        logger.info(f"[START] {model_id}: STOPPED → RUNNING")

        # GPU-Zuweisung bestimmen
        assigned_gpus = self._assign_gpu(model)
        needed_per_gpu = self._vram_needed_per_gpu(model)
        logger.info(
            f"  GPU-Zuweisung: {assigned_gpus}, "
            f"Bedarf/GPU: {needed_per_gpu:.2f} GB "
            f"(Weights: {model.vram_per_gpu_gb:.2f} + "
            f"KV: {self._kv_cache_reserve_gb / model.tensor_parallel_size:.2f} + "
            f"CUDA: {self.scheduler_cfg.cuda_context_overhead_gb:.2f})"
        )

        # VRAM freimachen auf jeder zugewiesenen GPU.
        # Wir evictieren bis das Modell laut Scheduler-Buchhaltung passt
        # (Weights + KV-Reserve + CUDA + bereits laufende Modelle ≤ gmu_cap × total).
        model.gpu_assignment = assigned_gpus
        for gpu_id in assigned_gpus:
            self._ensure_room_for_alloc(gpu_id, model)

        # GPU-Memory-Utilization aus dem Startbudget des Modells ableiten.
        gpu_mem_util = self._compute_gpu_mem_util(model)
        logger.info(
            f"  gpu_memory_utilization: {gpu_mem_util:.4f} ({gpu_mem_util*100:.1f}%)"
        )

        max_num_seqs = self._effective_max_num_seqs()
        before_usage = self._capture_gpu_usage(assigned_gpus)

        self.server_manager.start_server(
            model_id=model.id,
            model_name=model.name,
            port=model.port,
            gpu_memory_utilization=gpu_mem_util,
            tensor_parallel_size=model.tensor_parallel_size,
            dtype=self._vllm_cfg.get("dtype", "auto"),
            max_model_len=self._vllm_cfg.get("max_model_len"),
            enforce_eager=self._vllm_cfg.get("enforce_eager", False),
            gpu_ids=assigned_gpus,
            nvlink_available=self.nvlink_available,
            max_num_seqs=max_num_seqs,
        )

        after_usage = self._capture_gpu_usage(assigned_gpus)
        self._record_observed_footprint(model, before_usage, after_usage)

        model.state = ModelState.RUNNING
        model.last_used = datetime.now()

        # Auf GPU-State registrieren
        for gpu_id in assigned_gpus:
            self.gpus[gpu_id].running_models.add(model_id)

        self._log_event("start", model_id, f"GPU(s)={assigned_gpus}")

        # Client anlegen
        self._create_client(model)

    def _sleep_model(self, model_id: str) -> None:
        """Offloaded ein Modell in CPU-RAM (RUNNING -> SLEEPING)."""
        model = self.models[model_id]
        if model.state != ModelState.RUNNING:
            return

        logger.info(
            f"[SLEEP] {model_id}: RUNNING → SLEEPING (GPU(s) {model.gpu_assignment})"
        )

        # Prüfen ob RAM reicht
        if self.available_ram() < model.vram_gb:
            self._free_ram_for(model.vram_gb)

        self.server_manager.sleep_model(model_id, level=1)
        model.state = ModelState.SLEEPING

        # GPU-State aktualisieren: running -> sleeping
        for gpu_id in model.gpu_assignment:
            self.gpus[gpu_id].running_models.discard(model_id)
            self.gpus[gpu_id].sleeping_models.add(model_id)

        self._log_event("sleep", model_id)

    def _wake_model(self, model_id: str) -> None:
        """Weckt ein schlafendes Modell auf (SLEEPING -> RUNNING)."""
        model = self.models[model_id]
        if model.state != ModelState.SLEEPING:
            return

        logger.info(
            f"[WAKE] {model_id}: SLEEPING → RUNNING (GPU(s) {model.gpu_assignment})"
        )

        # Sicherstellen, dass laut Scheduler-Buchhaltung Platz ist.
        # Wake reaktiviert das Modell auf seinen ursprünglichen GPUs.
        for gpu_id in model.gpu_assignment:
            self._ensure_room_for_alloc(gpu_id, model)

        self.server_manager.wake_model(model_id)
        model.state = ModelState.RUNNING
        model.last_used = datetime.now()

        # GPU-State aktualisieren: sleeping -> running
        for gpu_id in model.gpu_assignment:
            self.gpus[gpu_id].sleeping_models.discard(model_id)
            self.gpus[gpu_id].running_models.add(model_id)

        self._log_event("wake", model_id)

    def _stop_model(self, model_id: str) -> None:
        """Stoppt einen Server komplett (RUNNING/SLEEPING -> STOPPED)."""
        model = self.models[model_id]
        if model.state == ModelState.STOPPED:
            return

        logger.info(
            f"[STOP] {model_id}: {model.state.value} → STOPPED (GPU(s) {model.gpu_assignment})"
        )

        self.server_manager.stop_server(model_id)

        # Von allen GPU-States entfernen
        for gpu_id in model.gpu_assignment:
            self.gpus[gpu_id].running_models.discard(model_id)
            self.gpus[gpu_id].sleeping_models.discard(model_id)

        model.state = ModelState.STOPPED
        model.gpu_assignment = []
        self._log_event("stop", model_id)

        # Client entfernen
        if model_id in self._clients:
            self._clients[model_id].close()
            del self._clients[model_id]

    # =========================================================================
    # Eviction (LRU, Per-GPU)
    # =========================================================================

    def _ensure_room_for_alloc(self, gpu_id: int, model: ManagedModel) -> None:
        """Stellt sicher, dass das Modell laut Scheduler-Buchhaltung passt.

        Prüft: reserved + sum(alloc_running) + alloc(model) ≤ gmu_cap × total
        und evictet bei Bedarf das LRU-Modell auf dieser GPU.

        Eviction-Strategie:
            1. Sleep (Weights in CPU-RAM, schneller Wake möglich) wenn RAM reicht
            2. Sonst: kompletter Stop
        """
        gpu = self.gpus[gpu_id]
        max_gpu_util = float(self._vllm_cfg.get("gpu_memory_utilization", 0.95))
        target = gpu.vram_total_gb * max_gpu_util
        my_alloc = self._model_alloc_gb(model)

        while self._used_by_others_on_gpu(gpu_id, model) + my_alloc > target:
            candidates = [
                self.models[mid]
                for mid in gpu.running_models
                if mid != model.id and self.models[mid].state == ModelState.RUNNING
            ]
            if not candidates:
                raise RuntimeError(
                    f"Modell {model.id} ({my_alloc:.1f}GB) passt nicht auf "
                    f"GPU {gpu_id} "
                    f"(belegt: {self._used_by_others_on_gpu(gpu_id, model):.1f}GB, "
                    f"Limit: {target:.1f}GB) und es gibt nichts zu evictieren."
                )
            lru = min(candidates, key=lambda m: m.last_used)
            logger.info(
                f"  GPU {gpu_id}: Eviction {lru.id} (LRU) für Platz für {model.id}"
            )
            if self.available_ram() >= lru.vram_gb:
                self._sleep_model(lru.id)
            else:
                self._stop_model(lru.id)

    def _free_ram_for(self, needed_gb: float) -> None:
        """Gibt RAM frei durch Stoppen der am längsten schlafenden Modelle."""
        while self.available_ram() < needed_gb:
            sleeping = [
                m for m in self.models.values() if m.state == ModelState.SLEEPING
            ]
            if not sleeping:
                raise RuntimeError(
                    f"Kann {needed_gb:.1f}GB RAM nicht freigeben: "
                    f"keine schlafenden Modelle zum Stoppen"
                )

            lru = min(sleeping, key=lambda m: m.last_used)
            logger.info(f"RAM-Eviction: Stop {lru.id}")
            self._stop_model(lru.id)

    # =========================================================================
    # Public API
    # =========================================================================

    def preload(self) -> list[str]:
        """Smart Preload: Lädt Modelle parallel über GPUs.

        Phase 1: Greedy-Planung — GPU-Zuweisung und VRAM-Reservierung (sequentiell).
        Phase 2: Paralleler Server-Start — Modelle auf verschiedenen GPUs
                 gleichzeitig starten, auf gleicher GPU sequentiell (vLLM's
                 EngineCore verträgt keine gleichzeitige Initialisierung).
        Phase 3: Finalisierung — Clients anlegen und Events loggen.

        Returns:
            Liste der geladenen Modell-IDs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        models_by_size = sorted(
            self.models.values(), key=lambda m: m.vram_gb, reverse=True
        )

        # ── Phase 1: Planung (sequentiell) ─────────────────────────────────
        planned: list[ManagedModel] = []
        planned_usage_by_gpu: dict[int, float] = {
            gpu.gpu_id: gpu.reserved_gb for gpu in self.gpus
        }
        max_gpu_util = float(self._vllm_cfg.get("gpu_memory_utilization", 0.95))
        per_gpu_limit = self.hw.per_gpu_vram_gb * max_gpu_util

        for model in models_by_size:
            startup_budget = self._startup_budget_per_gpu(model)
            candidate_gpus = self._assign_gpu(model)
            fits = all(
                planned_usage_by_gpu[g] + startup_budget <= per_gpu_limit
                for g in candidate_gpus
            )

            if not fits:
                continue

            # GPU zuweisen und geplanten Live-Footprint reservieren, damit
            # nachfolgende Fits-Checks die bereits eingeplanten Server
            # berücksichtigen.
            assigned_gpus = candidate_gpus
            model.gpu_assignment = assigned_gpus
            for gpu_id in assigned_gpus:
                planned_usage_by_gpu[gpu_id] += startup_budget
            planned.append(model)

        if not planned:
            return []

        # Die Reservierungen aus Phase 1 dienen nur der Greedy-Planung.
        # Vor dem echten Start werden sie zurückgesetzt, damit im GPU-State nur
        # tatsächlich laufende Modelle auftauchen.
        for model in planned:
            for gpu_id in model.gpu_assignment:
                self.gpus[gpu_id].running_models.discard(model.id)

        all_catalog_planned = len(planned) == len(self.models)

        # ── Phase 2: Paralleler Start nach GPU gruppiert ───────────────────
        # Große Modelle zuerst starten (mehr Headroom beim vLLM-Profiling).
        gpu_queues: dict[int, list[ManagedModel]] = defaultdict(list)
        for model in planned:
            primary_gpu = model.gpu_assignment[0]
            gpu_queues[primary_gpu].append(model)
        for gpu_id in gpu_queues:
            gpu_queues[gpu_id].sort(key=lambda model: model.vram_gb, reverse=True)

        vllm_cfg = self._vllm_cfg
        max_gpu_util = float(self._vllm_cfg.get("gpu_memory_utilization", 0.95))
        max_num_seqs = self._effective_max_num_seqs()

        def _start_gpu_queue(
            items: list[ManagedModel],
        ) -> list[tuple[str, bool, Exception | None]]:
            """Startet Modelle auf einer GPU sequentiell.

            vLLM's EngineCore-Initialisierung (CUDA-Kontext, Profiling,
            KV-Cache-Allokation) verträgt keine gleichzeitigen Starts auf
            derselben GPU — daher bleibt die Reihenfolge hier seriell.
            Cross-GPU-Parallelität wird eine Ebene höher sichergestellt.
            """
            results = []
            started_on_gpu: list[ManagedModel] = []
            for model in items:
                try:
                    gpu_id = model.gpu_assignment[0]
                    live_used = self.gpus[gpu_id].reserved_gb + sum(
                        self._running_footprint_per_gpu(started)
                        for started in started_on_gpu
                    )
                    startup_budget = self._startup_budget_per_gpu(model)
                    if (
                        live_used + startup_budget
                        > self.hw.per_gpu_vram_gb * max_gpu_util
                    ):
                        raise RuntimeError(
                            f"kein sicherer Preload-Start auf GPU {gpu_id}: "
                            f"{live_used:.1f}GB bereits belegt + {startup_budget:.1f}GB "
                            f"für {model.id} > {self.hw.per_gpu_vram_gb * max_gpu_util:.1f}GB Limit"
                        )

                    gpu_mem_util = self._compute_gpu_mem_util(model)
                    logger.info(f"[START] {model.id}: STOPPED → RUNNING")
                    logger.info(
                        f"  GPU-Zuweisung: {model.gpu_assignment}, "
                        f"Bedarf/GPU: {self._vram_needed_per_gpu(model):.2f} GB "
                        f"(Weights: {model.vram_per_gpu_gb:.2f} + "
                        f"KV: {self._kv_cache_reserve_gb / model.tensor_parallel_size:.2f} + "
                        f"CUDA: {self.scheduler_cfg.cuda_context_overhead_gb:.2f})"
                    )
                    logger.info(
                        f"  gpu_memory_utilization: {gpu_mem_util:.4f} "
                        f"({gpu_mem_util * 100:.1f}%)"
                    )
                    before_usage = self._capture_gpu_usage(model.gpu_assignment)
                    self.server_manager.start_server(
                        model_id=model.id,
                        model_name=model.name,
                        port=model.port,
                        gpu_memory_utilization=gpu_mem_util,
                        tensor_parallel_size=model.tensor_parallel_size,
                        dtype=vllm_cfg.get("dtype", "auto"),
                        max_model_len=vllm_cfg.get("max_model_len"),
                        enforce_eager=vllm_cfg.get("enforce_eager", False),
                        gpu_ids=model.gpu_assignment,
                        nvlink_available=self.nvlink_available,
                        max_num_seqs=max_num_seqs,
                    )
                    after_usage = self._capture_gpu_usage(model.gpu_assignment)
                    self._record_observed_footprint(model, before_usage, after_usage)
                    started_on_gpu.append(model)
                    results.append((model.id, True, None))
                except Exception as e:
                    results.append((model.id, False, e))
            return results

        loaded: list[str] = []
        failed: list[tuple[str, Exception]] = []

        if len(gpu_queues) > 1:
            # Paralleler Start über verschiedene GPUs
            with ThreadPoolExecutor(max_workers=len(gpu_queues)) as executor:
                futures = {
                    executor.submit(_start_gpu_queue, items): gpu_id
                    for gpu_id, items in gpu_queues.items()
                }
                for future in as_completed(futures):
                    for model_id, success, error in future.result():
                        if success:
                            loaded.append(model_id)
                        else:
                            failed.append((model_id, error))
        else:
            # Nur eine GPU — kein Threading nötig
            for items in gpu_queues.values():
                for model_id, success, error in _start_gpu_queue(items):
                    if success:
                        loaded.append(model_id)
                    else:
                        failed.append((model_id, error))

        # ── Phase 3: Finalisierung ─────────────────────────────────────────
        for model_id, error in failed:
            model = self.models[model_id]
            for gpu_id in model.gpu_assignment:
                self.gpus[gpu_id].running_models.discard(model_id)
            model.gpu_assignment = []
            logger.warning(f"Preload fehlgeschlagen für {model_id}: {error}")

        for model_id in loaded:
            model = self.models[model_id]
            model.state = ModelState.RUNNING
            model.last_used = datetime.now()
            for gpu_id in model.gpu_assignment:
                self.gpus[gpu_id].running_models.add(model_id)
            self._create_client(model)
            self._log_event("start", model_id, f"GPU(s)={model.gpu_assignment}")
            logger.info(
                f"Preload: {model_id} ({model.vram_gb}GB) "
                f"auf GPU(s) {model.gpu_assignment}, "
                f"VRAM verbleibend: {self.available_vram():.1f}GB"
            )

        if len(loaded) == len(self.models):
            logger.info(
                "Alle Katalog-Modelle erfolgreich preloaded — Headroom für "
                "spätere Cold-Starts bleibt erhalten"
            )
        elif all_catalog_planned:
            logger.info(
                "Nicht alle Katalog-Modelle konnten gleichzeitig preloaded werden; "
                "Scheduler faellt auf selektiven Preload plus Sleep/Eviction zurueck"
            )

        return loaded

    def preload_with_router_overlap(
        self, router_load_fn: Callable[[], None]
    ) -> list[str]:
        """Preload mit Überlappung: GPU-1-Modelle starten während Router lädt.

        Der Router belegt GPU 0 (offline vLLM LLM). Modelle auf GPU 1+ haben
        keine Konkurrenz mit dem Router und können parallel zu dessen Load
        gestartet werden. Danach werden GPU-0-Modelle sequentiell gestartet.

        Ablauf:
            1. Planung (wie preload)
            2. GPU ≠ 0 Queues: große Modelle starten (non-blocking)
            3. router_load_fn() aufrufen (lädt Router auf GPU 0)
            4. GPU 0 Queue: Modelle sequentiell starten (blocking)
            5. GPU ≠ 0: auf Health-Checks warten
            6. Finalisierung

        Args:
            router_load_fn: Callable das den Router lädt (blockiert).

        Returns:
            Liste der geladenen Modell-IDs.
        """
        models_by_size = sorted(
            self.models.values(), key=lambda m: m.vram_gb, reverse=True
        )

        # ── Phase 1: Planung (identisch mit preload) ───────────────────────
        planned: list[ManagedModel] = []
        planned_usage_by_gpu: dict[int, float] = {
            gpu.gpu_id: gpu.reserved_gb for gpu in self.gpus
        }
        max_gpu_util = float(self._vllm_cfg.get("gpu_memory_utilization", 0.95))
        per_gpu_limit = self.hw.per_gpu_vram_gb * max_gpu_util

        for model in models_by_size:
            startup_budget = self._startup_budget_per_gpu(model)
            candidate_gpus = self._assign_gpu(model)
            fits = all(
                planned_usage_by_gpu[g] + startup_budget <= per_gpu_limit
                for g in candidate_gpus
            )
            if not fits:
                continue
            model.gpu_assignment = candidate_gpus
            for gpu_id in candidate_gpus:
                planned_usage_by_gpu[gpu_id] += startup_budget
            planned.append(model)

        if not planned:
            router_load_fn()
            return []

        for model in planned:
            for gpu_id in model.gpu_assignment:
                self.gpus[gpu_id].running_models.discard(model.id)

        all_catalog_planned = len(planned) == len(self.models)

        # GPU-Queues aufbauen (große zuerst)
        gpu_queues: dict[int, list[ManagedModel]] = defaultdict(list)
        for model in planned:
            primary_gpu = model.gpu_assignment[0]
            gpu_queues[primary_gpu].append(model)
        for gpu_id in gpu_queues:
            gpu_queues[gpu_id].sort(key=lambda model: model.vram_gb, reverse=True)

        vllm_cfg = self._vllm_cfg
        max_num_seqs = self._effective_max_num_seqs()
        startup_timeout = float(vllm_cfg.get("startup_timeout", 240))

        # Router-GPU ist immer GPU 0
        router_gpu = 0
        non_router_queues = {
            gid: items for gid, items in gpu_queues.items() if gid != router_gpu
        }
        router_queue = gpu_queues.get(router_gpu, [])

        # ── Phase 2a: GPU ≠ 0 Modelle spawnen (non-blocking) ──────────────
        # Große Modelle zuerst, sequentiell pro GPU (EngineCore-Constraint),
        # aber cross-GPU parallel. Nur das jeweils ERSTE Modell pro GPU wird
        # non-blocking gestartet; weitere müssen warten bis das erste steht.
        spawned_non_router: list[ManagedModel] = []

        def _spawn_non_router_first_models():
            """Startet das erste (größte) Modell auf jeder non-router GPU."""
            for gpu_id, items in non_router_queues.items():
                if not items:
                    continue
                model = items[0]
                gpu_mem_util = self._compute_gpu_mem_util(model)
                logger.info(f"[START] {model.id}: STOPPED → RUNNING")
                logger.info(
                    f"  GPU-Zuweisung: {model.gpu_assignment}, "
                    f"Bedarf/GPU: {self._vram_needed_per_gpu(model):.2f} GB "
                    f"(Weights: {model.vram_per_gpu_gb:.2f} + "
                    f"KV: {self._kv_cache_reserve_gb / model.tensor_parallel_size:.2f} + "
                    f"CUDA: {self.scheduler_cfg.cuda_context_overhead_gb:.2f})"
                )
                logger.info(
                    f"  gpu_memory_utilization: {gpu_mem_util:.4f} "
                    f"({gpu_mem_util * 100:.1f}%)"
                )
                self.server_manager.start_server(
                    model_id=model.id,
                    model_name=model.name,
                    port=model.port,
                    gpu_memory_utilization=gpu_mem_util,
                    tensor_parallel_size=model.tensor_parallel_size,
                    dtype=vllm_cfg.get("dtype", "auto"),
                    max_model_len=vllm_cfg.get("max_model_len"),
                    enforce_eager=vllm_cfg.get("enforce_eager", False),
                    gpu_ids=model.gpu_assignment,
                    nvlink_available=self.nvlink_available,
                    max_num_seqs=max_num_seqs,
                    wait=False,
                )
                spawned_non_router.append(model)

        _spawn_non_router_first_models()

        # ── Phase 2b: Router laden (blockiert, GPU 0) ──────────────────────
        router_load_fn()

        # ── Phase 2c: GPU 0 Queue sequentiell starten (blocking) ───────────
        loaded: list[str] = []
        failed: list[tuple[str, Exception]] = []

        for model in router_queue:
            try:
                gpu_mem_util = self._compute_gpu_mem_util(model)
                logger.info(f"[START] {model.id}: STOPPED → RUNNING")
                logger.info(
                    f"  GPU-Zuweisung: {model.gpu_assignment}, "
                    f"Bedarf/GPU: {self._vram_needed_per_gpu(model):.2f} GB "
                    f"(Weights: {model.vram_per_gpu_gb:.2f} + "
                    f"KV: {self._kv_cache_reserve_gb / model.tensor_parallel_size:.2f} + "
                    f"CUDA: {self.scheduler_cfg.cuda_context_overhead_gb:.2f})"
                )
                logger.info(
                    f"  gpu_memory_utilization: {gpu_mem_util:.4f} "
                    f"({gpu_mem_util * 100:.1f}%)"
                )
                before_usage = self._capture_gpu_usage(model.gpu_assignment)
                self.server_manager.start_server(
                    model_id=model.id,
                    model_name=model.name,
                    port=model.port,
                    gpu_memory_utilization=gpu_mem_util,
                    tensor_parallel_size=model.tensor_parallel_size,
                    dtype=vllm_cfg.get("dtype", "auto"),
                    max_model_len=vllm_cfg.get("max_model_len"),
                    enforce_eager=vllm_cfg.get("enforce_eager", False),
                    gpu_ids=model.gpu_assignment,
                    nvlink_available=self.nvlink_available,
                    max_num_seqs=max_num_seqs,
                )
                after_usage = self._capture_gpu_usage(model.gpu_assignment)
                self._record_observed_footprint(model, before_usage, after_usage)
                loaded.append(model.id)
            except Exception as e:
                failed.append((model.id, e))

        # ── Phase 2d: Auf non-router GPU-Modelle warten ───────────────────
        # Erst auf die bereits gespawnten ersten Modelle warten...
        for model in spawned_non_router:
            try:
                if not self.server_manager.wait_for_ready(
                    model.id, timeout=startup_timeout
                ):
                    self.server_manager.stop_server(model.id)
                    raise RuntimeError(
                        f"vLLM-Server für {model.id} nicht innerhalb von "
                        f"{startup_timeout}s gestartet."
                    )
                logger.info(
                    f"Server bereit: {model.id} auf "
                    f"{self.server_manager.get_base_url(model.id)}"
                )
                loaded.append(model.id)
            except Exception as e:
                failed.append((model.id, e))

        # ...dann die restlichen Modelle auf non-router GPUs sequentiell starten.
        for gpu_id, items in non_router_queues.items():
            for model in items[1:]:  # Erstes wurde oben schon gestartet
                try:
                    gpu_mem_util = self._compute_gpu_mem_util(model)
                    logger.info(f"[START] {model.id}: STOPPED → RUNNING")
                    logger.info(
                        f"  GPU-Zuweisung: {model.gpu_assignment}, "
                        f"Bedarf/GPU: {self._vram_needed_per_gpu(model):.2f} GB "
                        f"(Weights: {model.vram_per_gpu_gb:.2f} + "
                        f"KV: {self._kv_cache_reserve_gb / model.tensor_parallel_size:.2f} + "
                        f"CUDA: {self.scheduler_cfg.cuda_context_overhead_gb:.2f})"
                    )
                    logger.info(
                        f"  gpu_memory_utilization: {gpu_mem_util:.4f} "
                        f"({gpu_mem_util * 100:.1f}%)"
                    )
                    before_usage = self._capture_gpu_usage(model.gpu_assignment)
                    self.server_manager.start_server(
                        model_id=model.id,
                        model_name=model.name,
                        port=model.port,
                        gpu_memory_utilization=gpu_mem_util,
                        tensor_parallel_size=model.tensor_parallel_size,
                        dtype=vllm_cfg.get("dtype", "auto"),
                        max_model_len=vllm_cfg.get("max_model_len"),
                        enforce_eager=vllm_cfg.get("enforce_eager", False),
                        gpu_ids=model.gpu_assignment,
                        nvlink_available=self.nvlink_available,
                        max_num_seqs=max_num_seqs,
                    )
                    after_usage = self._capture_gpu_usage(model.gpu_assignment)
                    self._record_observed_footprint(model, before_usage, after_usage)
                    loaded.append(model.id)
                except Exception as e:
                    failed.append((model.id, e))

        # ── Phase 3: Finalisierung ─────────────────────────────────────────
        for model_id, error in failed:
            model = self.models[model_id]
            for gpu_id in model.gpu_assignment:
                self.gpus[gpu_id].running_models.discard(model_id)
            model.gpu_assignment = []
            logger.warning(f"Preload fehlgeschlagen für {model_id}: {error}")

        for model_id in loaded:
            model = self.models[model_id]
            model.state = ModelState.RUNNING
            model.last_used = datetime.now()
            for gpu_id in model.gpu_assignment:
                self.gpus[gpu_id].running_models.add(model_id)
            self._create_client(model)
            self._log_event("start", model_id, f"GPU(s)={model.gpu_assignment}")
            logger.info(
                f"Preload: {model_id} ({model.vram_gb}GB) "
                f"auf GPU(s) {model.gpu_assignment}, "
                f"VRAM verbleibend: {self.available_vram():.1f}GB"
            )

        if len(loaded) == len(self.models):
            logger.info(
                "Alle Katalog-Modelle erfolgreich preloaded — Headroom für "
                "spätere Cold-Starts bleibt erhalten"
            )
        elif all_catalog_planned:
            logger.info(
                "Nicht alle Katalog-Modelle konnten gleichzeitig preloaded werden; "
                "Scheduler faellt auf selektiven Preload plus Sleep/Eviction zurueck"
            )

        return loaded

    def ensure_model_running(self, model_id: str) -> None:
        """Stellt sicher, dass ein Modell RUNNING ist.

        Transitions:
            RUNNING:  Nur last_used aktualisieren.
            SLEEPING: Wake (schnell, von RAM).
            STOPPED:  Start (langsam, von Disk). Ggf. vorher VRAM freimachen.

        Args:
            model_id: ID des benötigten Modells.

        Raises:
            KeyError: Wenn das Modell nicht im Katalog ist.
            RuntimeError: Wenn nicht genug Ressourcen verfügbar sind.
        """
        if model_id not in self.models:
            raise KeyError(
                f"Modell '{model_id}' nicht im Scheduler-Katalog. "
                f"Verfügbar: {list(self.models.keys())}"
            )

        model = self.models[model_id]

        if model.state == ModelState.RUNNING:
            model.last_used = datetime.now()
            logger.debug(f"[ENSURE] {model_id}: bereits RUNNING")
            return

        if model.state == ModelState.SLEEPING:
            logger.info(f"[ENSURE] {model_id}: SLEEPING → Wake benötigt")
            self._wake_model(model_id)
            return

        # STOPPED -> RUNNING (mit Eviction falls nötig)
        logger.info(f"[ENSURE] {model_id}: STOPPED → Kaltstart benötigt")
        self._start_model(model_id)

    def get_execution_plan(
        self,
        routing_decisions: list[tuple[str, bool]],
    ) -> list[ExecutionGroup]:
        """Erstellt einen optimierten Ausführungsplan.

        Gruppiert Prompts nach (model_id, thinking) und sortiert
        nach Modell-Zustand: RUNNING zuerst, dann SLEEPING, dann STOPPED.
        Minimiert die Anzahl der Modell-Swaps.

        Args:
            routing_decisions: Liste von (model_id, enable_thinking) pro Prompt.

        Returns:
            Sortierte Liste von ExecutionGroups mit GPU-Zuweisung.
        """
        groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for i, (model_id, thinking) in enumerate(routing_decisions):
            groups[(model_id, thinking)].append(i)

        execution_groups = []
        for (model_id, thinking), indices in groups.items():
            model = self.models.get(model_id)
            if model is None:
                priority = 2  # Unbekannt = teuerster Fall
                gpu_ids = []
            elif model.state == ModelState.RUNNING:
                priority = 0
                gpu_ids = model.gpu_assignment
            elif model.state == ModelState.SLEEPING:
                priority = 1
                gpu_ids = model.gpu_assignment
            else:
                priority = 2
                gpu_ids = []  # Noch nicht zugewiesen

            execution_groups.append(
                ExecutionGroup(
                    model_id=model_id,
                    enable_thinking=thinking,
                    indices=indices,
                    priority=priority,
                    gpu_ids=gpu_ids,
                )
            )

        # Sortieren: RUNNING zuerst, dann SLEEPING, dann STOPPED
        execution_groups.sort(key=lambda g: g.priority)

        # Execution Plan loggen
        priority_labels = {0: "RUNNING", 1: "SLEEPING", 2: "STOPPED"}
        logger.info(
            f"[PLAN] {len(execution_groups)} Gruppen für {len(routing_decisions)} Prompts:"
        )
        for g in execution_groups:
            thinking_str = "Thinking" if g.enable_thinking else "Non-Thinking"
            state_str = priority_labels.get(g.priority, "?")
            logger.info(
                f"  {g.model_id} [{state_str}] {thinking_str}: "
                f"{len(g.indices)} Prompt(s), GPU(s) {g.gpu_ids}"
            )

        return execution_groups

    def get_client(self, model_id: str) -> VLLMClient:
        """Gibt den HTTP-Client für ein laufendes Modell zurück.

        Args:
            model_id: ID des Modells.

        Returns:
            VLLMClient-Instanz.

        Raises:
            RuntimeError: Wenn das Modell nicht RUNNING ist.
        """
        if model_id not in self._clients:
            raise RuntimeError(
                f"Kein Client für {model_id}. "
                f"Modell muss zuerst mit ensure_model_running() geladen werden."
            )
        return self._clients[model_id]

    def get_model_states(self) -> dict[str, str]:
        """Gibt den aktuellen Zustand aller Modelle zurück."""
        return {m.id: m.state.value for m in self.models.values()}

    @property
    def events(self) -> list[SchedulerEvent]:
        """Alle protokollierten Scheduler-Events."""
        return self._events.copy()

    def shutdown(self) -> None:
        """Stoppt alle Server und gibt Ressourcen frei."""
        logger.info("Scheduler Shutdown: Stoppe alle Modelle")
        for model_id in list(self.models.keys()):
            try:
                self._stop_model(model_id)
            except Exception as e:
                logger.error(f"Fehler beim Stoppen von {model_id}: {e}")

        # Router-VRAM-Reservierung freigeben (Router wird extern entladen)
        for gpu in self.gpus:
            gpu.reserved_gb = 0.0

        self.server_manager.shutdown_all()

    # =========================================================================
    # Internes Logging
    # =========================================================================

    def _log_event(self, action: str, model_id: str, details: str = "") -> None:
        self._events.append(
            SchedulerEvent(
                timestamp=datetime.now(),
                action=action,
                model_id=model_id,
                details=details,
            )
        )
        # VRAM-Überblick nach jeder Aktion
        for i, gpu in enumerate(self.gpus):
            running = list(gpu.running_models) or ["–"]
            sleeping = list(gpu.sleeping_models) or ["–"]
            avail = self.available_vram_on_gpu(i)
            logger.info(
                f"  GPU {i}: {avail:.1f}/{gpu.vram_total_gb:.0f} GB frei | "
                f"Running: {', '.join(running)} | Sleeping: {', '.join(sleeping)}"
            )
