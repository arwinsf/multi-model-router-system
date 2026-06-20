"""Konfigurationsmodul mit Profil-Unterstützung (local / uni).

Lädt YAML-Profile und stellt Hilfsfunktionen für den Modellkatalog,
Hardware-Ressourcen und Scheduler-Konfiguration bereit.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent

PROFILES = {
    "local": CONFIG_DIR / "settings_local.yaml",
    "uni": CONFIG_DIR / "settings_uni.yaml",
}

DEFAULT_PROFILE = "local"
_DEFAULT_ROUTER_KV_CACHE_BUDGET_GB = 1.0
_DEFAULT_ROUTER_MIN_PROCESS_BUDGET_GB = 3.0
_DEFAULT_ROUTER_MAX_MODEL_LEN = 8192


def _resolve_existing_hf_token_path(hf_path: str) -> str | None:
    """Findet einen bereits vorhandenen HuggingFace-Token auf dem System.

    In Docker setzen wir HF_HOME auf einen gemounteten Cache-Pfad. Wurde
    ``hf auth login`` jedoch zuvor ohne dieses Environment ausgefuehrt, liegt
    der Token typischerweise noch unter ``~/.cache/huggingface/token``.
    Dieser Helper uebernimmt den vorhandenen Token-Pfad, statt die Authentifi-
    zierung durch das neue HF_HOME unabsichtlich zu verlieren.
    """
    candidates = [
        Path(os.environ.get("HF_TOKEN_PATH", Path(hf_path) / "token")).expanduser(),
        Path(hf_path).expanduser() / "token",
        Path.home() / ".cache" / "huggingface" / "token",
        Path("/root/.cache/huggingface/token"),
        Path.home() / ".huggingface" / "token",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)
    return None


@dataclass
class ModelInfo:
    """Informationen zu einem Zielmodell aus dem Katalog."""

    id: str
    name: str  # HuggingFace-Pfad
    vram_gb: float
    tier: int  # 1 (einfach) bis 5 (komplex)
    description: str
    sampling: dict = field(
        default_factory=dict
    )  # Pro-Modell Sampling-Parameter (Thinking)
    sampling_nothinking: dict = field(
        default_factory=dict
    )  # Sampling-Parameter (Non-Thinking)
    gpu: int | None = None  # Bevorzugte GPU (statisches Pinning, optional)


@dataclass
class BaselineConfig:
    """Konfiguration für das Always-Large Baseline-Experiment."""

    model_id: str
    model_name: str  # HuggingFace-Pfad
    vram_gb: float
    enable_thinking: bool = True
    sampling: dict = field(default_factory=dict)  # Pro-Modell Sampling-Parameter


@dataclass
class SchedulerConfig:
    """Konfiguration für den Modell-Scheduler."""

    preload_strategy: str = "fill_small_first"
    eviction_policy: str = "lru"
    kv_cache_reserve_gb: float = 2.0
    cuda_context_overhead_gb: float = 0.5


@dataclass
class HardwareConfig:
    """Hardware-Ressourcen des Rechners."""

    per_gpu_vram_gb: float = 24.0
    ram_for_offloading_gb: float = 24.0
    num_gpus: int = 1

    @property
    def total_vram_gb(self) -> float:
        """Gesamter VRAM über alle GPUs."""
        return self.per_gpu_vram_gb * self.num_gpus


@dataclass
class ServerConfig:
    """vLLM-Server-Einstellungen."""

    base_port: int = 8001
    host: str = "127.0.0.1"


@dataclass
class RouterRuntimeConfig:
    """Abgeleitete Laufzeitparameter fuer den Offline-Router."""

    budget_gb: float
    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool


def load_config(profile: str | None = None) -> dict:
    """Lädt die Konfiguration für ein bestimmtes Profil.

    Profil-Auswahl (Priorität):
        1. Explizites `profile` Argument
        2. Umgebungsvariable LLM_ROUTING_PROFILE
        3. Default: "local"

    Args:
        profile: "local" oder "uni". Falls None, wird die
                 Umgebungsvariable oder der Default verwendet.

    Returns:
        Dictionary mit der Konfiguration.

    Raises:
        ValueError: Wenn das Profil unbekannt ist.
        FileNotFoundError: Wenn die Config-Datei nicht existiert.
    """
    if profile is None:
        profile = os.environ.get("LLM_ROUTING_PROFILE", DEFAULT_PROFILE)

    profile = profile.lower()

    if profile not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unbekanntes Profil: '{profile}'. Verfügbar: {available}")

    config_path = PROFILES[profile]

    if not config_path.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_profile"] = profile

    return config


def get_config(profile: str | None = None) -> dict:
    """Gibt die Konfiguration zurück."""
    return load_config(profile)


def get_hardware(config: dict) -> HardwareConfig:
    """Extrahiert die Hardware-Konfiguration."""
    hw = config.get("hardware", {})
    return HardwareConfig(
        per_gpu_vram_gb=hw.get("per_gpu_vram_gb", 24.0),
        ram_for_offloading_gb=hw.get("ram_for_offloading_gb", 24.0),
        num_gpus=hw.get("num_gpus", 1),
    )


def get_models(config: dict) -> list[ModelInfo]:
    """Extrahiert alle Modelle aus dem Katalog."""
    models = []
    for m in config.get("models", []):
        models.append(
            ModelInfo(
                id=m["id"],
                name=m["name"],
                vram_gb=m["vram_gb"],
                tier=m.get("tier", 1),
                description=m.get("description", ""),
                sampling=m.get("sampling", {}),
                sampling_nothinking=m.get("sampling_nothinking", {}),
                gpu=m.get("gpu"),
            )
        )
    return models


def get_effective_kv_cache_reserve_gb(config: dict) -> float:
    """Gibt die profilierte Scheduler-KV-Reserve in GB zurück.

    Die Reserve wird pro Profil explizit gepflegt und gilt für das dort
    konfigurierte Kontextfenster.
    """
    return get_scheduler_config(config).kv_cache_reserve_gb


def get_router_runtime_config(config: dict) -> RouterRuntimeConfig:
    """Leitet die vLLM-Laufzeitparameter fuer den Offline-Router ab.

    Der Router erzeugt nur sehr kurze Labels, profitiert aber lokal kaum von
    CUDA Graphs. Gleichzeitig ist ein rein footprint-basiertes Budget fuer
    kleine Modelle in vLLM 0.20 oft zu knapp fuer Engine-Init und KV-Cache.
    Daher bekommt der Router einen eigenen konservativen Budget-Floor.
    """
    router_cfg = config.get("router", {})
    hw = get_hardware(config)
    scheduler_cfg = get_scheduler_config(config)
    vllm_cfg = config.get("inference", {}).get("vllm", {})

    max_gpu_util = float(vllm_cfg.get("gpu_memory_utilization", 0.95))

    router_weights_gb = float(router_cfg.get("vram_gb", 2.5))
    kv_cache_budget_gb = float(
        router_cfg.get(
            "kv_cache_budget_gb",
            _DEFAULT_ROUTER_KV_CACHE_BUDGET_GB,
        )
    )
    min_process_budget_gb = float(
        router_cfg.get(
            "min_process_budget_gb",
            _DEFAULT_ROUTER_MIN_PROCESS_BUDGET_GB,
        )
    )

    requested_budget_gb = (
        router_weights_gb + kv_cache_budget_gb + scheduler_cfg.cuda_context_overhead_gb
    )
    budget_cap_gb = hw.per_gpu_vram_gb * max_gpu_util
    budget_gb = min(budget_cap_gb, max(requested_budget_gb, min_process_budget_gb))
    gpu_memory_utilization = min(max_gpu_util, budget_gb / hw.per_gpu_vram_gb)

    max_model_len = int(router_cfg.get("max_model_len", _DEFAULT_ROUTER_MAX_MODEL_LEN))
    if max_model_len < 1:
        raise ValueError("router.max_model_len muss >= 1 sein.")

    return RouterRuntimeConfig(
        budget_gb=budget_gb,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=bool(router_cfg.get("enforce_eager", True)),
    )


def get_loadable_models(config: dict) -> list[ModelInfo]:
    """Gibt nur Modelle zurück, die in den verfügbaren VRAM passen.

    Ein Modell ist ladbar wenn:
      - TP=1: weights + effektive_kv_reserve + cuda_overhead <= 90% VRAM
      - TP>1: (weights/tp) + (effektive_kv_reserve/tp) + cuda_overhead <= 90% VRAM
    """
    hw = get_hardware(config)
    scheduler_cfg = get_scheduler_config(config)
    effective_kv_reserve_gb = get_effective_kv_cache_reserve_gb(config)
    threshold = 0.9  # AUTO_TP_THRESHOLD

    all_models = get_models(config)
    loadable = []
    for m in all_models:
        # Prüfe für TP=1, TP=2, TP=4, ... bis num_gpus
        tp = 1
        fits = False
        while tp <= hw.num_gpus:
            vram_per_gpu = m.vram_gb / tp
            kv_per_gpu = effective_kv_reserve_gb / tp
            needed = vram_per_gpu + kv_per_gpu + scheduler_cfg.cuda_context_overhead_gb
            if needed <= hw.per_gpu_vram_gb * threshold:
                fits = True
                break
            tp *= 2
        if fits:
            loadable.append(m)
    return loadable


def get_model_by_id(config: dict, model_id: str) -> ModelInfo | None:
    """Findet ein Modell anhand seiner ID."""
    for m in get_models(config):
        if m.id == model_id:
            return m
    return None


def get_baseline(config: dict) -> BaselineConfig:
    """Extrahiert die Baseline-Konfiguration."""
    bl = config.get("baseline", {})
    return BaselineConfig(
        model_id=bl.get("model_id", ""),
        model_name=bl.get("model_name", ""),
        vram_gb=bl.get("vram_gb", 0.0),
        enable_thinking=bl.get("enable_thinking", True),
        sampling=bl.get("sampling", {}),
    )


def get_scheduler_config(config: dict) -> SchedulerConfig:
    """Extrahiert die Scheduler-Konfiguration."""
    sc = config.get("scheduler", {})
    return SchedulerConfig(
        preload_strategy=sc.get("preload_strategy", "fill_small_first"),
        eviction_policy=sc.get("eviction_policy", "lru"),
        kv_cache_reserve_gb=sc.get("kv_cache_reserve_gb", 2.0),
        cuda_context_overhead_gb=sc.get("cuda_context_overhead_gb", 0.5),
    )


def get_server_config(config: dict) -> ServerConfig:
    """Extrahiert die Server-Konfiguration."""
    sv = config.get("server", {})
    return ServerConfig(
        base_port=sv.get("base_port", 8001),
        host=sv.get("host", "127.0.0.1"),
    )


def is_docker() -> bool:
    """Erkennt ob der Code in einem Docker-Container läuft.

    Prüft in dieser Reihenfolge:
        1. Umgebungsvariable DOCKER_MODE=1/true
        2. Existenz von /.dockerenv (Docker-Standard)
        3. /workspace vorhanden (Uni-Docker-Setup als Fallback)
    """
    if os.environ.get("DOCKER_MODE", "").lower() in ("1", "true"):
        return True
    if Path("/.dockerenv").exists():
        return True
    if Path("/workspace").is_dir():
        return True
    return False


def setup_docker_environment(config: dict | None = None) -> None:
    """Setzt HF_HOME und TRANSFORMERS_CACHE für Docker-Umgebung.

    Wird nur aufgerufen wenn is_docker() True ist. Liest den
    HF-Cache-Pfad aus der YAML-Config (docker.hf_cache_path)
    oder nutzt den Default /workspace/hf-models. Falls ein HF-Token bereits
    an einem Standardpfad existiert, wird dessen Pfad explizit uebernommen,
    damit ein spaeter gesetztes HF_HOME die bestehende Authentifizierung nicht
    aushebelt.
    """
    if config is None:
        config = {}
    hf_path = config.get("docker", {}).get("hf_cache_path", "/workspace/hf-models")
    os.environ.setdefault("HF_HOME", hf_path)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_path)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_path) / "hub"))

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return

    token_path = _resolve_existing_hf_token_path(hf_path)
    if token_path is not None:
        os.environ.setdefault("HF_TOKEN_PATH", token_path)
