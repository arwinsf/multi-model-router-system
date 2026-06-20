"""LLM-Routing Energiemessung - Bachelorarbeit."""

# vLLM v1 startet den EngineCore-Subprozess via multiprocessing. Auf Linux
# ist der Default 'fork', was mit bereits initialisiertem CUDA im Parent
# zu "Cannot re-initialize CUDA in forked subprocess" fuehrt. Der offline
# vllm.LLM(...)-Pfad (Router) setzt diese Variable nicht selbst, also
# erzwingen wir 'spawn' bevor irgendetwas vLLM/Torch importiert.
import os as _os
from pathlib import Path as _Path

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - optional bei partiellen Installs
    _load_dotenv = None


def _bootstrap_project_env() -> None:
    """Laedt ``src/.env`` frueh und vereinheitlicht HF-Token-Aliase.

    Viele Entry-Points importieren spaeter HuggingFace-Bibliotheken oder rufen
    Docker-Bootstrap-Code auf. Damit diese Pfade ohne separates ``hf login``
    funktionieren, wird ein in ``src/.env`` gesetzter Token direkt in die von
    huggingface_hub erwarteten Variablennamen gespiegelt.
    """
    if _load_dotenv is not None:
        _load_dotenv(_Path(__file__).with_name(".env"))

    token = (
        _os.environ.get("HF_TOKEN")
        or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _os.environ.get("HUGGINGFACE_TOKEN")
    )
    if token:
        _os.environ.setdefault("HF_TOKEN", token)
        _os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    flashinfer_sampler = _os.environ.get("LLM_ROUTING_USE_FLASHINFER_SAMPLER", "0")
    _os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = flashinfer_sampler


_bootstrap_project_env()

_os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

__version__ = "0.1.0"
