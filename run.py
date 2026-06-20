#!/usr/bin/env python3
"""Einstiegspunkt für die interaktive CLI.

Nutzung:
    python run.py

Funktioniert sowohl lokal als auch im Docker-Container.
Docker wird automatisch erkannt (/.dockerenv, DOCKER_MODE=1,
oder /workspace/hf-models vorhanden). Bei Docker-Erkennung
wird HF_HOME auf den konfigurierten Pfad gesetzt.
"""

import os
import sys
from pathlib import Path

# vLLM v1 EngineCore: 'spawn' erzwingen, bevor irgendein vLLM/Torch-Import
# CUDA initialisiert (sonst "Cannot re-initialize CUDA in forked subprocess").
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = os.environ.get(
    "LLM_ROUTING_USE_FLASHINFER_SAMPLER", "0"
)

# src/ zum Pfad hinzufügen, damit Imports funktionieren
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.config import is_docker, setup_docker_environment

# Docker-Erkennung: HF-Cache-Pfad setzen bevor andere Module geladen werden
if is_docker():
    try:
        from src.config import get_config

        config = get_config()
        setup_docker_environment(config)
    except Exception:
        # Fallback: Standard-Docker-Pfad ohne Config
        setup_docker_environment()

from src.cli import main

if __name__ == "__main__":
    main()
