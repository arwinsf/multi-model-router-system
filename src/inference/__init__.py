"""Inferenz-Modul für LLM-Generierung.

Bietet drei Modi:
    - VLLMInference: Offline-Modus (für Router, Baseline)
    - VLLMServerManager: Server-Lifecycle-Management mit Sleep/Wake
    - VLLMClient: HTTP-Client für Server-basierte Inferenz
"""

from .client import VLLMClient
from .llm import InferenceResult, VLLMInference
from .server import ServerInfo, VLLMServerManager

__all__ = [
    "InferenceResult",
    "VLLMInference",
    "VLLMServerManager",
    "VLLMClient",
    "ServerInfo",
]
