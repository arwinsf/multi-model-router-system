"""Logging-Konfiguration."""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Modulweiter Pfad zur Temp-Log-Datei
_log_file_path: Path | None = None
_stderr_log_stream = None


def setup_logging(
    level: int = logging.INFO, log_file: str | Path | None = None
) -> logging.Logger:
    """Konfiguriert das Logging für die Experimente.

    Args:
        level: Logging-Level (z.B. logging.DEBUG, logging.INFO).
        log_file: Optionaler Pfad für eine Log-Datei.

    Returns:
        Konfigurierter Logger.
    """
    logger = logging.getLogger("llm_routing")
    logger.setLevel(level)

    # Verhindere doppelte Handler bei mehrfachen Aufrufen
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        from src.utils.data import make_writable

        make_writable(log_path)
        make_writable(log_path.parent)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Gibt einen Child-Logger des llm_routing-Loggers zurück.

    Args:
        name: Name des Moduls (typisch __name__).

    Returns:
        Logger-Instanz.
    """
    return logging.getLogger(f"llm_routing.{name}")


def setup_file_logging() -> Path:
    """Erstellt eine temporäre Log-Datei und fügt einen FileHandler hinzu.

    Wird einmal zu Beginn eines Experiments aufgerufen. Alle Log-Nachrichten
    werden zusätzlich zur Konsole in die Temp-Datei geschrieben. Zusätzlich
    wird ``stderr`` auf die gleiche Datei umgeleitet, damit Fehler aus
    Bibliotheken und Kindprozessen (z.B. vLLM EngineCore) im Experiment-Log
    erhalten bleiben.

    Returns:
        Pfad zur temporären Log-Datei.
    """
    global _log_file_path, _stderr_log_stream

    logger = logging.getLogger("llm_routing")

    # Tempfile erstellen (wird nicht automatisch gelöscht)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", prefix="llm_routing_", delete=False
    )
    _log_file_path = Path(tmp.name)
    tmp.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # stderr auf die gleiche Datei umleiten, damit Child-Prozesse ihre
    # Root-Cause nicht nur im Terminal, sondern auch im Artefakt hinterlassen.
    if _stderr_log_stream is None:
        _stderr_log_stream = open(_log_file_path, "a", encoding="utf-8", buffering=1)
        sys.stderr.flush()
        os.dup2(_stderr_log_stream.fileno(), sys.stderr.fileno())

    return _log_file_path


def get_log_file_path() -> Path | None:
    """Gibt den Pfad zur aktuellen temporären Log-Datei zurück."""
    return _log_file_path


def finalize_log(target_dir: Path, filename: str = "experiment.txt") -> Path | None:
    """Kopiert die temporäre Log-Datei in das Ergebnisverzeichnis.

    Flusht alle Handler bevor die Datei kopiert wird.

    Args:
        target_dir: Zielverzeichnis (typisch das Experiment-Ergebnis-Verzeichnis).
        filename: Name der Zieldatei.

    Returns:
        Pfad zur kopierten Log-Datei oder None wenn kein Log vorhanden.
    """
    if _log_file_path is None or not _log_file_path.exists():
        return None

    # Alle Handler flushen
    logger = logging.getLogger("llm_routing")
    for handler in logger.handlers:
        handler.flush()
    if _stderr_log_stream is not None:
        _stderr_log_stream.flush()

    target_path = target_dir / filename
    shutil.copy2(_log_file_path, target_path)

    try:
        from src.utils.data import make_writable

        make_writable(target_path)
    except ImportError:
        pass

    return target_path
