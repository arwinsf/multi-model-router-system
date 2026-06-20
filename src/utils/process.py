"""Prozess-Management: Zombie-Erkennung und Signal-Handler.

Stellt sicher, dass vLLM-Serverprozesse bei Programmabbruch (Ctrl+C,
SIGTERM, etc.) sauber beendet werden und keine Zombie-Instanzen auf
der GPU verbleiben.
"""

import atexit
import os
import signal
import subprocess
from time import sleep

from src.utils.logging import get_logger

logger = get_logger(__name__)


def kill_zombie_vllm_processes() -> int:
    """Findet und beendet verwaiste vLLM-Prozesse auf der GPU.

    Sucht nach Prozessen die 'vllm' im Commandline enthalten und
    vom aktuellen Benutzer gestartet wurden. Wird beim Start eines
    Experiments aufgerufen, um Reste früherer Abbrüche aufzuräumen.

    Returns:
        Anzahl der beendeten Prozesse.
    """
    killed = 0
    my_uid = os.getuid()
    my_pid = os.getpid()

    seen_pids: set[int] = set()

    def _try_kill_pid(pid: int, reason: str, cmdline_hint: str | None = None) -> None:
        nonlocal killed
        if pid in seen_pids:
            return
        seen_pids.add(pid)
        if pid == my_pid:
            return

        try:
            status_path = f"/proc/{pid}/status"
            if os.path.exists(status_path):
                with open(status_path, "r", encoding="utf-8") as handle:
                    status = handle.read()
                uid_line = next(
                    (line for line in status.splitlines() if line.startswith("Uid:")),
                    None,
                )
                if uid_line is not None:
                    real_uid = int(uid_line.split()[1])
                    if real_uid != my_uid:
                        return

            cmdline = cmdline_hint or "(unbekannt)"
            cmdline_path = f"/proc/{pid}/cmdline"
            if os.path.exists(cmdline_path):
                with open(cmdline_path, "r", encoding="utf-8") as handle:
                    raw_cmdline = handle.read().replace("\0", " ").strip()
                if raw_cmdline:
                    cmdline = raw_cmdline

            logger.warning(f"Beende Zombie-vLLM-Prozess: PID {pid} ({reason})")
            logger.warning(f"  Commandline: {cmdline[:200]}")
            os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warning(f"Keine Berechtigung zum Beenden von PID {pid}")

    try:
        # Alle Prozesse finden die vLLM im Commandline haben
        result = subprocess.run(
            ["pgrep", "-u", str(my_uid), "-f", "vllm"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return 0

        pids = [
            int(pid.strip()) for pid in result.stdout.strip().split("\n") if pid.strip()
        ]
        for pid in pids:
            _try_kill_pid(pid, reason="pgrep")

    except FileNotFoundError:
        # pgrep nicht verfügbar
        logger.debug("pgrep nicht verfügbar, überspringe Zombie-Erkennung")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",", maxsplit=1)]
                if len(parts) != 2:
                    continue
                pid_str, process_name = parts
                if "vllm" not in process_name.lower():
                    continue
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                _try_kill_pid(pid, reason=f"nvidia-smi:{process_name}")
    except Exception:
        logger.debug(
            "nvidia-smi nicht verfügbar oder GPU-Prozessabfrage fehlgeschlagen"
        )

    if killed > 0:
        logger.warning(
            f"{killed} Zombie-vLLM-Prozess(e) beendet. "
            f"GPU-VRAM sollte in wenigen Sekunden freigegeben werden."
        )
        sleep(2)  # Kurz warten bis VRAM freigegeben

    return killed


def register_cleanup(cleanup_fn) -> None:
    """Registriert eine Cleanup-Funktion für Signal-Handler und atexit.

    Bei SIGINT (Ctrl+C) und SIGTERM wird die Cleanup-Funktion aufgerufen,
    bevor der Prozess beendet wird. SIGHUP (SSH-Disconnect, Konsole
    geschlossen) wird bewusst ignoriert, damit laufende Experimente bei
    SSH-Verbindungsabbruch nicht abgebrochen werden.
    atexit fängt zusätzlich normales Programmende und Escape/Quit der
    TUI ab.

    Args:
        cleanup_fn: Callable ohne Argumente, das alle Ressourcen freigibt.
    """
    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    _cleanup_done = False

    # SIGHUP ignorieren: SSH-Disconnect soll laufende Experimente nicht beenden
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def _do_cleanup():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        logger.info("Cleanup: Beende alle Server-Prozesse...")
        try:
            cleanup_fn()
        except Exception as e:
            logger.error(f"Fehler beim Cleanup: {e}")

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"\n{sig_name} empfangen — räume auf...")
        _do_cleanup()
        # Original-Handler aufrufen oder beenden
        if signum == signal.SIGINT and callable(_original_sigint):
            _original_sigint(signum, frame)
        elif signum == signal.SIGTERM and callable(_original_sigterm):
            _original_sigterm(signum, frame)
        else:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_do_cleanup)
