"""vLLM Server-Management für Multi-Modell Scheduling.

Verwaltet vLLM-Serverprozesse mit Sleep/Wake-Support.
Jedes Zielmodell läuft als separater HTTP-Server (OpenAI-kompatible API).

Architektur:
    - Jeder Server ist ein Subprozess: vllm serve <model> --port <port>
    - Sleep Mode: Offloading von Weights in CPU-RAM (Level 1)
    - Wake: Reload von CPU-RAM zurück in VRAM
    - Erfordert VLLM_SERVER_DEV_MODE=1 und --enable-sleep-mode

Referenz: https://docs.vllm.ai/en/latest/features/sleep_mode.html
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Timeout-Werte
HEALTH_CHECK_TIMEOUT = 5.0  # Sekunden pro Health-Check-Request
HEALTH_CHECK_INTERVAL = 2.0  # Sekunden zwischen Checks
DEFAULT_STARTUP_TIMEOUT = 240.0  # Max. Wartezeit beim Serverstart
SLEEP_WAKE_TIMEOUT = 120.0  # Max. Wartezeit für Sleep/Wake

_VLLM_API_SERVER_WRAPPER = """
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    def _disable_request_instrumentation(self, app, *args, **kwargs):
        return self

    Instrumentator.instrument = _disable_request_instrumentation
except Exception:
    pass

import runpy

runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
"""


@dataclass
class ServerInfo:
    """Laufzeitinformationen eines vLLM-Servers."""

    model_id: str
    model_name: str
    port: int
    host: str
    process: subprocess.Popen | None = None
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int = 1
    gpu_ids: list[int] = field(default_factory=list)
    stderr_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class VLLMServerManager:
    """Verwaltet vLLM-Serverprozesse für Zielmodelle.

    Stellt Start/Stop/Sleep/Wake-Operationen bereit.
    Der Router selbst läuft NICHT als Server (bleibt offline vllm.LLM).
    """

    def __init__(self, host: str = "127.0.0.1"):
        self.host = host
        self._servers: dict[str, ServerInfo] = {}
        self._http_client = httpx.Client(timeout=30.0)

    def _format_startup_failure_log(self, stderr: str) -> str:
        """Kompakter, aber ursachenorientierter Auszug aus vLLM-Startup-Logs."""
        if not stderr:
            return ""

        lines = stderr.splitlines()
        needles = (
            "traceback",
            "error",
            "exception",
            "outofmemory",
            "out of memory",
            "free memory on device",
            "less than desired gpu memory utilization",
            "no available memory",
            "available kv cache memory",
            "failed to load model",
            "runtimeerror",
            "valueerror",
        )
        important_indices = [
            idx
            for idx, line in enumerate(lines)
            if any(needle in line.lower() for needle in needles)
        ]

        if not important_indices:
            return stderr[-8000:]

        selected: list[str] = []
        seen: set[int] = set()
        for idx in important_indices:
            start = max(0, idx - 2)
            end = min(len(lines), idx + 3)
            for line_no in range(start, end):
                if line_no in seen:
                    continue
                seen.add(line_no)
                selected.append(lines[line_no])

        excerpt = "\n".join(selected)
        tail = "\n".join(lines[-80:])
        if tail and tail not in excerpt:
            excerpt = f"{excerpt}\n\n--- Log-Ende ---\n{tail}"
        return excerpt[-12000:]

    def start_server(
        self,
        model_id: str,
        model_name: str,
        port: int,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        gpu_ids: list[int] | None = None,
        nvlink_available: bool = False,
        max_num_seqs: int | None = None,
        wait: bool = True,
    ) -> ServerInfo:
        """Startet einen neuen vLLM-Server für ein Modell.

        Args:
            model_id: Eindeutige ID des Modells.
            model_name: HuggingFace-Modellpfad.
            port: HTTP-Port für den Server.
            gpu_memory_utilization: Anteil des VRAM den vLLM nutzen darf.
            tensor_parallel_size: Anzahl GPUs für Tensor Parallelism.
            dtype: Datentyp (auto, bfloat16, float16).
            max_model_len: Maximale Sequenzlänge.
            enforce_eager: Wenn True, CUDA Graphs deaktivieren.
            startup_timeout: Max. Wartezeit bis Server bereit ist.
            gpu_ids: Liste der physischen GPU-IDs für CUDA_VISIBLE_DEVICES.
            wait: Wenn False, Prozess starten ohne auf Health-Check zu warten.

        Returns:
            ServerInfo mit Prozess-Handle und Verbindungsdaten.

        Raises:
            RuntimeError: Wenn Server nicht innerhalb des Timeouts startet (nur bei wait=True).
        """
        if model_id in self._servers:
            info = self._servers[model_id]
            if info.process is not None and info.process.poll() is None:
                logger.warning(
                    f"Server für {model_id} läuft bereits auf Port {info.port}"
                )
                return info

        cmd = [
            sys.executable,
            "-c",
            _VLLM_API_SERVER_WRAPPER,
            "--model",
            model_name,
            "--port",
            str(port),
            "--host",
            self.host,
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            dtype,
            "--trust-remote-code",
            "--enable-sleep-mode",
            "--limit-mm-per-prompt",
            '{"image": 0}',  # Text-only, kein Vision-Encoder (JSON-Format)
        ]

        if max_model_len is not None:
            cmd.extend(["--max-model-len", str(max_model_len)])

        if max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(max_num_seqs)])

        if enforce_eager:
            cmd.append("--enforce-eager")

        # Custom All-Reduce deaktivieren bei TP>1: schlägt auf A100 fehl
        # (cuda error custom_all_reduce.cuh:455). NCCL über NVLink ist ohnehin optimal.
        if tensor_parallel_size > 1:
            cmd.append("--disable-custom-all-reduce")

        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # GPU-Pinning: beschränke sichtbare GPUs für diesen Prozess
        if gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        # GPU-Speicher vor dem Start loggen (nvidia-smi)
        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if smi.returncode == 0:
                for line in smi.stdout.strip().splitlines():
                    logger.info(f"  nvidia-smi: GPU {line.strip()}")
        except Exception:
            pass

        logger.info(f"Starte vLLM-Server: {model_id} auf Port {port}")
        logger.info(f"  Modell: {model_name}")
        logger.info(f"  GPU Memory: {gpu_memory_utilization:.0%}")
        logger.info(f"  TP Size: {tensor_parallel_size}")
        logger.info(f"  GPU(s): {gpu_ids}")
        if max_num_seqs is not None:
            logger.info(f"  max_num_seqs: {max_num_seqs}")

        # stdout/stderr in temporäre Datei schreiben. Ein ungelesenes PIPE auf
        # stdout kann den vLLM-Server nach genug Logs blockieren.
        stderr_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"vllm_{model_id}_",
            suffix=".log",
            delete=False,
        )
        logger.info(f"  vLLM-Logdatei: {stderr_file.name}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stderr_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        stderr_file.close()

        info = ServerInfo(
            model_id=model_id,
            model_name=model_name,
            port=port,
            host=self.host,
            process=process,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            gpu_ids=gpu_ids or [],
            stderr_path=Path(stderr_file.name),
        )
        self._servers[model_id] = info

        if not wait:
            logger.info(f"Server gestartet (non-blocking): {model_id} auf Port {port}")
            return info

        if not self.wait_for_ready(model_id, timeout=startup_timeout):
            stderr = ""
            if info.stderr_path and info.stderr_path.exists():
                try:
                    stderr = info.stderr_path.read_text(errors="replace")
                except Exception:
                    pass
            stderr_excerpt = self._format_startup_failure_log(stderr)
            if stderr_excerpt:
                logger.error(
                    f"Server {model_id} nicht bereit nach {startup_timeout}s. "
                    f"vLLM-Startup-Log:\n{stderr_excerpt}"
                )
            self.stop_server(model_id)
            raise RuntimeError(
                f"vLLM-Server für {model_id} nicht innerhalb von "
                f"{startup_timeout}s gestartet."
            )

        logger.info(f"Server bereit: {model_id} auf {info.base_url}")
        return info

    def stop_server(self, model_id: str) -> None:
        """Beendet einen vLLM-Serverprozess."""
        info = self._servers.get(model_id)
        if info is None:
            return

        process_returncode = info.process.poll() if info.process is not None else None

        if info.process is not None and process_returncode is None:
            logger.info(f"Stoppe Server: {model_id} (PID {info.process.pid})")
            try:
                os.killpg(info.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                info.process.terminate()
            try:
                info.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning(f"Server {model_id} reagiert nicht, sende SIGKILL")
                try:
                    os.killpg(info.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    info.process.kill()
                info.process.wait(timeout=5)
        elif info.process is not None:
            stderr = ""
            if info.stderr_path and info.stderr_path.exists():
                try:
                    stderr = info.stderr_path.read_text(errors="replace")
                except Exception:
                    pass
            stderr_excerpt = self._format_startup_failure_log(stderr)
            if stderr_excerpt:
                logger.error(
                    f"Server {model_id} war bereits beendet (Exit {process_returncode}). "
                    f"vLLM-Logauszug:\n{stderr_excerpt}"
                )
            else:
                logger.error(
                    f"Server {model_id} war bereits beendet (Exit {process_returncode})."
                )

        info.process = None
        if info.stderr_path:
            logger.info(f"vLLM-Logdatei behalten: {info.stderr_path}")
        del self._servers[model_id]
        logger.info(f"Server gestoppt: {model_id}")

    def sleep_model(self, model_id: str, level: int = 1) -> None:
        """Versetzt ein Modell in den Sleep-Modus.

        Level 1: Offloading der Weights in CPU-RAM, KV-Cache verworfen.
        Level 2: Alles verworfen (Weights + KV-Cache).

        Wartet nach dem POST /sleep darauf, dass GET /is_sleeping
        tatsächlich True zurückgibt, um sicherzustellen, dass das
        Offloading abgeschlossen ist.

        Args:
            model_id: ID des Modells.
            level: Sleep-Level (1 oder 2).

        Raises:
            RuntimeError: Wenn der Server nicht antwortet oder Sleep
                nicht innerhalb des Timeouts abgeschlossen wird.
        """
        info = self._servers.get(model_id)
        if info is None:
            raise RuntimeError(f"Kein Server für {model_id} vorhanden")

        logger.info(f"Sleep Level {level}: {model_id}")
        url = f"{info.base_url}/sleep"
        response = self._http_client.post(
            url, params={"level": level}, timeout=SLEEP_WAKE_TIMEOUT
        )
        response.raise_for_status()

        # Warten bis Offloading abgeschlossen (is_sleeping == True)
        deadline = time.monotonic() + SLEEP_WAKE_TIMEOUT
        while time.monotonic() < deadline:
            if self.is_sleeping(model_id):
                logger.info(f"Sleep erfolgreich: {model_id}")
                return
            time.sleep(1.0)

        raise RuntimeError(
            f"Server {model_id} nicht innerhalb von {SLEEP_WAKE_TIMEOUT}s "
            f"in Sleep-Zustand gewechselt"
        )

    def wake_model(self, model_id: str) -> None:
        """Weckt ein schlafendes Modell auf (Reload von CPU-RAM in VRAM).

        Zweistufige Verifikation:
          1. Warten bis GET /is_sleeping False zurückgibt
             (Weights zurück in VRAM)
          2. Warten bis GET /health 200 zurückgibt
             (Server bereit für Inferenz)

        Args:
            model_id: ID des Modells.

        Raises:
            RuntimeError: Wenn der Server nicht innerhalb des
                Timeouts aufwacht oder nicht inferenzbereit wird.
        """
        info = self._servers.get(model_id)
        if info is None:
            raise RuntimeError(f"Kein Server für {model_id} vorhanden")

        logger.info(f"Wake: {model_id}")
        url = f"{info.base_url}/wake_up"
        response = self._http_client.post(url, timeout=SLEEP_WAKE_TIMEOUT)
        response.raise_for_status()

        # Phase 1: Warten bis Weights zurück in VRAM (is_sleeping == False)
        deadline = time.monotonic() + SLEEP_WAKE_TIMEOUT
        while time.monotonic() < deadline:
            if not self.is_sleeping(model_id):
                break
            time.sleep(1.0)
        else:
            raise RuntimeError(
                f"Server {model_id} nicht innerhalb von {SLEEP_WAKE_TIMEOUT}s "
                f"aufgewacht (is_sleeping bleibt True)"
            )

        # Phase 2: Warten bis Server inferenzbereit (/health == 200)
        while time.monotonic() < deadline:
            if self.health_check(model_id):
                logger.info(f"Wake erfolgreich: {model_id}")
                return
            time.sleep(1.0)

        raise RuntimeError(
            f"Server {model_id} aufgewacht, aber nicht inferenzbereit "
            f"innerhalb von {SLEEP_WAKE_TIMEOUT}s"
        )

    def is_sleeping(self, model_id: str) -> bool:
        """Prüft ob ein Modell im Sleep-Modus ist."""
        info = self._servers.get(model_id)
        if info is None:
            return False

        try:
            response = self._http_client.get(
                f"{info.base_url}/is_sleeping",
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            response.raise_for_status()
            return response.json().get("is_sleeping", False)
        except Exception:
            return False

    def health_check(self, model_id: str) -> bool:
        """Prüft ob ein Server bereit ist."""
        info = self._servers.get(model_id)
        if info is None:
            return False

        # Prüfen ob Prozess noch lebt
        if info.process is not None and info.process.poll() is not None:
            return False

        try:
            response = self._http_client.get(
                f"{info.base_url}/health",
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass

        try:
            response = self._http_client.get(
                f"{info.base_url}/v1/models",
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_ready(
        self,
        model_id: str,
        timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ) -> bool:
        """Wartet bis ein Server bereit ist.

        Args:
            model_id: ID des Modells.
            timeout: Maximale Wartezeit in Sekunden.

        Returns:
            True wenn Server bereit, False bei Timeout.
        """
        info = self._servers.get(model_id)
        if info is None:
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Prüfen ob Prozess abgestürzt ist
            if info.process is not None and info.process.poll() is not None:
                stderr = ""
                if info.stderr_path and info.stderr_path.exists():
                    try:
                        stderr = info.stderr_path.read_text(errors="replace")
                    except Exception:
                        pass
                stderr_excerpt = self._format_startup_failure_log(stderr)
                logger.error(
                    f"Server {model_id} abgestürzt (Exit {info.process.returncode}): "
                    f"{stderr_excerpt}"
                )
                return False

            if self.health_check(model_id):
                return True

            time.sleep(HEALTH_CHECK_INTERVAL)

        return False

    def get_server_info(self, model_id: str) -> ServerInfo | None:
        """Gibt ServerInfo für ein Modell zurück (oder None)."""
        return self._servers.get(model_id)

    def get_base_url(self, model_id: str) -> str | None:
        """Gibt die Base-URL eines Servers zurück."""
        info = self._servers.get(model_id)
        return info.base_url if info else None

    def shutdown_all(self) -> None:
        """Stoppt alle laufenden Server."""
        model_ids = list(self._servers.keys())
        for model_id in model_ids:
            try:
                self.stop_server(model_id)
            except Exception as e:
                logger.error(f"Fehler beim Stoppen von {model_id}: {e}")

        self._http_client.close()

    def __del__(self):
        """Aufräum-Sicherheit: alle Server stoppen."""
        try:
            self.shutdown_all()
        except Exception:
            pass
