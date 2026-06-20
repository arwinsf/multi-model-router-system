"""BigCodeBench-Auswertung via offizieller Execution-Engine.

Nutzt ``bigcodebench.evaluate.check_correctness`` fuer per-Sample Pass/Fail
ohne Umweg ueber CLI oder Sandbox-API. Ergebnisse werden in das
kanonische Sample-Schema (`is_correct`, `eval_score`, ...) eingetragen,
sodass die TUI BigCodeBench wie GPQA/LiveBench als Prozentwert anzeigt.

Pro Sample werden zusaetzlich folgende Felder gesetzt:
    - ``bigcodebench_status``: Status-String (``pass``/``fail``/``timeout``/...)
    - ``bigcodebench_is_passed``: ``True`` wenn der Status ``pass`` ist
    - ``bigcodebench_solution``: extrahierter Codeblock (Python)
"""

from __future__ import annotations

import re
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_AS_LIMIT_MB = 30 * 1024
DEFAULT_MAX_DATA_LIMIT_MB = 30 * 1024
DEFAULT_MAX_STACK_LIMIT_MB = 10
DEFAULT_MIN_TIME_LIMIT_S = 1.0
DEFAULT_GT_TIME_LIMIT_S = 20.0


class BigCodeBenchExecutionUnavailable(RuntimeError):
    """Signalisiert, dass die offizielle BigCodeBench-Auswertung lokal nicht bereit ist."""


def _import_bigcodebench_eval():
    try:
        from bigcodebench.eval import PASS  # type: ignore
        from bigcodebench.evaluate import check_correctness  # type: ignore
    except ImportError as exc:
        raise BigCodeBenchExecutionUnavailable(
            "BigCodeBench ist nicht installiert. Installiere es mit "
            "`pip install bigcodebench` und die Execution-Abhaengigkeiten via "
            "`pip install -r https://raw.githubusercontent.com/bigcode-project/"
            "bigcodebench/main/Requirements/requirements-eval.txt`."
        ) from exc

    return PASS, check_correctness


def _import_bigcodebench_data():
    try:
        from bigcodebench.data import get_bigcodebench  # type: ignore
    except ImportError as exc:
        raise BigCodeBenchExecutionUnavailable(
            "BigCodeBench-Daten konnten nicht geladen werden. "
            "Installiere `bigcodebench` (pip install bigcodebench)."
        ) from exc

    return get_bigcodebench


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Thinking-Tags (Qwen3.5 etc.) — alles vor dem letzten </think> ist Denkprozess.
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Entfernt den Thinking-Abschnitt aus der Modellantwort.

    Bei aktiviertem Reasoning enthaelt die Antwort ``<think>...</think>``
    gefolgt von der eigentlichen Antwort. Nur der Teil nach dem letzten
    ``</think>``-Tag wird fuer die Code-Extraktion verwendet.
    """
    match = None
    for match in _THINK_CLOSE_RE.finditer(text):
        pass
    if match:
        return text[match.end() :]
    return text


def _get_sample_value(
    sample: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Liest Benchmark-Metadaten aus verschachtelten oder flachen Samples.

    Roh-Benchmark-Samples tragen die Felder unter ``metadata``. Persistierte
    Experiment-Resultate flatten diese Keys jedoch auf Top-Level. Fuer die
    Neu-Auswertung akzeptieren wir beide Formen.
    """
    meta = sample.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    for key in keys:
        value = meta.get(key)
        if value not in (None, ""):
            return value

        value = sample.get(key)
        if value not in (None, ""):
            return value

    return default


def extract_code(output_text: str) -> str:
    """Extrahiert Python-Code aus einer Modellantwort.

    Bei aktiviertem Reasoning (Thinking-Tags vorhanden) wird zuerst der
    Thinking-Abschnitt entfernt, damit nur die finale Antwort durchsucht
    wird. Bevorzugt den laengsten ```python ...```-Block, faellt sonst auf
    den laengsten Fenceblock zurueck und nutzt im Notfall den Rohtext.
    """
    if not output_text:
        return ""

    # Thinking-Abschnitt entfernen — Draft-Code in <think> ignorieren
    final_text = _strip_thinking(output_text)

    matches = _CODE_FENCE_RE.findall(final_text)
    if matches:
        # Laengsten Block waehlen (haeufig die finale Loesung)
        return max(matches, key=len).strip()

    # Fallback: gesamten Output durchsuchen (kein Thinking vorhanden oder
    # finale Antwort enthaelt keinen Fenceblock)
    matches = _CODE_FENCE_RE.findall(output_text)
    if matches:
        return max(matches, key=len).strip()

    return output_text.strip()


def _resolve_problems(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, dict], str, str]:
    """Bestimmt Subset/Split aus den Sample-Metadaten und laedt die Probleme."""
    subset = "hard"
    split = "instruct"
    for sample in samples:
        subset = _get_sample_value(
            sample, "subset", "bigcodebench_subset", default=subset
        )
        split = _get_sample_value(sample, "split", "bigcodebench_split", default=split)
        if subset != "hard" or split != "instruct":
            break

    get_bigcodebench = _import_bigcodebench_data()
    problems = get_bigcodebench(subset=subset)
    return problems, subset, split


_DEFAULT_MAX_WORKERS = 16


def _should_use_subprocess() -> bool:
    """Prueft ob die Evaluation in einem Subprozess laufen soll.

    Nach Router-/vLLM-Nutzung ist der CUDA-Kontext initialisiert und der
    Prozess traegt vererbte Thread-/Speicher-Ressourcen, die
    check_correctness zum Scheitern bringen (EAGAIN bei Thread-Erzeugung,
    RLIMIT-Verletzungen durch vererbte CUDA-Mappings).  Ein frischer
    Subprozess umgeht diese Ressourcenkonflikte.
    """
    try:
        import torch

        return torch.cuda.is_initialized()
    except ImportError:
        return False


def _evaluate_via_subprocess(
    samples: list[dict[str, Any]],
    *,
    identifier: str | None = None,
    max_as_limit_mb: int = DEFAULT_MAX_AS_LIMIT_MB,
    max_data_limit_mb: int = DEFAULT_MAX_DATA_LIMIT_MB,
    max_stack_limit_mb: int = DEFAULT_MAX_STACK_LIMIT_MB,
    min_time_limit_s: float = DEFAULT_MIN_TIME_LIMIT_S,
    gt_time_limit_s: float = DEFAULT_GT_TIME_LIMIT_S,
) -> list[dict[str, Any]]:
    """Fuehrt die BigCodeBench-Evaluation in einem sauberen Subprozess aus.

    Vermeidet Ressourcenkonflikte (Thread-Limits, CUDA-Mappings) die nach
    vLLM/Router-Nutzung im selben Prozess auftreten koennen.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parent.parent.parent)

    payload = {
        "samples": samples,
        "identifier": identifier,
        "max_as_limit_mb": max_as_limit_mb,
        "max_data_limit_mb": max_data_limit_mb,
        "max_stack_limit_mb": max_stack_limit_mb,
        "min_time_limit_s": min_time_limit_s,
        "gt_time_limit_s": gt_time_limit_s,
    }

    fd_in, input_path = tempfile.mkstemp(suffix=".json", prefix="bcb_eval_")
    output_path = input_path + ".out"

    try:
        with os.fdopen(fd_in, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        logger.info(
            "Starte BigCodeBench-Evaluation in Subprozess (%d Samples)",
            len(samples),
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluation.bigcodebench_execution",
                input_path,
                output_path,
            ],
            cwd=project_root,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"BigCodeBench-Subprozess fehlgeschlagen "
                f"(Exit-Code {result.returncode})"
            )

        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        logger.info("BigCodeBench-Subprozess abgeschlossen")
        return results

    finally:
        for p in (input_path, output_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def evaluate_bigcodebench_samples(
    samples: list[dict[str, Any]],
    *,
    identifier: str | None = None,
    max_as_limit_mb: int = DEFAULT_MAX_AS_LIMIT_MB,
    max_data_limit_mb: int = DEFAULT_MAX_DATA_LIMIT_MB,
    max_stack_limit_mb: int = DEFAULT_MAX_STACK_LIMIT_MB,
    min_time_limit_s: float = DEFAULT_MIN_TIME_LIMIT_S,
    gt_time_limit_s: float = DEFAULT_GT_TIME_LIMIT_S,
) -> list[dict[str, Any]]:
    """Wertet BigCodeBench-Samples gegen die offiziellen Testfaelle aus.

    Fuer jedes Sample wird per-Aufgabe ``check_correctness`` aufgerufen.
    Die Ausfuehrung erfolgt parallel (ThreadPool), da jeder
    ``check_correctness``-Aufruf einen eigenen Subprozess spawnt und das
    20-Sekunden-Timeout pro Sample bei sequenzieller Ausfuehrung zu
    O(timeout * failing_samples) Gesamtlaufzeit fuehrt.

    Rueckgabe ist eine Liste in derselben Reihenfolge wie ``samples`` mit
    dem kanonischen ``postprocess.py``-Schema.
    """
    if not samples:
        return []

    # Nach vLLM/CUDA-Nutzung: Subprozess fuer saubere Evaluation starten,
    # um Ressourcenkonflikte (Thread-Limits, CUDA-Mappings) zu vermeiden.
    if _should_use_subprocess():
        return _evaluate_via_subprocess(
            samples,
            identifier=identifier,
            max_as_limit_mb=max_as_limit_mb,
            max_data_limit_mb=max_data_limit_mb,
            max_stack_limit_mb=max_stack_limit_mb,
            min_time_limit_s=min_time_limit_s,
            gt_time_limit_s=gt_time_limit_s,
        )

    import gc
    import multiprocessing
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    PASS, check_correctness = _import_bigcodebench_eval()
    problems, subset, split = _resolve_problems(samples)

    # PIDs der bereits existierenden Kindprozesse merken (z.B. vLLM TP-Worker).
    # Nur NEUE Kinder (Manager-Server von check_correctness) werden am Ende
    # bereinigt, damit die stdout-Pipe sauber geschlossen wird und die CLI
    # den Prozess-Exit erkennt.
    _children_before = {p.pid for p in multiprocessing.active_children()}

    run_id = identifier or "bigcodebench"
    results: list[dict[str, Any] | None] = [None] * len(samples)

    def _make_skip_result(status: str) -> dict[str, Any]:
        return {
            "correct": None,
            "score": None,
            "extracted_answer": None,
            "evaluation_method": "bigcodebench_execution",
            "evaluation_backend": "bigcodebench",
            "sample_updates": {
                "bigcodebench_status": status,
                "bigcodebench_is_passed": False,
            },
        }

    def _eval_one(offset: int, sample: dict[str, Any]) -> dict[str, Any]:
        task_id = _get_sample_value(sample, "task_id")
        if not task_id:
            return _make_skip_result("missing_task_id")

        problem = problems.get(task_id)
        if problem is None:
            return _make_skip_result("unknown_task_id")

        solution_code = extract_code(sample.get("output_text", ""))

        try:
            result = check_correctness(
                offset,
                problem,
                solution_code,
                max_as_limit_mb,
                max_data_limit_mb,
                max_stack_limit_mb,
                f"{run_id}:{task_id}",
                min_time_limit_s,
                gt_time_limit_s,
            )
        except Exception as exc:
            return {
                "correct": False,
                "score": 0.0,
                "extracted_answer": solution_code[:200],
                "evaluation_method": "bigcodebench_execution",
                "evaluation_backend": "bigcodebench",
                "sample_updates": {
                    "bigcodebench_status": f"error:{exc.__class__.__name__}",
                    "bigcodebench_is_passed": False,
                    "bigcodebench_solution": solution_code,
                },
            }

        status, _details = result.get("base", (None, None))
        is_passed = status == PASS
        return {
            "correct": bool(is_passed),
            "score": 1.0 if is_passed else 0.0,
            "extracted_answer": (status or "")[:200],
            "evaluation_method": "bigcodebench_execution",
            "evaluation_backend": "bigcodebench",
            "sample_updates": {
                "bigcodebench_status": status,
                "bigcodebench_is_passed": bool(is_passed),
                "bigcodebench_solution": solution_code,
                "bigcodebench_subset": subset,
                "bigcodebench_split": split,
            },
        }

    max_workers = min(os.cpu_count() or 4, _DEFAULT_MAX_WORKERS, len(samples))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_eval_one, offset, sample): offset
            for offset, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    # check_correctness erstellt pro Aufruf einen multiprocessing.Manager(),
    # dessen Server-Kindprozess die stdout-Pipe erbt.  Ohne expliziten
    # Cleanup blockiert der uebergeordnete CLI-Subprozess beim Exit, weil
    # die Pipe nie geschlossen wird und ProcessScreen.EOF nie empfaengt.
    gc.collect()
    for child in multiprocessing.active_children():
        if child.pid not in _children_before:
            child.terminate()
    for child in multiprocessing.active_children():
        if child.pid not in _children_before:
            child.join(timeout=2)

    return results  # type: ignore[return-value]


if __name__ == "__main__":
    import json
    import sys

    _input_path = sys.argv[1]
    _output_path = sys.argv[2]

    with open(_input_path, "r", encoding="utf-8") as _f:
        _payload = json.load(_f)

    _results = evaluate_bigcodebench_samples(
        _payload["samples"],
        identifier=_payload.get("identifier"),
        max_as_limit_mb=_payload.get("max_as_limit_mb", DEFAULT_MAX_AS_LIMIT_MB),
        max_data_limit_mb=_payload.get("max_data_limit_mb", DEFAULT_MAX_DATA_LIMIT_MB),
        max_stack_limit_mb=_payload.get(
            "max_stack_limit_mb", DEFAULT_MAX_STACK_LIMIT_MB
        ),
        min_time_limit_s=_payload.get("min_time_limit_s", DEFAULT_MIN_TIME_LIMIT_S),
        gt_time_limit_s=_payload.get("gt_time_limit_s", DEFAULT_GT_TIME_LIMIT_S),
    )

    with open(_output_path, "w", encoding="utf-8") as _f:
        json.dump(_results, _f)
