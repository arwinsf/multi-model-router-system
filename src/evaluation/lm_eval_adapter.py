"""Replay-based lm-eval integration for post-hoc benchmark scoring."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import Any

SUPPORTED_LM_EVAL_BENCHMARKS = frozenset({"gpqa", "mmlu-pro"})
INVALID_EXTRACTION = "[INVALID]"
LM_EVAL_BACKEND = "lm-eval"
_TASK_PREFIX = "thesis_replay"

# Regex patterns aligned with the prompt templates and lm-eval conventions.
#
# GPQA prompt instructs: 'Your response should end with "The answer is (X)"'
#   → same pattern family but with (A-D)
_CHOICE_REGEX_PATTERNS = {
    "gpqa": r"(?i)answer\s+is\s*\(?([A-D])\)?",
    "mmlu-pro": r"(?i)answer\s+is\s*\(?([A-J])\)?",
}


class ReplayLM:
    """Minimal LM interface that replays stored generations by task/doc id."""

    def __init__(self, responses_by_task: dict[str, list[str]]) -> None:
        self._responses_by_task = responses_by_task
        self._rank = 0
        self._world_size = 1

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:  # pragma: no cover
        raise NotImplementedError("ReplayLM only supports generate_until requests.")

    def loglikelihood_rolling(self, requests) -> list[float]:  # pragma: no cover
        raise NotImplementedError("ReplayLM only supports generate_until requests.")

    def generate_until(self, requests) -> list[str]:
        responses: list[str] = []
        for request in requests:
            task_name = request.task_name
            doc_id = request.doc_id
            if task_name is None or doc_id is None:
                raise ValueError("lm-eval replay request without task_name/doc_id")

            try:
                responses.append(self._responses_by_task[task_name][doc_id])
            except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
                raise KeyError(
                    f"No replay response for task={task_name!r}, doc_id={doc_id!r}"
                ) from exc

        return responses


def get_lm_eval_version() -> str | None:
    try:
        return version("lm-eval")
    except PackageNotFoundError:
        return None


def benchmark_uses_lm_eval(benchmark_name: str | None) -> bool:
    return benchmark_name in SUPPORTED_LM_EVAL_BENCHMARKS or benchmark_name == "mixed"


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning prefix from model output."""
    marker = "</think>"
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx + len(marker) :].lstrip()
    return text


def replay_evaluate_samples(
    samples: list[dict[str, Any]],
    benchmark_name: str | None,
) -> dict[int, dict[str, Any]]:
    """Run lm-eval on stored generations and return per-sample results."""
    task_dict, responses_by_task = _build_replay_tasks(samples, benchmark_name)
    if not task_dict:
        return {}

    evaluate = _load_lm_eval_evaluate()
    result_bundle = evaluate(
        lm=ReplayLM(responses_by_task),
        task_dict=task_dict,
        bootstrap_iters=0,
        log_samples=True,
    )
    if result_bundle is None:  # pragma: no cover - world_size is always 1 here
        return {}

    backend_label = LM_EVAL_BACKEND
    backend_version = get_lm_eval_version()
    if backend_version:
        backend_label = f"{LM_EVAL_BACKEND}=={backend_version}"

    per_sample_results: dict[int, dict[str, Any]] = {}
    for task_samples in result_bundle.get("samples", {}).values():
        for logged_sample in task_samples:
            doc = logged_sample.get("doc", {})
            sample_index = doc.get("sample_index")
            sample_benchmark = doc.get("benchmark")
            if not isinstance(sample_index, int) or not isinstance(
                sample_benchmark, str
            ):
                continue

            extracted = _get_logged_filtered_response(logged_sample)
            if extracted in (None, ""):
                extracted = INVALID_EXTRACTION
            else:
                extracted = str(extracted).strip()
            score = float(logged_sample.get("exact_match", 0.0))
            per_sample_results[sample_index] = {
                "correct": score == 1.0,
                "score": score,
                "extracted_answer": extracted,
                "evaluation_method": _evaluation_method_name(sample_benchmark),
                "evaluation_backend": backend_label,
            }

    return per_sample_results


def _load_lm_eval_evaluate():
    try:
        from lm_eval.evaluator import evaluate
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "lm-eval ist fuer die Neu-Auswertung erforderlich. Installiere die Abhaengigkeiten aus requirements.txt."
        ) from exc

    return evaluate


def _load_configurable_task():
    try:
        from lm_eval.api.task import ConfigurableTask
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "lm-eval ist fuer die Neu-Auswertung erforderlich. Installiere die Abhaengigkeiten aus requirements.txt."
        ) from exc

    return ConfigurableTask


def _build_replay_tasks(
    samples: list[dict[str, Any]],
    benchmark_name: str | None,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for sample_index, sample in enumerate(samples):
        sample_benchmark = _resolve_sample_benchmark(benchmark_name, sample)
        if sample_benchmark in SUPPORTED_LM_EVAL_BENCHMARKS:
            grouped_indices[sample_benchmark].append(sample_index)

    if not grouped_indices:
        return {}, {}

    ConfigurableTask = _load_configurable_task()
    task_dict: dict[str, Any] = {}
    responses_by_task: dict[str, list[str]] = {}

    for sample_benchmark, indices in grouped_indices.items():
        task_name = _task_name(sample_benchmark)
        docs = [
            {
                "prompt": samples[index].get("prompt", ""),
                "target": samples[index].get("reference_answer"),
                "sample_index": index,
                "benchmark": sample_benchmark,
            }
            for index in indices
        ]
        task_dict[task_name] = ConfigurableTask(
            config=_task_config(task_name, sample_benchmark, docs)
        )
        responses_by_task[task_name] = [
            _strip_thinking(str(samples[index].get("output_text") or ""))
            for index in indices
        ]

    return task_dict, responses_by_task


def _task_config(
    task_name: str, benchmark_name: str, docs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "task": task_name,
        "custom_dataset": partial(_dataset_from_docs, docs),
        "test_split": "test",
        "doc_to_text": _doc_to_text,
        "doc_to_target": _doc_to_target,
        "output_type": "generate_until",
        "generation_kwargs": {"until": []},
        "filter_list": [
            {
                "name": "extract",
                "filter": [
                    {
                        "function": "regex",
                        "regex_pattern": _regex_pattern_for_benchmark(benchmark_name),
                        "fallback": INVALID_EXTRACTION,
                    },
                    {"function": "take_first"},
                ],
            }
        ],
        "metric_list": _metric_list_for_benchmark(benchmark_name),
        "num_fewshot": 0,
        "metadata": {
            "benchmark": benchmark_name,
            "mode": "replay",
        },
    }


def _dataset_from_docs(docs: list[dict[str, Any]], **_: Any):
    from datasets import Dataset, DatasetDict

    return DatasetDict({"test": Dataset.from_list(docs)})


def _doc_to_text(doc: dict[str, Any]) -> str:
    return str(doc.get("prompt", ""))


def _doc_to_target(doc: dict[str, Any]) -> str | None:
    target = doc.get("target")
    if target is None:
        return None
    return str(target)


def _resolve_sample_benchmark(
    benchmark_name: str | None,
    sample: dict[str, Any],
) -> str | None:
    if benchmark_name == "mixed":
        return sample.get("source_benchmark")
    return benchmark_name


def _task_name(benchmark_name: str) -> str:
    return f"{_TASK_PREFIX}__{benchmark_name.replace('-', '_')}"


def _regex_pattern_for_benchmark(benchmark_name: str) -> str:
    if benchmark_name in _CHOICE_REGEX_PATTERNS:
        return _CHOICE_REGEX_PATTERNS[benchmark_name]
    raise ValueError(f"Unsupported lm-eval replay benchmark: {benchmark_name}")


def _metric_list_for_benchmark(benchmark_name: str) -> list[dict[str, Any]]:
    """Return lm-eval metric_list config matching the official task definitions."""
    if benchmark_name == "gpqa":
        return [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_case": True,
                "ignore_punctuation": True,
            }
        ]
    if benchmark_name == "mmlu-pro":
        return [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "higher_is_better": True,
                "ignore_case": True,
                "ignore_punctuation": True,
            }
        ]
    raise ValueError(f"Unsupported lm-eval replay benchmark: {benchmark_name}")


def _evaluation_method_name(benchmark_name: str) -> str:
    return f"lm_eval_exact_match"


def _get_logged_filtered_response(logged_sample: dict[str, Any]) -> Any:
    filtered_resps = logged_sample.get("filtered_resps") or []
    if not filtered_resps:
        return INVALID_EXTRACTION

    first_response = filtered_resps[0]
    if isinstance(first_response, list):
        return first_response[0] if first_response else INVALID_EXTRACTION
    return first_response


__all__ = [
    "INVALID_EXTRACTION",
    "LM_EVAL_BACKEND",
    "SUPPORTED_LM_EVAL_BENCHMARKS",
    "benchmark_uses_lm_eval",
    "get_lm_eval_version",
    "replay_evaluate_samples",
]
