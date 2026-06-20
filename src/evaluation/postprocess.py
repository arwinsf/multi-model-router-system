"""Post-Processing fuer Benchmark-Evaluation ausserhalb des Messfensters."""

from __future__ import annotations

from typing import Any

from src.evaluation.lm_eval_adapter import (
    LM_EVAL_BACKEND,
    benchmark_uses_lm_eval,
    get_lm_eval_version,
    replay_evaluate_samples,
)
from src.evaluation.bigcodebench_execution import (
    BigCodeBenchExecutionUnavailable,
    evaluate_bigcodebench_samples,
)
from src.evaluation.livebench_da_scoring import (
    evaluate_livebench_da_samples,
    score_livebench_da_sample,
)
from src.utils.logging import get_logger
from src.utils.metrics import EQ_SCORE_UNIT_KEY, compute_accuracy, compute_eq_score

logger = get_logger(__name__)


def benchmark_uses_lm_eval_backend(benchmark_name: str | None) -> bool:
    """Gibt zurueck, ob ein Benchmark ueber lm-eval neu ausgewertet wird."""
    return benchmark_uses_lm_eval(benchmark_name)


def build_evaluation_config(config: dict, benchmark_name: str | None) -> dict[str, Any]:
    """Erstellt ein Konfigurationsabbild fuer die Evaluationsstrategie."""
    version = get_lm_eval_version()
    if benchmark_name == "bigcodebench":
        return {
            "mode": "postprocess",
            "backend": "bigcodebench_execution",
        }

    if benchmark_name == "livebench-da":
        return {
            "mode": "postprocess",
            "backend": "livebench_data_analysis",
        }

    if benchmark_name == "mixed":
        evaluation_config: dict[str, Any] = {
            "mode": "postprocess",
            "backend": "hybrid",
            "subtask_backends": {
                "livebench-da": "livebench_data_analysis",
                "gpqa": LM_EVAL_BACKEND,
                "bigcodebench": "bigcodebench_execution",
                "mmlu-pro": LM_EVAL_BACKEND,
            },
            "replay_from_saved_generations": True,
        }
        if version:
            evaluation_config["backend_version"] = version
        return evaluation_config

    if benchmark_uses_lm_eval(benchmark_name):
        evaluation_config = {
            "mode": "postprocess",
            "backend": LM_EVAL_BACKEND,
            "replay_from_saved_generations": True,
        }
        if version:
            evaluation_config["backend_version"] = version
        return evaluation_config

    return {
        "mode": "postprocess",
        "backend": "string_match",
    }


def _resolve_sample_benchmark(run_benchmark: str | None, sample: dict) -> str | None:
    if run_benchmark == "mixed":
        return sample.get("source_benchmark")
    return run_benchmark


def _evaluate_without_lm_eval(sample: dict, benchmark_name: str | None) -> dict:
    prediction = sample.get("output_text", "")
    reference = sample.get("reference_answer")

    if benchmark_name == "bigcodebench":
        execution_passed = sample.get("bigcodebench_is_passed")
        if execution_passed is not None:
            return {
                "correct": bool(execution_passed),
                "score": 1.0 if execution_passed else 0.0,
                "extracted_answer": sample.get("bigcodebench_status"),
                "evaluation_method": "bigcodebench_execution",
                "evaluation_backend": "bigcodebench",
            }

        return {
            "correct": None,
            "score": None,
            "extracted_answer": None,
            "evaluation_method": "bigcodebench_execution",
            "evaluation_backend": "bigcodebench",
        }

    if benchmark_name == "livebench-da":
        return score_livebench_da_sample(sample)

    match = prediction.strip().lower() == (reference or "").strip().lower()
    return {
        "correct": match,
        "score": 1.0 if match else 0.0,
        "extracted_answer": prediction.strip()[:200],
        "evaluation_method": "string_match",
        "evaluation_backend": "builtin",
    }


def _refresh_measurement_metrics(measurement: dict) -> None:
    samples = measurement.get("samples", [])
    accuracy = compute_accuracy(samples)
    measurement["measurement_accuracy"] = accuracy
    measurement["measurement_eq_score"] = compute_eq_score(
        accuracy,
        measurement.get("measurement_dynamic_energy_joules"),
    )
    measurement["measurement_eq_score_unit"] = EQ_SCORE_UNIT_KEY


def reevaluate_measurement(
    measurement: dict,
    benchmark_name: str | None,
    log=None,
) -> dict[str, Any]:
    """Bewertet alle Samples einer Messung ausserhalb des Energie-Messfensters neu."""
    samples = measurement.get("samples", [])
    if not samples:
        _refresh_measurement_metrics(measurement)
        return {
            "total_samples": 0,
            "evaluated_samples": 0,
            "changed_extractions": 0,
            "changed_correctness": 0,
            "accuracy": measurement.get("measurement_accuracy"),
        }

    if log is None:
        log = logger

    changed_extractions = 0
    changed_correctness = 0
    evaluated_samples = 0
    results_by_index = replay_evaluate_samples(samples, benchmark_name)

    bigcodebench_indices = [
        index
        for index, sample in enumerate(samples)
        if _resolve_sample_benchmark(benchmark_name, sample) == "bigcodebench"
    ]
    if bigcodebench_indices:
        try:
            bcb_results = evaluate_bigcodebench_samples(
                [samples[index] for index in bigcodebench_indices],
                identifier=str(
                    measurement.get("measurement_id", "bigcodebench_reeval")
                ),
            )
        except BigCodeBenchExecutionUnavailable as exc:
            log.warning("BigCodeBench-Auswertung uebersprungen: %s", exc)
        else:
            for offset, sample_index in enumerate(bigcodebench_indices):
                results_by_index[sample_index] = bcb_results[offset]

    livebench_da_indices = [
        index
        for index, sample in enumerate(samples)
        if _resolve_sample_benchmark(benchmark_name, sample) == "livebench-da"
    ]
    if livebench_da_indices:
        lda_results = evaluate_livebench_da_samples(
            [samples[index] for index in livebench_da_indices]
        )
        for offset, sample_index in enumerate(livebench_da_indices):
            results_by_index[sample_index] = lda_results[offset]

    for index, sample in enumerate(samples):
        sample_benchmark = _resolve_sample_benchmark(benchmark_name, sample)
        if index not in results_by_index:
            results_by_index[index] = _evaluate_without_lm_eval(
                sample,
                sample_benchmark,
            )

    for index, sample in enumerate(samples):
        result = results_by_index[index]
        if sample.get("extracted_answer") != result["extracted_answer"]:
            changed_extractions += 1
        if sample.get("is_correct") != result["correct"]:
            changed_correctness += 1

        sample["is_correct"] = result["correct"]
        sample["eval_score"] = result["score"]
        sample["extracted_answer"] = result["extracted_answer"]
        sample["evaluation_method"] = result["evaluation_method"]

        evaluation_backend = result.get("evaluation_backend")
        if evaluation_backend:
            sample["evaluation_backend"] = evaluation_backend
        else:
            sample.pop("evaluation_backend", None)
        sample.pop("evaluation_model", None)

        sample_updates = result.get("sample_updates") or {}
        for key, value in sample_updates.items():
            sample[key] = value

        if result["correct"] is not None:
            evaluated_samples += 1

    _refresh_measurement_metrics(measurement)
    return {
        "total_samples": len(samples),
        "evaluated_samples": evaluated_samples,
        "changed_extractions": changed_extractions,
        "changed_correctness": changed_correctness,
        "accuracy": measurement.get("measurement_accuracy"),
    }


def reevaluate_result_bundle(
    data: dict,
    benchmark_name: str | None,
    log=None,
) -> dict[str, Any]:
    """Bewertet alle Messungen eines geladenen Result-Bundles neu."""
    if log is None:
        log = logger

    summary = {
        "measurements": 0,
        "total_samples": 0,
        "evaluated_samples": 0,
        "changed_extractions": 0,
        "changed_correctness": 0,
        "used_lm_eval": False,
        "backend": LM_EVAL_BACKEND if benchmark_uses_lm_eval(benchmark_name) else None,
        "backend_version": get_lm_eval_version(),
    }

    measurements = []
    if isinstance(data.get("measurement"), dict):
        measurements.append(data["measurement"])
    elif isinstance(data.get("measurements"), list):
        measurements.extend(
            measurement
            for measurement in data["measurements"]
            if isinstance(measurement, dict)
        )

    for measurement in measurements:
        measurement_benchmark = benchmark_name or measurement.get("benchmark")
        measurement_summary = reevaluate_measurement(
            measurement,
            benchmark_name=measurement_benchmark,
            log=log,
        )
        summary["measurements"] += 1
        summary["total_samples"] += measurement_summary["total_samples"]
        summary["evaluated_samples"] += measurement_summary["evaluated_samples"]
        summary["changed_extractions"] += measurement_summary["changed_extractions"]
        summary["changed_correctness"] += measurement_summary["changed_correctness"]

        if benchmark_uses_lm_eval(measurement_benchmark):
            summary["used_lm_eval"] = True

    evaluation_cfg = data.setdefault("config", {}).setdefault("evaluation", {})
    evaluation_cfg.pop("llm_extractor_enabled", None)
    evaluation_cfg.pop("answer_extractor_model", None)
    evaluation_cfg.pop("answer_extractor_thinking", None)
    evaluation_cfg.pop("llm_extractor_used", None)
    evaluation_cfg["last_evaluation_method"] = (
        "lm_eval_replay" if summary["used_lm_eval"] else "postprocess"
    )
    evaluation_cfg["last_evaluated_samples"] = summary["evaluated_samples"]
    evaluation_cfg["last_changed_extractions"] = summary["changed_extractions"]
    evaluation_cfg["last_changed_correctness"] = summary["changed_correctness"]
    if summary["used_lm_eval"]:
        evaluation_cfg["backend"] = LM_EVAL_BACKEND
        evaluation_cfg["replay_from_saved_generations"] = True
        if summary.get("backend_version"):
            evaluation_cfg["backend_version"] = summary["backend_version"]

    return summary
