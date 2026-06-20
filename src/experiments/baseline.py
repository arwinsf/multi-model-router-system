"""Baseline-Experiment: Always-Large mit Reasoning.

Misst den Energieverbrauch und die Qualität für ein einzelnes großes
Modell mit immer aktiviertem Thinking/Reasoning. Dient als Vergleichsbasis
für das Router-Experiment mit Multi-Modell-Scheduler.

- Lokal: Qwen3.5-9B (9.09GB)
- Uni: Qwen3.5-122B-A10B (~80GB, TP=2)
"""

import signal
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.benchmarks import get_benchmark, list_benchmarks
from src.config import (
    get_baseline,
    get_config,
    get_hardware,
    get_model_by_id,
    get_models,
    BaselineConfig,
)
from src.energy import create_energy_monitor
from src.evaluation.postprocess import build_evaluation_config, reevaluate_result_bundle
from src.inference import VLLMInference
from src.plotting import generate_experiment_plots
from src.utils.metrics import (
    compute_accuracy,
    compute_eq_score,
    compute_percentile,
    EQ_SCORE_UNIT_KEY,
)
from src.utils import (
    save_results,
    setup_logging,
    setup_file_logging,
    get_log_file_path,
    finalize_log,
)
from src.utils.process import kill_zombie_vllm_processes, register_cleanup


def execute_baseline_measurement(
    baseline_config,
    inference_config: dict,
    hw_config,
    energy_monitor,
    benchmark_dataset,
    batch_size: int = 32,
    nvlink_available: bool = False,
    _active_llm: list | None = None,
) -> dict:
    """Führt ein Always-Large-Baseline-Experiment mit Reasoning durch.

    Args:
        baseline_config: BaselineConfig mit Modell und Thinking-Flag.
        inference_config: Inferenz-Einstellungen aus der Config.
        hw_config: HardwareConfig für Auto-TP-Berechnung.
        energy_monitor: Initialisierter EnergyMonitor.
        benchmark_dataset: BenchmarkDataset-Instanz.
        batch_size: Anzahl Prompts pro Batch.

    Returns:
        Dictionary mit Messdaten und verschachtelten Samples.
    """
    logger = setup_logging()

    benchmark_name = benchmark_dataset.name
    vllm_config = inference_config.get("vllm", {})

    # Auto-Parallelisierung: min. TP für VRAM, restliche GPUs als DP
    auto_tp_threshold = 0.9
    tp = 1
    while (
        baseline_config.vram_gb / tp
    ) > hw_config.per_gpu_vram_gb * auto_tp_threshold:
        tp *= 2
        if tp > hw_config.num_gpus:
            tp = hw_config.num_gpus
            break
    dp = hw_config.num_gpus // tp

    gpu_mem_util = float(vllm_config.get("gpu_memory_utilization", 0.95))

    # Sampling-Parameter: Pro-Modell aus YAML, Fallback auf Defaults
    sampling = baseline_config.sampling
    temperature = sampling.get("temperature", 0.0)
    top_p = sampling.get("top_p", 0.9)
    top_k = sampling.get("top_k", -1)
    min_p = sampling.get("min_p", 0.0)
    presence_penalty = sampling.get("presence_penalty", 0.0)
    repetition_penalty = sampling.get("repetition_penalty", 1.0)
    max_model_len = vllm_config.get("max_model_len")
    max_new_tokens = int(max_model_len) if max_model_len else 32768

    llm = VLLMInference(
        model_name=baseline_config.model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        dtype=vllm_config.get("dtype", "auto"),
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        max_model_len=max_model_len,
        enforce_eager=vllm_config.get("enforce_eager", False),
        enable_thinking=baseline_config.enable_thinking,
        nvlink_available=nvlink_available,
    )
    if _active_llm is not None:
        _active_llm[0] = llm

    thinking_suffix = "reasoning" if baseline_config.enable_thinking else "non_thinking"
    scenario_name = f"{baseline_config.model_id}_{thinking_suffix}"
    logger.info(f"Starte {scenario_name} Experiment")
    logger.info(f"  Modell: {baseline_config.model_name}")
    logger.info(f"  Thinking: {baseline_config.enable_thinking}")
    logger.info(f"  TP: {tp}, DP: {dp} (Auto-Parallelisierung: min. TP + Rest als DP)")

    # Messfenster der Baseline: startet vor dem ersten Zielmodell-Load,
    # sodass dessen initiale GPU-Loading-Phase Teil der Messung ist.
    logger.info(">>> Energiemessung START (vor Modell-Laden)")
    energy_monitor.start_measurement()
    llm.load()

    loader = DataLoader(
        benchmark_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=benchmark_dataset.collate_fn,
        drop_last=False,
    )

    results = []
    num_batches = len(loader)
    logger.info(
        f"Starte Inferenz: {len(benchmark_dataset)} Anfragen "
        f"({num_batches} Batches, Batch-Size: {batch_size})"
    )

    sample_counter = 0
    total_input_tokens = 0
    total_output_tokens = 0
    latencies: list[float] = []

    for batch_index, batch in enumerate(
        tqdm(loader, desc=scenario_name, total=num_batches), start=1
    ):
        batch_prompts = batch["prompts"]
        batch_references = batch["references"]
        batch_metadata = batch["metadata"]

        logger.info(
            "Baseline Batch %d/%d (%d Prompts)",
            batch_index,
            num_batches,
            len(batch_prompts),
        )

        inference_results = llm.generate_batch(batch_prompts)

        for i, (prompt, inf_result, reference, sample_metadata) in enumerate(
            zip(batch_prompts, inference_results, batch_references, batch_metadata)
        ):
            sample_id = sample_counter + i
            total_input_tokens += inf_result.input_tokens
            total_output_tokens += inf_result.output_tokens
            latencies.append(inf_result.latency_seconds)

            result_row = {
                "sample_id": sample_id,
                "prompt": prompt,
                "output_text": inf_result.output_text,
                "reference_answer": reference,
                "is_correct": None,
                "eval_score": None,
                "extracted_answer": None,
                "input_tokens": inf_result.input_tokens,
                "output_tokens": inf_result.output_tokens,
                "total_tokens": inf_result.input_tokens + inf_result.output_tokens,
                "latency_seconds": inf_result.latency_seconds,
                "sample_tokens_per_second": inf_result.tokens_per_second,
                "enable_thinking": baseline_config.enable_thinking,
                "model_id": baseline_config.model_id,
                "model": baseline_config.model_name,
            }
            for key, value in (sample_metadata or {}).items():
                if key not in result_row:
                    result_row[key] = value
            results.append(result_row)

        sample_counter += len(batch_prompts)

    energy_measurement = energy_monitor.stop_measurement()
    logger.info(">>> Energiemessung STOP (nach Inferenz, vor Modell-Entladen)")
    llm.unload()
    if _active_llm is not None:
        _active_llm[0] = None

    # Messungs-Aggregate
    dyn_e = energy_measurement.dynamic_energy_joules
    dyn_p = energy_measurement.dynamic_power_watts
    dur = energy_measurement.duration_seconds
    measurement_total_tokens = total_input_tokens + total_output_tokens
    num_results = len(results)

    measurement_accuracy = compute_accuracy(results)

    measurement_metrics = {
        "measurement_energy_joules": energy_measurement.energy_joules,
        "measurement_dynamic_energy_joules": dyn_e,
        "measurement_duration_seconds": dur,
        "measurement_avg_power_watts": energy_measurement.avg_power_watts,
        "measurement_dynamic_power_watts": dyn_p,
        "measurement_min_power_watts": energy_measurement.min_power_watts,
        "measurement_max_power_watts": energy_measurement.max_power_watts,
        "measurement_idle_power_watts": energy_measurement.idle_power_watts,
        "measurement_num_power_samples": energy_measurement.num_samples,
        "measurement_total_input_tokens": total_input_tokens,
        "measurement_total_output_tokens": total_output_tokens,
        "measurement_total_tokens": measurement_total_tokens,
        "measurement_num_samples": num_results,
        "measurement_avg_input_tokens_per_sample": (
            total_input_tokens / num_results if num_results > 0 else 0
        ),
        "measurement_avg_output_tokens_per_sample": (
            total_output_tokens / num_results if num_results > 0 else 0
        ),
        "measurement_avg_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "measurement_p50_latency_seconds": compute_percentile(latencies, 50),
        "measurement_p95_latency_seconds": compute_percentile(latencies, 95),
        "measurement_accuracy": measurement_accuracy,
        "measurement_eq_score": compute_eq_score(measurement_accuracy, dyn_e),
        "measurement_eq_score_unit": EQ_SCORE_UNIT_KEY,
        "measurement_requests_per_second": (num_results / dur if dur > 0 else 0),
        "measurement_tokens_per_second": (total_output_tokens / dur if dur > 0 else 0),
        "measurement_joules_per_output_token": (
            dyn_e / total_output_tokens if total_output_tokens > 0 else 0
        ),
        "measurement_millijoules_per_output_token": (
            (dyn_e / total_output_tokens) * 1000 if total_output_tokens > 0 else 0
        ),
        "measurement_tokens_per_joule": (
            total_output_tokens / dyn_e if dyn_e > 0 else 0
        ),
        "measurement_tokens_per_watthour": (
            total_output_tokens / (dyn_e / 3600) if dyn_e > 0 else 0
        ),
        "measurement_energy_watthours": energy_measurement.energy_joules / 3600,
        "measurement_dynamic_energy_watthours": dyn_e / 3600,
        "measurement_thinking_share": 1.0 if baseline_config.enable_thinking else 0.0,
        "measurement_routing_failure_rate": 0.0,
        "measurement_scheduler_event_count": 0,
        "measurement_scheduler_startup_event_count": 0,
        "measurement_scheduler_runtime_event_count": 0,
    }

    logger.info(
        f"Messung abgeschlossen: {dyn_e:.2f} J dynamisch, "
        f"{dyn_p:.1f} W dynamisch, {dur:.1f} s, "
        f"{total_output_tokens} Output-Tokens"
    )

    return {
        "scenario": scenario_name,
        "benchmark": benchmark_name,
        "model": baseline_config.model_name,
        "model_id": baseline_config.model_id,
        "enable_thinking": baseline_config.enable_thinking,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        **measurement_metrics,
        "samples": results,
        "power_samples": energy_measurement.power_samples,
    }


def _build_experiment_config(
    config: dict,
    baseline_cfg,
    batch_size: int,
    benchmark: str,
    num_samples: int,
    nvlink_available: bool = False,
) -> dict:
    """Erstellt ein Konfigurationsabbild für die JSON-Ergebnisdatei."""
    hw = config.get("hardware", {})
    # Auto-Parallelisierung nachrechnen für JSON-Snapshot
    per_gpu = hw.get("per_gpu_vram_gb", 80.0)
    num_gpus = hw.get("num_gpus", 1)
    tp = 1
    while (baseline_cfg.vram_gb / tp) > per_gpu * 0.9:
        tp *= 2
        if tp > num_gpus:
            tp = num_gpus
            break
    dp = num_gpus // tp
    return {
        "profile": config.get("_profile", "unknown"),
        "experiment_type": "baseline",
        "benchmark": benchmark,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "measurement_mode": "single",
        "model": {
            "id": baseline_cfg.model_id,
            "name": baseline_cfg.model_name,
            "vram_gb": baseline_cfg.vram_gb,
            "enable_thinking": baseline_cfg.enable_thinking,
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
        },
        "hardware": {
            "per_gpu_vram_gb": hw.get("per_gpu_vram_gb"),
            "num_gpus": hw.get("num_gpus"),
            "ram_for_offloading_gb": hw.get("ram_for_offloading_gb"),
            "nvlink_available": nvlink_available,
        },
        "inference": config.get("inference", {}),
    }


def main():
    """Hauptfunktion für Baseline-Experiment."""
    import argparse

    available_benchmarks = list_benchmarks()

    parser = argparse.ArgumentParser(description="Baseline: Always-Large mit Reasoning")
    parser.add_argument(
        "--profile",
        "-p",
        choices=["local", "uni"],
        default=None,
        help="Config-Profil (default: LLM_ROUTING_PROFILE oder 'local')",
    )
    parser.add_argument(
        "--prompts",
        "-n",
        type=int,
        default=None,
        help="Anzahl der Prompts (default: Benchmark-Default)",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        choices=available_benchmarks,
        default="mmlu-pro",
        help="Benchmark (default: mmlu-pro)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch-Size (default: aus Config)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Einzelnes Modell aus dem Katalog testen (z.B. qwen3.5-9b). Thinking wird aktiviert.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Temperatur fuer die Messung (default: 0.0)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Optionaler Anzeigename fuer den Run (Plots/Listen)",
    )
    args = parser.parse_args()

    # SIGHUP ignorieren (SSH-Disconnect), SIGTERM → KeyboardInterrupt (Ctrl+Q via TUI)
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    logger = setup_logging()
    setup_file_logging()
    config = get_config(args.profile)

    # Modellauswahl: --model überschreibt die Baseline-Config
    if args.model:
        model_info = get_model_by_id(config, args.model)
        if model_info is None:
            available_ids = [m.id for m in get_models(config)]
            logger.error(
                f"Unbekanntes Modell: '{args.model}'. "
                f"Verfügbar: {', '.join(available_ids)}"
            )
            sys.exit(1)
        baseline_cfg = BaselineConfig(
            model_id=model_info.id,
            model_name=model_info.name,
            vram_gb=model_info.vram_gb,
            enable_thinking=True,
            sampling=model_info.sampling,
        )
        is_single_model = True
    else:
        baseline_cfg = get_baseline(config)
        is_single_model = False

    # Temperatur fuer Messungen standardmaessig auf Greedy-Decoding fixieren.
    if args.temperature is not None:
        baseline_cfg.sampling["temperature"] = args.temperature
        logger.info(f"Temperatur: {args.temperature}")

    run_alias = args.alias.strip() if args.alias else None
    if run_alias:
        logger.info(f"Run-Alias: {run_alias}")

    # Zombie-vLLM-Prozesse von früheren Abbrüchen beenden
    kill_zombie_vllm_processes()

    # Cleanup-Handler: VLLMInference bei Abbruch entladen
    _active_llm = [None]

    def _cleanup():
        if _active_llm[0] is not None:
            try:
                _active_llm[0].unload()
            except Exception:
                pass
            _active_llm[0] = None

    register_cleanup(_cleanup)

    logger.info("=" * 60)
    if is_single_model:
        logger.info("LLM-Routing Energiemessung - Einzelmodell-Test")
    else:
        logger.info("LLM-Routing Energiemessung - Baseline (Always-Large + Reasoning)")
    logger.info(f"Profil: {config['_profile']}")
    logger.info(f"Modell: {baseline_cfg.model_name}")
    logger.info(f"Thinking: {baseline_cfg.enable_thinking}")
    logger.info("=" * 60)

    batch_size = args.batch_size or config.get("experiment", {}).get("batch_size", 32)
    gpu_config = config.get("gpu", {})

    measurement = None
    failed = False
    error_msg = None
    experiment_config = None
    output_dir = Path(config["experiment"]["output_dir"])

    try:
        with create_energy_monitor(
            sampling_interval=gpu_config.get("sampling_interval", 0.1),
        ) as energy_monitor:

            gpu_info = energy_monitor.get_gpu_info()
            logger.info(f"GPU: {gpu_info['name']}")
            nvlink_available = gpu_info.get("nvlink_available", False)
            logger.info(f"NVLink: {'aktiv' if nvlink_available else 'nicht verfügbar'}")

            energy_monitor.measure_idle_power(duration=5.0)

            num_samples = args.prompts  # None = Benchmark-Default
            benchmark_dataset = get_benchmark(args.benchmark, num_samples=num_samples)
            logger.info(
                f"Benchmark: {args.benchmark}, {len(benchmark_dataset)} Samples"
            )

            inference_config = config.get("inference", {})
            num_samples = len(benchmark_dataset)

            experiment_config = _build_experiment_config(
                config,
                baseline_cfg,
                batch_size,
                args.benchmark,
                num_samples,
                nvlink_available=nvlink_available,
            )
            experiment_config["evaluation"] = build_evaluation_config(
                config,
                args.benchmark,
            )
            if run_alias:
                experiment_config["experiment_alias"] = run_alias

            measurement = execute_baseline_measurement(
                baseline_config=baseline_cfg,
                inference_config=inference_config,
                hw_config=get_hardware(config),
                energy_monitor=energy_monitor,
                benchmark_dataset=benchmark_dataset,
                batch_size=batch_size,
                nvlink_available=nvlink_available,
                _active_llm=_active_llm,
            )
            measurement["measurement_id"] = 1

            reevaluation_payload = {
                "config": experiment_config,
                "measurement": measurement,
            }
            reevaluation_summary = reevaluate_result_bundle(
                reevaluation_payload,
                benchmark_name=args.benchmark,
                log=logger,
            )
            measurement = reevaluation_payload["measurement"]
            eval_accuracy = measurement.get("measurement_accuracy")
            acc_str = f"{eval_accuracy:.1%}" if eval_accuracy is not None else "n/a"
            logger.info(
                "Evaluation abgeschlossen: %s Samples, Accuracy: %s | Extraktionen geaendert: %s, Correctness geaendert: %s",
                reevaluation_summary["evaluated_samples"],
                acc_str,
                reevaluation_summary["changed_extractions"],
                reevaluation_summary["changed_correctness"],
            )

    except KeyboardInterrupt:
        failed = True
        error_msg = "Experiment durch Benutzer abgebrochen (Ctrl+C)"
        logger.error(error_msg)
    except Exception as e:
        failed = True
        error_msg = f"Experiment fehlgeschlagen: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

    # Sicherstellen dass LLM entladen ist
    if _active_llm[0] is not None:
        try:
            _active_llm[0].unload()
            _active_llm[0] = None
        except Exception:
            pass

    # Ergebnisse speichern (auch bei Fehler, wenn Daten vorhanden)
    if is_single_model:
        prefix = f"single_{baseline_cfg.model_id}_{args.benchmark}"
    else:
        prefix = f"baseline_{args.benchmark}"

    if measurement is not None or failed:
        sampling_interval = gpu_config.get("sampling_interval", 0.1)
        power_data = None
        if measurement is not None:
            power_samples = measurement.pop("power_samples", [])
            power_data = (power_samples, sampling_interval)

        result_data = {"config": experiment_config}
        if measurement is not None:
            result_data["measurement"] = measurement
        if failed and error_msg:
            result_data["error"] = error_msg

        experiment_dir = save_results(
            result_data,
            output_dir,
            prefix,
            power_data=power_data,
            log_file=get_log_file_path(),
            failed=failed,
        )

        # Log nochmal kopieren (finale Version mit Save-Meldungen)
        finalize_log(experiment_dir)

        if not failed:
            try:
                generate_experiment_plots(experiment_dir)
            except Exception as e:
                logger.warning(f"Plot-Generierung fehlgeschlagen: {e}")

        if failed:
            logger.info(
                f"\nFehlgeschlagene Ergebnisse gespeichert in: {experiment_dir}"
            )
        elif is_single_model:
            logger.info("\nEinzelmodell-Experiment abgeschlossen!")
        else:
            logger.info("\nBaseline-Experiment abgeschlossen!")
    elif failed:
        result_data = {"config": experiment_config, "error": error_msg}
        experiment_dir = save_results(
            result_data,
            output_dir,
            prefix,
            log_file=get_log_file_path(),
            failed=True,
        )
        finalize_log(experiment_dir)
        logger.info(f"\nFehlgeschlagene Ergebnisse gespeichert in: {experiment_dir}")


if __name__ == "__main__":
    main()
