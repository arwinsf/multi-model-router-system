"""Random-Routing-Experiment: Zufällige Modellzuweisung als Vergleichsbasis.

Misst den Energieverbrauch und die Qualität für zufällige Modell- und
Thinking-Zuweisung ohne Router-LLM. Dient als untere Vergleichsbasis
für das Router-Experiment (wie schlecht performt zufälliges Routing?).

Der Ablauf ist identisch zum Router-Experiment, aber:
- Kein Router-Modell wird geladen (0 VRAM für Router)
- Modell + Thinking werden per Zufall gewählt
- Scheduler arbeitet normal mit VRAM-Management
"""

import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.benchmarks import get_benchmark, list_benchmarks
from src.config import (
    get_config,
    get_loadable_models,
    get_server_config,
)
from src.energy import create_energy_monitor
from src.evaluation.postprocess import build_evaluation_config, reevaluate_result_bundle
from src.inference import VLLMServerManager
from src.routing import RandomRouter
from src.scheduler import ModelScheduler
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


def _build_parallel_waves(execution_plan: list) -> list[list]:
    """Gruppiert ExecutionGroups in parallele Wellen.

    Innerhalb einer Welle liegen alle Gruppen auf verschiedenen GPUs
    und können parallel ausgeführt werden. Zwischen Wellen wird
    sequentiell ausgeführt (für Modell-Swaps).
    """
    waves: list[list] = []
    for group in execution_plan:
        placed = False
        for wave in waves:
            wave_gpus: set[int] = set()
            for existing in wave:
                wave_gpus.update(existing.gpu_ids)
            if not wave_gpus.intersection(group.gpu_ids):
                wave.append(group)
                placed = True
                break
        if not placed:
            waves.append([group])
    return waves


def execute_random_routing_measurement(
    config: dict,
    energy_monitor,
    benchmark_dataset,
    batch_size: int = 32,
    seed: int | None = None,
    nvlink_available: bool = False,
) -> dict:
    """Führt das Random-Routing-Experiment durch.

    Args:
        config: Vollständige Konfiguration.
        energy_monitor: Initialisierter EnergyMonitor.
        benchmark_dataset: BenchmarkDataset-Instanz.
        batch_size: Anzahl Prompts pro Batch.
        seed: Optionaler Seed für Reproduzierbarkeit.

    Returns:
        Dictionary mit Messdaten und verschachtelten Samples.
    """
    logger = setup_logging()

    config.setdefault("experiment", {})["batch_size"] = batch_size

    benchmark_name = benchmark_dataset.name
    models_config = config["models"]
    inference_config = config.get("inference", {})

    loadable = get_loadable_models(config)
    logger.info(f"Ladbare Modelle: {[m.id for m in loadable]}")

    loadable_dicts = [m for m in models_config if m["id"] in {lm.id for lm in loadable}]
    random_router = RandomRouter(models=loadable_dicts, seed=seed)

    server_manager = VLLMServerManager(
        host=get_server_config(config).host,
    )
    scheduler = ModelScheduler(
        config,
        server_manager,
        router_vram_override=0.0,
        nvlink_available=nvlink_available,
    )

    def _cleanup():
        scheduler.shutdown()

    register_cleanup(_cleanup)

    loader = DataLoader(
        benchmark_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=benchmark_dataset.collate_fn,
        drop_last=False,
    )
    num_batches = len(loader)

    logger.info(
        f"Starte Messungen: {len(benchmark_dataset)} Anfragen "
        f"({num_batches} Batches, Batch-Size: {batch_size})"
    )

    # Für einen faireren Vergleich mit der Baseline beginnt die Messung vor
    # dem initialen Preload des Scheduler-Setups.
    logger.info(">>> Energiemessung START (vor initialem Preload)")
    energy_monitor.start_measurement()

    # Smart Preload
    preloaded = scheduler.preload()
    logger.info(f"Preloaded: {preloaded}")
    logger.info(f"VRAM verbleibend: {scheduler.available_vram():.1f}GB")

    # Scheduler-Events serialisieren (Preload-Events markieren)
    preload_event_count = len(scheduler.events)

    results = []
    routing_stats = defaultdict(int)
    thinking_stats = {"thinking": 0, "no_thinking": 0}

    sample_counter = 0
    total_input_tokens = 0
    total_output_tokens = 0
    latencies: list[float] = []

    for batch in tqdm(loader, desc="Random-Routing", total=num_batches):
        batch_prompts = batch["prompts"]
        batch_references = batch["references"]
        batch_metadata = batch["metadata"]

        # Random-Routing: Batch klassifizieren
        routing_results = random_router.route_batch(batch_prompts)

        for rr in routing_results:
            routing_stats[rr.assigned_model] += 1
            if rr.enable_thinking:
                thinking_stats["thinking"] += 1
            else:
                thinking_stats["no_thinking"] += 1

        # Execution Plan erstellen
        decisions = [(rr.assigned_model, rr.enable_thinking) for rr in routing_results]
        execution_plan = scheduler.get_execution_plan(decisions)

        # Parallele Inferenz in Wellen
        waves = _build_parallel_waves(execution_plan)
        batch_inference: dict[int, dict] = {}

        for wave in waves:
            for group in wave:
                scheduler.ensure_model_running(group.model_id)

            if len(wave) == 1:
                group = wave[0]
                client = scheduler.get_client(group.model_id)
                group_prompts = [batch_prompts[i] for i in group.indices]
                inference_results = client.generate_batch(
                    group_prompts, enable_thinking=group.enable_thinking
                )
                for local_idx, global_idx in enumerate(group.indices):
                    batch_inference[global_idx] = {
                        "inference_result": inference_results[local_idx],
                        "model_id": group.model_id,
                        "enable_thinking": group.enable_thinking,
                    }
            else:
                with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                    futures = {}
                    for group in wave:
                        client = scheduler.get_client(group.model_id)
                        group_prompts = [batch_prompts[i] for i in group.indices]
                        future = executor.submit(
                            client.generate_batch,
                            group_prompts,
                            enable_thinking=group.enable_thinking,
                        )
                        futures[future] = group

                    for future in as_completed(futures):
                        group = futures[future]
                        inference_results = future.result()
                        for local_idx, global_idx in enumerate(group.indices):
                            batch_inference[global_idx] = {
                                "inference_result": inference_results[local_idx],
                                "model_id": group.model_id,
                                "enable_thinking": group.enable_thinking,
                            }

        # Ergebnisse zusammenführen (Original-Reihenfolge)
        for idx in range(len(batch_prompts)):
            sample_id = sample_counter + idx
            prompt = batch_prompts[idx]
            reference = batch_references[idx]
            sample_metadata = batch_metadata[idx]
            rr = routing_results[idx]
            bi = batch_inference[idx]
            inf_result = bi["inference_result"]

            total_input_tokens += inf_result.input_tokens
            total_output_tokens += inf_result.output_tokens
            latencies.append(inf_result.latency_seconds)

            result_row = {
                "sample_id": sample_id,
                "prompt": prompt,
                "output_text": inf_result.output_text,
                "routing_decision": rr.assigned_model,
                "enable_thinking": bi["enable_thinking"],
                "router_output": rr.router_output,
                "routing_failed": False,
                "routing_error": None,
                "model_id": bi["model_id"],
                "reference_answer": reference,
                "is_correct": None,
                "eval_score": None,
                "extracted_answer": None,
                "input_tokens": inf_result.input_tokens,
                "output_tokens": inf_result.output_tokens,
                "total_tokens": inf_result.input_tokens + inf_result.output_tokens,
                "latency_seconds": inf_result.latency_seconds,
                "sample_tokens_per_second": inf_result.tokens_per_second,
            }
            for key, value in (sample_metadata or {}).items():
                if key not in result_row:
                    result_row[key] = value
            results.append(result_row)

        sample_counter += len(batch_prompts)

    # Energiemessung stoppen
    energy_measurement = energy_monitor.stop_measurement()

    # Statistiken loggen
    total = sum(routing_stats.values())
    logger.info("-" * 40)
    logger.info("Random-Routing-Statistik:")
    for model_id, count in sorted(routing_stats.items()):
        pct = 100 * count / total if total > 0 else 0
        logger.info(f"  {model_id}: {count}/{total} ({pct:.1f}%)")

    logger.info(f"Thinking: {thinking_stats}")

    logger.info(f"Scheduler-Events: {len(scheduler.events)}")
    for event in scheduler.events:
        logger.info(
            f"  [{event.timestamp.strftime('%H:%M:%S')}] "
            f"{event.action}: {event.model_id} {event.details}"
        )

    # Aufräumen
    scheduler.shutdown()

    # Messungs-Aggregate
    dyn_e = energy_measurement.dynamic_energy_joules
    dyn_p = energy_measurement.dynamic_power_watts
    dur = energy_measurement.duration_seconds
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
        "measurement_total_tokens": total_input_tokens + total_output_tokens,
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
        "measurement_thinking_share": (
            thinking_stats["thinking"] / num_results if num_results > 0 else 0
        ),
        "measurement_routing_failure_rate": 0.0,
        "measurement_scheduler_event_count": len(scheduler.events),
        "measurement_scheduler_startup_event_count": preload_event_count,
        "measurement_scheduler_runtime_event_count": max(
            len(scheduler.events) - preload_event_count, 0
        ),
    }

    # Scheduler-Events serialisieren: Startup-Preload und Laufzeit getrennt
    # markieren; beide Phasen liegen innerhalb der Messung.
    scheduler_events_log = [
        {
            "timestamp": e.timestamp.isoformat(),
            "action": e.action,
            "model_id": e.model_id,
            "details": e.details,
            "phase": "startup_preload" if i < preload_event_count else "runtime",
        }
        for i, e in enumerate(scheduler.events)
    ]

    return {
        "scenario": "random_routing",
        "benchmark": benchmark_name,
        "routing_stats": dict(routing_stats),
        "thinking_stats": thinking_stats,
        "routing_failures": 0,
        "scheduler_events": scheduler_events_log,
        **measurement_metrics,
        "samples": results,
        "power_samples": energy_measurement.power_samples,
    }


def _build_experiment_config(
    config: dict,
    batch_size: int,
    benchmark: str,
    num_samples: int,
    loadable_models: list,
    seed: int | None,
    nvlink_available: bool = False,
) -> dict:
    """Erstellt ein Konfigurationsabbild für die JSON-Ergebnisdatei."""
    hw = config.get("hardware", {})
    scheduler_cfg = config.get("scheduler", {})
    return {
        "profile": config.get("_profile", "unknown"),
        "experiment_type": "random_routing",
        "benchmark": benchmark,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "measurement_mode": "single",
        "seed": seed,
        "router": "random",
        "models": [
            {"id": m.id, "name": m.name, "vram_gb": m.vram_gb, "tier": m.tier}
            for m in loadable_models
        ],
        "hardware": {
            "per_gpu_vram_gb": hw.get("per_gpu_vram_gb"),
            "num_gpus": hw.get("num_gpus"),
            "ram_for_offloading_gb": hw.get("ram_for_offloading_gb"),
            "nvlink_available": nvlink_available,
        },
        "scheduler": scheduler_cfg,
        "inference": config.get("inference", {}),
    }


def main():
    """Hauptfunktion für Random-Routing-Experiment."""
    import argparse

    available_benchmarks = list_benchmarks()

    parser = argparse.ArgumentParser(
        description="Random-Routing-Experiment: Zufällige Modellzuweisung"
    )
    parser.add_argument(
        "--profile",
        "-p",
        choices=["local", "uni"],
        default=None,
        help="Config-Profil",
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
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random-Seed für Reproduzierbarkeit",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Temperatur fuer alle Zielmodelle (default: 0.0)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Optionaler Anzeigename fuer den Run (Plots/Listen)",
    )
    args = parser.parse_args()

    # SIGHUP ignorieren (SSH-Disconnect), SIGTERM → KeyboardInterrupt (Ctrl+Q via TUI)
    import signal

    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    logger = setup_logging()
    setup_file_logging()
    config = get_config(args.profile)

    if args.temperature is not None:
        for m in config.get("models", []):
            if "sampling" in m:
                m["sampling"]["temperature"] = args.temperature
            if "sampling_nothinking" in m:
                m["sampling_nothinking"]["temperature"] = args.temperature
        logger.info(f"Temperatur: {args.temperature}")

    run_alias = args.alias.strip() if args.alias else None
    if run_alias:
        logger.info(f"Run-Alias: {run_alias}")

    # Zombie-vLLM-Prozesse beenden
    kill_zombie_vllm_processes()

    logger.info("=" * 60)
    logger.info("LLM-Routing Energiemessung - Random-Routing (Vergleichsbasis)")
    logger.info(f"Profil: {config['_profile']}")
    models_info = get_loadable_models(config)
    logger.info(f"Ladbare Modelle: {[m.id for m in models_info]}")
    logger.info(f"Seed: {args.seed or 'keiner (zufällig)'}")
    logger.info("=" * 60)

    batch_size = args.batch_size or config.get("experiment", {}).get("batch_size", 32)
    gpu_config = config.get("gpu", {})
    output_dir = Path(config["experiment"]["output_dir"])
    measurement = None
    failed = False
    error_msg = None
    experiment_config = None

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

            experiment_config = _build_experiment_config(
                config,
                batch_size,
                args.benchmark,
                len(benchmark_dataset),
                models_info,
                args.seed,
                nvlink_available=nvlink_available,
            )
            experiment_config["evaluation"] = build_evaluation_config(
                config,
                args.benchmark,
            )
            if run_alias:
                experiment_config["experiment_alias"] = run_alias

            measurement = execute_random_routing_measurement(
                config=config,
                energy_monitor=energy_monitor,
                benchmark_dataset=benchmark_dataset,
                batch_size=batch_size,
                seed=args.seed,
                nvlink_available=nvlink_available,
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

    # Ergebnisse speichern (auch bei Fehler, wenn Daten vorhanden)
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
            f"random_{args.benchmark}",
            power_data=power_data,
            log_file=get_log_file_path(),
            failed=failed,
        )

        finalize_log(experiment_dir)

        if not failed:
            try:
                generate_experiment_plots(experiment_dir)
            except Exception as e:
                logger.warning(f"Plot-Generierung fehlgeschlagen: {e}")

            # Zusammenfassung
            logger.info("\n" + "=" * 60)
            logger.info("Zusammenfassung")
            logger.info("=" * 60)

            import pandas as pd

            df = pd.DataFrame(measurement["samples"])

            # Routing-Verteilung
            logger.info("\nRandom-Routing-Verteilung:")
            for model_id, count in df["routing_decision"].value_counts().items():
                pct = 100 * count / len(df)
                logger.info(f"  {model_id}: {count} ({pct:.1f}%)")

            # Thinking-Verteilung
            thinking_dist = df["enable_thinking"].value_counts()
            logger.info(f"\nThinking: {thinking_dist.to_dict()}")

            # Accuracy
            if df["is_correct"].notna().any():
                logger.info(f"\nGesamt-Accuracy: {df['is_correct'].mean():.1%}")
                for mid, acc in df.groupby("model_id")["is_correct"].mean().items():
                    logger.info(f"  {mid}: {acc:.1%}")

            logger.info("\nRandom-Routing-Experiment abgeschlossen!")
        else:
            logger.info(
                f"\nFehlgeschlagene Ergebnisse gespeichert in: {experiment_dir}"
            )


if __name__ == "__main__":
    main()
