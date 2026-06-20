"""Benchmark-Modul: Registry und Factory für alle Benchmark-Wrapper.

Nutzung:
    from src.benchmarks import get_benchmark, list_benchmarks

    benchmark = get_benchmark("gpqa", num_samples=100)
    loader = DataLoader(benchmark, batch_size=4, collate_fn=benchmark.collate_fn)
"""

from .base import BenchmarkDataset
from .bigcodebench import WrapperBigCodeBench
from .gpqa import WrapperGPQA
from .livebench_da import WrapperLiveBenchDataAnalysis
from .mmlu_pro import WrapperMMLUPro
from .mixed import WrapperMixed

BENCHMARKS: dict[str, type[BenchmarkDataset]] = {
    "gpqa": WrapperGPQA,
    "livebench-da": WrapperLiveBenchDataAnalysis,
    "mmlu-pro": WrapperMMLUPro,
    "bigcodebench": WrapperBigCodeBench,
    "mixed": WrapperMixed,
}

# Empfohlene Sample-Anzahl pro Benchmark (None = volles Dataset)
RECOMMENDED_SAMPLES: dict[str, int | None] = {
    name: cls.recommended_num_samples for name, cls in BENCHMARKS.items()
}


def get_benchmark(
    name: str, num_samples: int | None = None, **kwargs
) -> BenchmarkDataset:
    """Factory: gibt den passenden Benchmark-Wrapper zurück.

    Args:
        name: Benchmark-Name (z.B. "gpqa", "livebench-da", "mmlu-pro", "bigcodebench").
        num_samples: Optionale Beschränkung der Sample-Anzahl.
        **kwargs: Weitere Parameter für den Wrapper.

    Returns:
        Initialisierter BenchmarkDataset-Wrapper.

    Raises:
        ValueError: Wenn der Benchmark-Name unbekannt ist.
    """
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unbekannter Benchmark: '{name}'. Verfügbar: {available}")

    return BENCHMARKS[name](num_samples=num_samples, **kwargs)


def list_benchmarks() -> list[str]:
    """Gibt alle verfügbaren Benchmark-Namen zurück."""
    return list(BENCHMARKS.keys())


__all__ = [
    "BenchmarkDataset",
    "WrapperGPQA",
    "WrapperLiveBenchDataAnalysis",
    "WrapperMMLUPro",
    "WrapperBigCodeBench",
    "WrapperMixed",
    "get_benchmark",
    "list_benchmarks",
    "BENCHMARKS",
    "RECOMMENDED_SAMPLES",
]
