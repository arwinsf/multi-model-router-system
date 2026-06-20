"""Mixed-Benchmark: Gleichverteilte Stichprobe aus allen 4 Benchmark-Suites.

Zieht gleichverteilt Fragen aus GPQA, LiveBench Data Analysis, MMLU-Pro und BigCodeBench.
Jede Suite liefert num_samples/4 Fragen; die Auswertung delegiert an den
jeweiligen Source-Benchmark. Nach dem Ziehen werden alle Samples gemischt,
damit die Inferenz nicht benchmarkweise geblockt abläuft.

Beispiel:
    benchmark = WrapperMixed(num_samples=400, seed=42)
    # -> 100 GPQA + 100 LiveBench DA + 100 MMLU-Pro + 100 BigCodeBench
"""

import random

from .base import BenchmarkDataset


class WrapperMixed(BenchmarkDataset):
    """PyTorch Dataset-Wrapper für den gemischten Benchmark.

    Zieht gleichverteilt Samples aus allen 4 Benchmark-Suites und speichert
    den Source-Benchmark in den Metadaten, damit evaluate() korrekt delegieren
    kann. Die finale Sample-Liste wird deterministisch mit demselben Seed
    gemischt.
    """

    name = "mixed"
    task_type = "mixed"
    evaluation_type = "mixed"
    recommended_num_samples: int | None = 400

    _SUITE_NAMES = ["gpqa", "livebench-da", "mmlu-pro", "bigcodebench"]

    def __init__(self, num_samples: int | None = None, seed: int | None = None):
        from .bigcodebench import WrapperBigCodeBench
        from .gpqa import WrapperGPQA
        from .livebench_da import WrapperLiveBenchDataAnalysis
        from .mmlu_pro import WrapperMMLUPro

        total = num_samples or 400
        rng = random.Random(seed)

        # Gleichverteilung: jede Suite bekommt total//4, Rest geht an die letzten
        per_suite = total // 4
        counts = [per_suite] * 4
        remainder = total - sum(counts)
        for i in range(remainder):
            counts[-(i + 1)] += 1

        # Suites instanziieren (jeweils deren Benchmark-Default laden, dann samplen)
        suite_classes = [
            WrapperGPQA,
            WrapperLiveBenchDataAnalysis,
            WrapperMMLUPro,
            WrapperBigCodeBench,
        ]
        self._suites: dict[str, BenchmarkDataset] = {}
        self._samples: list[dict] = []

        for suite_name, suite_cls, count in zip(
            self._SUITE_NAMES, suite_classes, counts
        ):
            suite = suite_cls()
            self._suites[suite_name] = suite

            # Zufällige Indizes wählen
            available = list(range(len(suite)))
            selected = rng.sample(available, min(count, len(available)))

            for idx in selected:
                sample = suite[idx]
                # Source-Benchmark in Metadaten speichern für Delegation
                sample["metadata"]["source_benchmark"] = suite_name
                sample["metadata"]["source_index"] = idx
                self._samples.append(sample)

        rng.shuffle(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]
