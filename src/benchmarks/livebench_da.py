"""LiveBench Data Analysis Benchmark-Wrapper.

LiveBench Data Analysis (White et al., 2025, ICLR Spotlight) testet die Fähigkeit
von LLMs, strukturierte Daten (CSVs, JSON, Tabellen) zu manipulieren und zu
analysieren. Der Benchmark ist kontaminationsfrei durch regelmäßig aktualisierte
Aufgaben und nutzt objektive, automatisiert überprüfbare Ground-Truth-Antworten.

Tasks:
    - tablereformat: Konvertierung zwischen Tabellenformaten (CSV, JSON, HTML, Markdown, TSV)
    - cta: Column Type Annotation — Spaltentypklassifikation
    - tablejoin: Tabellenverknüpfung (Join-Operationen)
    - consecutive_events: Erkennung aufeinanderfolgender Ereignisse in Datenreihen

Dataset: livebench/data_analysis (HuggingFace, Test-Split: 150 Fragen)
Evaluation: Objektiver Ground-Truth-Vergleich (kein externes Bewertungsmodell nötig)

Referenz: https://arxiv.org/abs/2406.19314
"""

from .base import BenchmarkDataset


class WrapperLiveBenchDataAnalysis(BenchmarkDataset):
    """PyTorch Dataset-Wrapper für LiveBench Data Analysis.

    Lädt das livebench/data_analysis Dataset (Test-Split) und formatiert
    die Aufgaben als Prompts. Die Evaluation erfolgt durch objektiven
    Vergleich mit der Ground-Truth-Antwort.
    """

    name = "livebench-da"
    task_type = "data_analysis"
    evaluation_type = "ground_truth"
    recommended_num_samples: int | None = None  # Test-Split hat 150 Fragen

    def __init__(
        self,
        split: str = "test",
        num_samples: int | None = None,
    ):
        from datasets import load_dataset

        self.dataset = load_dataset("livebench/data_analysis", split=split)

        if num_samples is not None:
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # LiveBench speichert den Prompt im "turns"-Feld (Liste mit einem Element)
        turns = item["turns"]
        if isinstance(turns, list):
            prompt = turns[0]
        else:
            prompt = str(turns)

        ground_truth = item.get("ground_truth", "")
        task = item.get("task", "unknown")
        question_id = item.get("question_id", f"livebench_da_{idx}")

        return {
            "prompt": prompt,
            "reference": ground_truth,
            "metadata": {
                "task": task,
                "question_id": question_id,
                "category": "data_analysis",
            },
        }
