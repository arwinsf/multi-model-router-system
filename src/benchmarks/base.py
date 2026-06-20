"""Abstrakte Basisklasse für alle Benchmark-Wrapper.

Jeder Benchmark implementiert das PyTorch Dataset-Interface.

Die Extraktion und Bewertung für GPQA erfolgt über lm-eval
(lm-evaluation-harness) im Post-Processing-Schritt
(src.evaluation.lm_eval_adapter). MMLU-Pro nutzt denselben Replay-Pfad mit
zehn Antwortoptionen. LiveBench Data Analysis nutzt einen eigenen objektiven
Ground-Truth-Vergleich; BigCodeBench wird über die offizielle Testcase-Ausführung bewertet
(src.evaluation.bigcodebench_execution).
"""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BenchmarkDataset(Dataset, ABC):
    """Abstrakte Basis für alle Benchmark-Wrapper.

    Jeder Wrapper lädt ein HuggingFace-Dataset und stellt Prompts,
    Referenzantworten und Metadaten bereit.

    Attribute:
        name: Benchmark-Name (z.B. "gpqa").
        task_type: z.B. "multiple_choice", "open_ended", "code_generation" oder "mixed".
        evaluation_type: z.B. "exact_match", "deterministic_short_answer" oder "mixed".
    """

    name: str
    task_type: str
    evaluation_type: str
    recommended_num_samples: int | None = None  # None = volles Dataset

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Gibt ein Sample zurück.

        Returns:
            Dict mit Keys: "prompt" (str), "reference" (str | None),
            "metadata" (dict).
        """
        ...

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Gruppiert eine Liste von Samples zu einem Batch-Dict.

        Args:
            batch: Liste von __getitem__-Ergebnissen.

        Returns:
            Dict mit Keys: "prompts" (list[str]),
            "references" (list[str | None]), "metadata" (list[dict]).
        """
        return {
            "prompts": [s["prompt"] for s in batch],
            "references": [s["reference"] for s in batch],
            "metadata": [s["metadata"] for s in batch],
        }
