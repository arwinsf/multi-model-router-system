"""BigCodeBench Benchmark-Wrapper.

BigCodeBench (Zhuo et al., 2024, ICLR'25) ist ein praxisnaher Code-Generierungs-
Benchmark mit komplexen Instruktionen und vielfaeltigen Bibliotheksaufrufen.
Dieser Wrapper laedt die offiziellen Problemstellungen aus dem
``bigcode/bigcodebench`` HuggingFace-Dataset und liefert pro Sample einen
ausfuehrbaren Code-Prompt.

Splits:
    - ``complete``: Codecompletion auf Basis der Docstring (alle Modelle)
    - ``instruct``: Natural-Language-Instruktionen (Chat-/Instruct-Modelle)

Subsets:
    - ``full``: 1140 Aufgaben
    - ``hard`` : 148 besonders schwierige Aufgaben

Die Auswertung erfolgt nach der Inferenz ueber die offizielle
``bigcodebench``-Execution-Engine (siehe ``src.evaluation.bigcodebench_execution``).
Fehlt diese lokal, bleiben Samples unbewertet und koennen spaeter via
``python -m src.evaluation.reevaluate`` nachgezogen werden.

Referenz: https://arxiv.org/abs/2406.15877
"""

from .base import BenchmarkDataset


class WrapperBigCodeBench(BenchmarkDataset):
    """PyTorch Dataset-Wrapper fuer BigCodeBench.

    Standardmaessig ``instruct``/``hard`` (148 Aufgaben, Chat-Format).
    """

    name = "bigcodebench"
    task_type = "code_generation"
    evaluation_type = "execution"
    recommended_num_samples: int | None = None  # Volles hard-Set hat 148 Aufgaben

    def __init__(
        self,
        split: str = "instruct",
        subset: str = "hard",
        num_samples: int | None = None,
    ):
        if split not in {"instruct", "complete"}:
            raise ValueError(
                f"Unbekannter BigCodeBench-Split: {split!r} (erwartet: instruct, complete)"
            )
        if subset not in {"hard", "full"}:
            raise ValueError(
                f"Unbekanntes BigCodeBench-Subset: {subset!r} (erwartet: hard, full)"
            )

        self._split = split
        self._subset = subset

        problems = self._load_problems(subset)
        # Stabil nach task_id sortieren (BigCodeBench/0, BigCodeBench/1, ...)
        self._items = sorted(problems, key=lambda p: _task_index(p.get("task_id", "")))

        if num_samples is not None:
            self._items = self._items[: min(num_samples, len(self._items))]

    @staticmethod
    def _load_problems(subset: str) -> list[dict]:
        """Laedt die BigCodeBench-Aufgaben fuer das gewuenschte Subset."""
        # Bevorzugt offizielle Loader-Funktion, faellt sonst auf datasets zurueck.
        try:
            from bigcodebench.data import get_bigcodebench  # type: ignore

            mapping = get_bigcodebench(subset=subset)
            return list(mapping.values())
        except ImportError:
            pass

        from datasets import load_dataset

        config_name = "v0.1.4" if subset == "full" else "v0.1.4_hard"
        ds = load_dataset("bigcode/bigcodebench", config_name, split="train")
        return [dict(row) for row in ds]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        task_id = item.get("task_id", f"BigCodeBench/{idx}")
        if self._split == "instruct":
            prompt = item.get("instruct_prompt") or item.get("complete_prompt", "")
        else:
            prompt = item.get("complete_prompt") or item.get("instruct_prompt", "")

        return {
            "prompt": prompt,
            "reference": None,
            "metadata": {
                "benchmark": "bigcodebench",
                "index": idx,
                "task_id": task_id,
                "split": self._split,
                "subset": self._subset,
            },
        }


def _task_index(task_id: str) -> int:
    try:
        return int(task_id.rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        return 0
