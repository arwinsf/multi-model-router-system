"""GPQA Diamond Benchmark-Wrapper: Graduate-Level Google-Proof Q&A.

Der Wrapper folgt dem offiziellen GPQA-Task aus lm-eval-harness:
- Dataset: ``Idavidrein/gpqa`` mit Konfiguration ``gpqa_diamond``
- Preprocessing: identisch zur offiziellen ``utils.py`` (Whitespace + [title])
- Antwortoptionen: 3 falsche + 1 korrekte, zufällig gemischt

Für dieses Framework wird das offizielle Multiple-Choice-Template um einen
expliziten final-answer cue ergänzt, damit die nachgelagerte Replay-Auswertung
generierte Antworten robust extrahieren kann.
"""

import random

from .base import BenchmarkDataset


def _preprocess(text: str | None) -> str:
    """Bereinigt Antworttext (analog zu lm-eval-harness)."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


class WrapperGPQA(BenchmarkDataset):
    """PyTorch Dataset-Wrapper für GPQA Diamond.

    Lädt das Idavidrein/gpqa Dataset (gpqa_diamond Konfiguration)
    und formatiert Fragen als Multiple-Choice Prompts mit 4 Optionen (A-D).
    Die Antwortoptionen werden zufällig gemischt (wie in lm-eval-harness).
    """

    name = "gpqa"
    task_type = "multiple_choice"
    evaluation_type = "exact_match"
    recommended_num_samples: int | None = None  # 198 Fragen = volles Diamond-Set

    def __init__(
        self,
        split: str = "train",
        num_samples: int | None = None,
        seed: int = 42,
    ):
        from datasets import load_dataset

        self.dataset = load_dataset(
            "Idavidrein/gpqa",
            "gpqa_diamond",
            split=split,
            token=True,
        )
        self._seed = seed

        if num_samples is not None:
            self.dataset = self.dataset.select(
                range(min(num_samples, len(self.dataset)))
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        question = item["Question"]

        # Antworten sammeln und mischen (analog zu lm-eval-harness utils.py)
        choices = [
            _preprocess(item["Incorrect Answer 1"]),
            _preprocess(item["Incorrect Answer 2"]),
            _preprocess(item["Incorrect Answer 3"]),
            _preprocess(item["Correct Answer"]),
        ]

        # Deterministisch mischen basierend auf Seed + Index
        rng = random.Random(self._seed + idx)
        rng.shuffle(choices)

        correct_answer = _preprocess(item["Correct Answer"])
        correct_index = choices.index(correct_answer)
        answer_letter = chr(65 + correct_index)
        answer_target = f"({answer_letter})"

        # Offizielles GPQA-Template aus lm-eval-harness, ergänzt um das
        # finale Antwortformat fuer generative Replay-Auswertung.
        choice_str = "\n".join(
            f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices)
        )
        prompt = (
            f"What is the correct answer to this question:{question.strip()}\n"
            f"Choices:\n{choice_str}\n"
            'Answer: Think through the options and finish with "The answer is (X)" '
            "where X is one of A, B, C, or D."
        )

        return {
            "prompt": prompt,
            "reference": answer_target,
            "metadata": {
                "benchmark": "gpqa",
                "index": idx,
                "correct_answer": answer_target,
                "correct_answer_letter": answer_letter,
                "domain": item.get("High-level domain", "unknown"),
            },
        }
