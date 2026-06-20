"""MMLU-Pro Benchmark-Wrapper.

MMLU-Pro (Wang et al., 2024) erweitert MMLU auf bis zu zehn Antwortoptionen
und erschwert dadurch Raten und Pattern-Matching. Fuer diese Arbeit wird eine
alltagsnahe Teilmenge genutzt: Business, Health, Law und Psychology mit jeweils
den ersten 250 Fragen aus dem Test-Split.
"""

from __future__ import annotations

from .base import BenchmarkDataset

MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_CATEGORIES = ("business", "health", "law", "psychology")
MMLU_PRO_SAMPLES_PER_CATEGORY = 250
_LETTERS = "ABCDEFGHIJ"


class WrapperMMLUPro(BenchmarkDataset):
    """PyTorch Dataset-Wrapper fuer die MMLU-Pro-Alltagswissen-Teilstichprobe."""

    name = "mmlu-pro"
    task_type = "multiple_choice"
    evaluation_type = "exact_match"
    recommended_num_samples: int | None = (
        len(MMLU_PRO_CATEGORIES) * MMLU_PRO_SAMPLES_PER_CATEGORY
    )

    def __init__(
        self,
        split: str = "test",
        num_samples: int | None = None,
        categories: tuple[str, ...] = MMLU_PRO_CATEGORIES,
        samples_per_category: int = MMLU_PRO_SAMPLES_PER_CATEGORY,
    ):
        from datasets import load_dataset

        dataset = load_dataset(MMLU_PRO_DATASET, split=split)
        selected: list[dict] = []

        for category in categories:
            category_rows = [
                dict(row)
                for row in dataset
                if _normalize_category(row.get("category")) == category
            ]
            selected.extend(category_rows[:samples_per_category])

        self._items = selected
        if num_samples is not None:
            self._items = self._items[: min(num_samples, len(self._items))]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        item = self._items[idx]
        question = str(item.get("question", "")).strip()
        options = [str(option).strip() for option in item.get("options", [])]
        answer_index = _resolve_answer_index(item, options)
        answer_letter = _LETTERS[answer_index]
        category = _normalize_category(item.get("category"))

        choice_str = "\n".join(
            f"{letter}. {option}"
            for letter, option in zip(_LETTERS, options, strict=False)
        )
        prompt = (
            "The following are multiple choice questions (with answers) about "
            f"{_category_label(category)}. Think step by step and then finish your "
            'answer with "the answer is (X)" where X is the correct letter choice.\n\n'
            f"Question:\n{question}\n"
            f"Options:\n{choice_str}\n"
            "Answer: Let's think step by step."
        )

        return {
            "prompt": prompt,
            "reference": answer_letter,
            "metadata": {
                "benchmark": "mmlu-pro",
                "index": idx,
                "question_id": item.get("question_id") or item.get("id"),
                "category": category,
                "source": item.get("src") or item.get("source"),
                "correct_answer": answer_letter,
                "correct_answer_letter": answer_letter,
                "correct_answer_text": options[answer_index],
            },
        }


def _normalize_category(value) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def _category_label(category: str) -> str:
    return category.replace("-", " ")


def _resolve_answer_index(item: dict, options: list[str]) -> int:
    answer_index = item.get("answer_index")
    if answer_index is not None:
        if isinstance(answer_index, str):
            stripped = answer_index.strip()
            if len(stripped) == 1 and stripped.upper() in _LETTERS:
                return _LETTERS.index(stripped.upper())
            return int(stripped)
        return int(answer_index)

    answer = str(item.get("answer", "")).strip()
    if len(answer) == 1 and answer.upper() in _LETTERS:
        return _LETTERS.index(answer.upper())

    normalized_answer = answer.strip().lower()
    for index, option in enumerate(options):
        if option.strip().lower() == normalized_answer:
            return index

    raise ValueError(
        f"MMLU-Pro sample ohne aufloesbare Antwort: {item.get('question_id') or item.get('id')}"
    )
