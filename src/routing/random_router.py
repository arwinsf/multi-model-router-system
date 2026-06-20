"""Random-Router: Zufällige Modell- und Thinking-Zuweisung.

Dient als Vergleichsbasis (Baseline), um zu messen wie schlecht
zufälliges Routing gegenüber dem trainierten LLM-Router performt.
Benötigt kein Router-Modell und keinen GPU-VRAM.
"""

import random

from src.routing.router import RoutingResult


class RandomRouter:
    """Zufälliger Router: Wählt Modell und Thinking-Modus per Zufall.

    Generiert die gleichen RoutingResult-Objekte wie LLMRouter,
    aber ohne tatsächliche LLM-Inferenz. Für jede Anfrage wird
    zufällig ein Modell und T/N gewählt.
    """

    def __init__(self, models: list[dict], seed: int | None = None):
        """Initialisiert den Random-Router.

        Args:
            models: Liste von Model-Configs mit id, tier (sortiert nach tier).
            seed: Optionaler Seed für Reproduzierbarkeit.
        """
        # Modelle nach Tier sortieren und nummerieren (wie LLMRouter)
        sorted_models = sorted(models, key=lambda m: m.get("tier", 1))
        self._models = sorted_models
        self._model_ids = [m["id"] for m in sorted_models]
        self._rng = random.Random(seed)

    def route(self, prompt: str) -> RoutingResult:
        """Routet einen einzelnen Prompt zufällig."""
        model_idx = self._rng.randint(0, len(self._model_ids) - 1)
        model_id = self._model_ids[model_idx]
        enable_thinking = self._rng.choice([True, False])

        label_num = model_idx + 1
        label_thinking = "T" if enable_thinking else "N"
        label = f"{label_num}{label_thinking}"

        return RoutingResult(
            prompt=prompt,
            assigned_model=model_id,
            enable_thinking=enable_thinking,
            router_output=label,
            routing_failed=False,
        )

    def route_batch(self, prompts: list[str]) -> list[RoutingResult]:
        """Routet einen Batch von Prompts zufällig."""
        return [self.route(prompt) for prompt in prompts]
