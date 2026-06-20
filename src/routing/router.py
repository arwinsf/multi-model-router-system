"""LLM-as-a-Router: Prompt-Klassifikation mit einem kleinen Sprachmodell.

Output-Format: Compound-Label z.B. "2T", "1N"
    - Prefix: Modellnummer (1-N)
    - Suffix: T (Thinking an) oder N (Thinking aus)

Constrained Decoding: Nutzt vLLM's guided_choice für valide Labels.
"""

from dataclasses import dataclass

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingResult:
    """Ergebnis einer Routing-Entscheidung."""

    prompt: str
    assigned_model: str  # Model-ID (z.B. "qwen3.5-9b")
    enable_thinking: bool  # Ob Thinking/Reasoning aktiviert werden soll
    router_output: str  # Rohes Label (z.B. "2T")
    routing_failed: bool = False
    error_message: str | None = None


ROUTER_SYSTEM_PROMPT = """\
You are an energy-aware routing controller for a multi-model LLM system.
For each user prompt, choose the lowest model tier that is still comfortably
likely to answer correctly and decide whether reasoning mode should be enabled.

Return ONLY one label in the form <tier number><N|T>.
Do not output any explanation, words, punctuation, or extra text.

N = direct-answer mode. Use only for simple factual recall, definitions,
straightforward extraction, translation, or obvious questions with no ambiguity.
T = reasoning mode. Use whenever the task benefits from deliberate step-by-step
thinking, including multiple-choice questions, math, code, analysis, comparisons,
or anything where reasoning improves correctness.

Available model tiers:
{model_descriptions}

Routing policy:
- First choose the lowest tier that is likely to answer correctly with a
    safety margin. Then decide between N and T.
- Lower tiers have less reasoning headroom. If you choose tier 1 or tier 2
    for anything beyond direct recall or literal rewriting, prefer T.
- Tier 1 with N should be rare and reserved for single-hop facts, literal
    extraction, short translation, or routine rewriting with no reasoning.
- Default to T for any task involving reasoning, analysis, multiple-choice
    with plausible distractors, math, code, science, synthesis, or multi-step
    logic.
- If a multiple-choice question looks answerable by immediate recognition and
    one option is clearly correct without careful elimination, N is acceptable.
- Otherwise use N only for mostly clearly trivial tasks: simple factual recall,
    definitions, greetings, straightforward extraction, translation, or
    routine rewriting where no reasoning is needed.
- When the cost of a wrong answer is high, or when the task appears ambiguous,
    subtle, adversarial, or benchmark-style, prefer T.
- Do not avoid the highest tier just to save energy. Use it when expert knowledge,
    difficult distractors, formal reasoning, complex code generation, or
    many interacting constraints make failure of the 2nd highest tier plausible.
- Difficult multiple-choice questions, nuanced domain knowledge, formal
    reasoning, tricky math, code analysis, or long dependency chains should
    lean upward in tier and use T.
- If the task is clearly direct-recognition and confidence is high, choose N.
- If the task is genuinely borderline or uncertainty remains, choose T.
- If the task is borderline between two tiers, prefer the larger tier.
- The lower / worse you go in tier (esp. Tier 1), the even more likely you should be to
    choose T over N for definitive non-trivial tasks.
- Questions involving organic chemistry reactions, stereochemistry,
    retrosynthesis, or molecular biology mechanisms should ALWAYS use T,
    even if they appear to be simple recall. These domains require
    elimination reasoning.
- "All X except", "Which is NOT", negation-style questions always require
    T mode regardless of apparent difficulty.
- Lastly, don't decide based on the amount of Router few shot examples that are provided
    to you for each tier. The few shot examples are just examples, not a checklist. Use
    your best judgment based on the above principles."""

# Few-Shot Pool. _build_prompt() filtert automatisch auf verfügbare Labels.
ROUTER_FEW_SHOT_EXAMPLES = [
    # Tier 1 (kleinstes Modell): N nur fuer wirklich triviale Faelle
    ("What is the capital of France?", "1N"),
    ("Translate 'Good morning' into German.", "1N"),
    (
        "Rewrite this sentence in the passive voice: 'The team shipped the update yesterday.'",
        "1N",
    ),
    (
        "Which planet is closest to the sun? A. Venus B. Mercury C. Mars D. Jupiter",
        "1N",
    ),
    ("Solve: 3x + 7 = 22. What is x?", "1T"),
    (
        "A recipe uses 3 cups of flour for 12 cookies. How many cups are needed for 30 cookies?",
        "1T",
    ),
    # Tier 2: begrenzte, aber nicht triviale Aufgaben
    (
        "Pick the column's class based on the provided column sample. Choose exactly one of the listed classes. Please respond only with the name of the class.",
        "2N",
    ),
    ("Summarize the key differences between TCP and UDP.", "2T"),
    (
        "Read a short paragraph about climate policy and state the author's main claim in one sentence.",
        "2T",
    ),
    ("Calculate the integral of sin(x) * e^x dx.", "2T"),
    # Tier 3: anspruchsvolle Reasoning- und Benchmark-Aufgaben
    (
        "A biology multiple-choice question asks which organelle produces most ATP in eukaryotic cells, and one option is mitochondrion while the distractors are clearly unrelated.",
        "3N",
    ),
    (
        "Please create a valid join mapping between CSV Table A and CSV Table B.",
        "3N",
    ),
    (
        "A biology multiple-choice question asks which membrane transport mechanism best explains an observed ion gradient, with four plausible distractors.",
        "3T",
    ),
    (
        "Debug a Python function that sometimes drops duplicates in the wrong order and explain the likely bug.",
        "3T",
    ),
    (
        "Compare two policy proposals and identify the strongest trade-offs in terms of incentives, fairness, and implementation risk.",
        "3T",
    ),
    # Tier 4: hochriskante oder expertische Aufgaben mit vielen Abhaengigkeiten
    (
        "Given a Python function with a subtle off-by-one error that only manifests on certain edge cases, identify the bug and propose a fix.",
        "4T",
    ),
    (
        "Prove that every finite subgroup of the multiplicative group of a field is cyclic.",
        "4T",
    ),
    (
        "Implement a recursive red-black tree in C++ with all rotation cases.",
        "4T",
    ),
    (
        "You are given a failing multi-file web service with race conditions, database deadlocks, and flaky tests. Identify the likely root causes and propose a repair plan.",
        "4T",
    ),
    (
        "trans-cinnamaldehyde is treated with methylmagnesium bromide forming product 1. "
        "1 is treated with PCC forming product 2. 2 is treated with a Wittig reagent "
        "forming product 3. How many carbon atoms are in product 3?",
        "4T",
    ),
    (
        "Given measured phase shifts in an elastic scattering experiment, calculate "
        "the imaginary part of the forward scattering amplitude.",
        "4T",
    ),
]


def build_router_system_prompt(models: list[dict]) -> str:
    """Erzeugt den Router-System-Prompt aus den aktuell verfügbaren Modellen."""
    descriptions = []
    sorted_models = sorted(models, key=lambda m: m.get("tier", 1))
    for i, model in enumerate(sorted_models):
        label = str(i + 1)
        descriptions.append(f"- {label}: {model.get('description', '')}")

    model_descriptions = "\n".join(descriptions)
    return ROUTER_SYSTEM_PROMPT.format(model_descriptions=model_descriptions)


def build_router_few_shot_examples(models: list[dict]) -> list[dict[str, str]]:
    """Filtert Few-Shot-Beispiele auf die aktuell verfügbaren Tier-Labels."""
    sorted_models = sorted(models, key=lambda m: m.get("tier", 1))
    valid_prefixes = {str(i + 1) for i, _ in enumerate(sorted_models)}

    filtered_examples = []
    for example_prompt, example_label in ROUTER_FEW_SHOT_EXAMPLES:
        prefix = example_label[:-1]
        if prefix in valid_prefixes:
            filtered_examples.append(
                {
                    "prompt": example_prompt,
                    "label": example_label,
                }
            )

    return filtered_examples


class LLMRouter:
    """LLM-as-a-Router: Multi-Modell-Klassifikation mit Thinking-Entscheidung.

    Nutzt vLLM Offline-Modus für schnelle Inferenz. Der System Prompt wird
    dynamisch aus der Modell-Konfiguration generiert. Constrained Decoding
    via guided_choice stellt valide Compound-Labels sicher.
    """

    def __init__(
        self,
        router_model: str,
        models: list[dict],
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192,
        max_output_tokens: int = 10,
        dtype: str = "auto",
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
    ):
        self.router_model = router_model
        self.models = models
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_output_tokens = max_output_tokens
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.tensor_parallel_size = tensor_parallel_size

        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

        # Tier-Labels: "1" -> model_id, "2" -> model_id, ...
        sorted_models = sorted(models, key=lambda m: m.get("tier", 1))
        self._label_to_id: dict[str, str] = {}
        self._id_to_label: dict[str, str] = {}
        for i, m in enumerate(sorted_models):
            label = str(i + 1)
            self._label_to_id[label] = m["id"]
            self._id_to_label[m["id"]] = label

        # Gültige Compound-Labels (Tier + Thinking-Modus)
        self._valid_choices: list[str] = []
        for label in self._label_to_id:
            self._valid_choices.append(f"{label}N")
            self._valid_choices.append(f"{label}T")

        self._fallback_id = sorted_models[-1]["id"]
        self._sorted_models = sorted_models
        self._system_prompt = self._build_system_prompt()
        self._few_shot_examples = build_router_few_shot_examples(self._sorted_models)

    def _build_system_prompt(self) -> str:
        return build_router_system_prompt(self._sorted_models)

    def load(self) -> None:
        """Lädt das Router-Modell mit vLLM."""
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        logger.info(
            "Lade Router-Modell: %s (gmu=%.4f, dtype=%s, tp=%d, max_model_len=%d, enforce_eager=%s)",
            self.router_model,
            self.gpu_memory_utilization,
            self.dtype,
            self.tensor_parallel_size,
            self.max_model_len,
            self.enforce_eager,
        )

        try:
            self._llm = LLM(
                model=self.router_model,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                enforce_eager=self.enforce_eager,
                limit_mm_per_prompt={
                    "image": 0,
                    "video": 0,
                    "audio": 0,
                },  # Text-only, kein Multimodal-Encoder
            )
        except Exception:
            logger.exception(
                "Router-Modell konnte nicht geladen werden: %s", self.router_model
            )
            raise

        self._tokenizer = self._llm.get_tokenizer()

        # Constrained Decoding via guided_choice
        logger.info("Gültige Router-Labels: %s", self._valid_choices)
        guided = StructuredOutputsParams(choice=self._valid_choices)

        self._sampling_params = SamplingParams(
            max_tokens=self.max_output_tokens,
            temperature=0.0,
            structured_outputs=guided,
        )

        logger.info("Router-Modell geladen: %s", self.router_model)
        logger.info("Router-System-Prompt:\n%s", self._system_prompt)

    def _build_prompt(self, user_prompt: str) -> str:
        """Baut den vollständigen Prompt via Chat-Template.

        Nutzt Few-Shot-Beispiele als Multi-Turn-Messages. Beispiele mit
        Labels die nicht im aktuellen Label-Set sind werden übersprungen.
        """
        messages = [{"role": "system", "content": self._system_prompt}]

        for example in self._few_shot_examples:
            messages.append({"role": "user", "content": example["prompt"]})
            messages.append({"role": "assistant", "content": example["label"]})

        messages.append({"role": "user", "content": user_prompt})

        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _parse_output(self, output_text: str) -> tuple[str, bool]:
        """Extrahiert Model-ID und Thinking-Flag aus dem Router-Output."""
        text = output_text.strip().upper()

        if len(text) >= 2 and text[-1] in ("T", "N"):
            label = text[:-1]
            thinking = text[-1] == "T"

            if label in self._label_to_id:
                return self._label_to_id[label], thinking

        # Fallback: längste Label-Matches zuerst
        for label, model_id in sorted(
            self._label_to_id.items(), key=lambda x: -len(x[0])
        ):
            if label in text:
                thinking = "T" in text
                return model_id, thinking

        return self._fallback_id, True

    def route(self, prompt: str) -> RoutingResult:
        """Routet einen einzelnen Prompt."""
        if self._llm is None:
            raise RuntimeError("Router nicht geladen! Rufe load() auf.")

        try:
            full_prompt = self._build_prompt(prompt)
            outputs = self._llm.generate([full_prompt], self._sampling_params)
            torch.cuda.synchronize()

            raw_output = outputs[0].outputs[0].text
            model_id, thinking = self._parse_output(raw_output)

            return RoutingResult(
                prompt=prompt,
                assigned_model=model_id,
                enable_thinking=thinking,
                router_output=raw_output.strip(),
            )

        except Exception as e:
            error_msg = f"Routing fehlgeschlagen: {e}"
            logger.error(error_msg)

            return RoutingResult(
                prompt=prompt,
                assigned_model=self._fallback_id,
                enable_thinking=True,
                router_output="",
                routing_failed=True,
                error_message=error_msg,
            )

    def route_batch(self, prompts: list[str]) -> list[RoutingResult]:
        """Routet mehrere Prompts gleichzeitig (vLLM Continuous Batching)."""
        if self._llm is None:
            raise RuntimeError("Router nicht geladen! Rufe load() auf.")

        full_prompts = [self._build_prompt(p) for p in prompts]

        try:
            outputs = self._llm.generate(full_prompts, self._sampling_params)
            torch.cuda.synchronize()

            results = []
            for prompt, output in zip(prompts, outputs):
                raw_output = output.outputs[0].text
                model_id, thinking = self._parse_output(raw_output)
                results.append(
                    RoutingResult(
                        prompt=prompt,
                        assigned_model=model_id,
                        enable_thinking=thinking,
                        router_output=raw_output.strip(),
                    )
                )
            return results

        except Exception as e:
            error_msg = f"Batch-Routing fehlgeschlagen: {e}"
            logger.error(error_msg)

            return [
                RoutingResult(
                    prompt=p,
                    assigned_model=self._fallback_id,
                    enable_thinking=True,
                    router_output="",
                    routing_failed=True,
                    error_message=error_msg,
                )
                for p in prompts
            ]

    def unload(self) -> None:
        """Entlädt das Router-Modell."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._tokenizer = None
            self._sampling_params = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Router-Modell entladen: %s", self.router_model)

    @property
    def is_loaded(self) -> bool:
        """Prüft ob das Router-Modell geladen ist."""
        return self._llm is not None

    @property
    def valid_choices(self) -> list[str]:
        """Alle gültigen Compound-Labels."""
        return self._valid_choices.copy()
