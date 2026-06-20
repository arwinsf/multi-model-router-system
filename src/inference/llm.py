"""LLM-Inferenz mit vLLM.

Optimiert für NVIDIA GPUs (RTX 3090, A100).

vLLM bietet:
    - Continuous Batching
    - PagedAttention für effiziente KV-Cache Nutzung
    - Optimierte CUDA Kernels
    - Tensor Parallelism und Data Parallelism für Multi-GPU
"""

from dataclasses import dataclass
import multiprocessing as mp
from typing import Optional
import os
import time
import torch

# spawn-Kontext: CUDA kann nicht in geforkten Prozessen re-initialisiert werden.
# spawn erstellt saubere Kind-Prozesse ohne geerbten CUDA-State.
_mp_ctx = mp.get_context("spawn")


@dataclass
class InferenceResult:
    """Ergebnis einer LLM-Inferenz."""

    output_text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    tokens_per_second: float
    model_name: str


def _dp_worker(
    rank: int,
    gpu_id: int,
    llm_kwargs: dict,
    sampling_kwargs: dict,
    enable_thinking: bool,
    input_queue,
    output_queue,
):
    """Worker-Prozess für Data Parallelism.

    Jeder Worker lädt eine eigene vLLM-Instanz auf seiner GPU
    und wartet auf Prompts aus der Input-Queue. Komplett unabhängig —
    keine vLLM-DP-Koordination nötig.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from vllm import LLM, SamplingParams

    print(f"[DP Worker {rank}] Lade Modell auf GPU {gpu_id}")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(**sampling_kwargs)
    print(f"[DP Worker {rank}] Modell geladen")

    output_queue.put(("ready", rank))

    while True:
        msg = input_queue.get()
        if msg is None:  # Sentinel → beenden
            break

        prompts, batch_id = msg

        # Chat-Template im Worker anwenden (Tokenizer ist hier verfügbar)
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            try:
                f = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                f = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            formatted.append(f)

        outputs = llm.generate(formatted, sampling_params)

        # Ergebnisse als serialisierbare Dicts zurückgeben
        results = []
        for output in outputs:
            results.append(
                {
                    "text": output.outputs[0].text,
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                }
            )

        output_queue.put(("result", rank, batch_id, results))

    # Aufräumen
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VLLMInference:
    """Hochperformante LLM-Inferenz mit vLLM.

    Unterstützt Einzel- und Batch-Inferenz.
    Bei data_parallel_size > 1: Multi-Prozess Data Parallelism.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = -1,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        enforce_eager: bool = False,
        enable_thinking: bool = False,
        nvlink_available: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.enable_thinking = enable_thinking
        self.nvlink_available = nvlink_available

        # Single-Process (DP=1)
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

        # Multi-Process (DP>1)
        self._workers: list = []
        self._input_queues: list = []
        self._output_queues: list = []

    def _build_llm_kwargs(self) -> dict:
        """Erstellt die vLLM LLM-Kwargs (gemeinsam für DP=1 und DP>1 Worker)."""
        llm_kwargs = {
            "model": self.model_name,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": True,
            "enforce_eager": self.enforce_eager,
            "limit_mm_per_prompt": {
                "image": 0,
                "video": 0,
                "audio": 0,
            },  # Text-only, kein Multimodal-Encoder
        }

        if self.tensor_parallel_size > 1:
            # Custom All-Reduce deaktivieren: schlägt auf A100 fehl
            # (cuda error custom_all_reduce.cuh:455). NCCL über NVLink ist ohnehin optimal.
            llm_kwargs["disable_custom_all_reduce"] = True

        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len

        return llm_kwargs

    def _build_sampling_kwargs(self) -> dict:
        """Erstellt die SamplingParams-Kwargs."""
        sampling_kwargs = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.temperature > 0 else 0,
            "top_p": self.top_p if self.temperature > 0 else 1.0,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.top_k > 0:
            sampling_kwargs["top_k"] = self.top_k
        if self.min_p > 0:
            sampling_kwargs["min_p"] = self.min_p
        if self.presence_penalty != 0.0:
            sampling_kwargs["presence_penalty"] = self.presence_penalty
        return sampling_kwargs

    def load(self) -> None:
        """Lädt das Modell mit vLLM.

        Bei DP=1: Direkte LLM-Instanz im Hauptprozess.
        Bei DP>1: Spawnt Worker-Prozesse (einer pro GPU-Replika).
        """
        print(f"Lade Modell mit vLLM: {self.model_name}")
        print(f"  Dtype: {self.dtype}")
        print(f"  GPU Memory Utilization: {self.gpu_memory_utilization}")
        print(f"  Tensor Parallel Size: {self.tensor_parallel_size}")
        print(f"  Data Parallel Size: {self.data_parallel_size}")
        print(f"  Enable Thinking: {self.enable_thinking}")

        if self.data_parallel_size <= 1:
            self._load_single_process()
        else:
            self._load_multi_process()

    def _load_single_process(self) -> None:
        """Lädt das Modell direkt im Hauptprozess (DP=1)."""
        import os

        from vllm import LLM, SamplingParams

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        llm_kwargs = self._build_llm_kwargs()
        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(**self._build_sampling_kwargs())

        print(f"Modell geladen: {self.model_name}")

    def _load_multi_process(self) -> None:
        """Spawnt DP Worker-Prozesse (DP>1).

        Jeder Worker lädt eine komplett unabhängige vLLM-Instanz auf seiner GPU.
        Keine vLLM-interne DP-Koordination — nur CUDA_VISIBLE_DEVICES-Isolation.
        Kommunikation über multiprocessing.Queue.
        """
        dp_size = self.data_parallel_size
        tp_size = self.tensor_parallel_size

        llm_kwargs = self._build_llm_kwargs()
        sampling_kwargs = self._build_sampling_kwargs()

        print(f"Starte {dp_size} DP-Worker (je TP={tp_size})")

        for rank in range(dp_size):
            input_q = _mp_ctx.Queue()
            output_q = _mp_ctx.Queue()
            self._input_queues.append(input_q)
            self._output_queues.append(output_q)

            # GPU-Zuweisung: Worker i bekommt GPUs [i*tp .. (i+1)*tp-1]
            gpu_start = rank * tp_size
            gpu_id = gpu_start  # Bei TP=1 pro Worker ist das eine einzelne GPU

            worker = _mp_ctx.Process(
                target=_dp_worker,
                args=(
                    rank,
                    gpu_id,
                    llm_kwargs,
                    sampling_kwargs,
                    self.enable_thinking,
                    input_q,
                    output_q,
                ),
                daemon=False,
            )
            worker.start()
            self._workers.append(worker)

        # Warte bis alle Worker bereit sind
        ready_count = 0
        while ready_count < dp_size:
            for q in self._output_queues:
                try:
                    msg = q.get(timeout=600)  # 10 Min Timeout für Modell-Laden
                    if msg[0] == "ready":
                        ready_count += 1
                        print(f"[DP Worker {msg[1]}] Bereit")
                except Exception:
                    pass

        # Tokenizer im Hauptprozess laden (nur für is_loaded-Check etc.)
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        print(f"Alle {dp_size} DP-Worker bereit: {self.model_name}")

    def _apply_chat_template(self, prompt: str) -> str:
        """Wendet das Chat-Template auf einen Prompt an.

        Bei Modellen mit Thinking-Support (z.B. Qwen3.5) wird
        enable_thinking entsprechend gesetzt.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, prompt: str) -> InferenceResult:
        """Generiert eine Antwort für einen einzelnen Prompt."""
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[InferenceResult]:
        """Generiert Antworten für mehrere Prompts gleichzeitig.

        Bei DP=1: Direkte vLLM-Inferenz mit Continuous Batching.
        Bei DP>1: Aufteilen auf Worker-Prozesse, parallel verarbeiten,
        Ergebnisse in Original-Reihenfolge zusammenführen.

        Args:
            prompts: Liste von Prompts.

        Returns:
            Liste von InferenceResults (gleiche Reihenfolge wie Eingabe).
        """
        if self.data_parallel_size <= 1:
            return self._generate_batch_single(prompts)
        else:
            return self._generate_batch_multi(prompts)

    def _generate_batch_single(self, prompts: list[str]) -> list[InferenceResult]:
        """Batch-Inferenz im Single-Process-Modus (DP=1)."""
        if self._llm is None:
            raise RuntimeError("Modell nicht geladen! Rufe load() auf.")

        formatted_prompts = [self._apply_chat_template(p) for p in prompts]

        start_time = time.perf_counter()
        outputs = self._llm.generate(formatted_prompts, self._sampling_params)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_latency = end_time - start_time

        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            tokens_per_second = (
                output_tokens / total_latency if total_latency > 0 else 0
            )

            results.append(
                InferenceResult(
                    output_text=generated_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_seconds=total_latency,
                    tokens_per_second=tokens_per_second,
                    model_name=self.model_name,
                )
            )

        return results

    def _generate_batch_multi(self, prompts: list[str]) -> list[InferenceResult]:
        """Batch-Inferenz im Multi-Process-Modus (DP>1).

        Teilt Prompts gleichmäßig auf Worker auf, sammelt Ergebnisse,
        fügt sie in Original-Reihenfolge zusammen.
        """
        if not self._workers:
            raise RuntimeError("Modell nicht geladen! Rufe load() auf.")

        dp_size = self.data_parallel_size
        n = len(prompts)

        # Prompts aufteilen (Floor-Division + Rest, wie vLLM-Beispiel)
        floor = n // dp_size
        remainder = n % dp_size

        def start_idx(rank):
            return rank * floor + min(rank, remainder)

        start_time = time.perf_counter()

        # Prompts an Worker senden
        batch_id = id(prompts)  # Eindeutige Batch-ID
        for rank in range(dp_size):
            s = start_idx(rank)
            e = start_idx(rank + 1)
            worker_prompts = prompts[s:e]
            self._input_queues[rank].put((worker_prompts, batch_id))

        # Ergebnisse sammeln (ungeordnet)
        rank_results: dict[int, list[dict]] = {}
        collected = 0
        while collected < dp_size:
            for q in self._output_queues:
                try:
                    msg = q.get(timeout=1800)  # 30 Min Timeout pro Batch
                    if msg[0] == "result":
                        _, rank, _, results = msg
                        rank_results[rank] = results
                        collected += 1
                except Exception:
                    pass

        end_time = time.perf_counter()
        total_latency = end_time - start_time

        # Ergebnisse in Original-Reihenfolge zusammenführen
        all_results = []
        for rank in range(dp_size):
            for r in rank_results[rank]:
                output_tokens = r["output_tokens"]
                tokens_per_second = (
                    output_tokens / total_latency if total_latency > 0 else 0
                )
                all_results.append(
                    InferenceResult(
                        output_text=r["text"],
                        input_tokens=r["input_tokens"],
                        output_tokens=output_tokens,
                        latency_seconds=total_latency,
                        tokens_per_second=tokens_per_second,
                        model_name=self.model_name,
                    )
                )

        return all_results

    def unload(self) -> None:
        """Entlädt das Modell.

        Bei DP=1: Direkte Freigabe.
        Bei DP>1: Sentinel an Worker senden, Prozesse beenden.
        """
        if self._workers:
            # Multi-Process: Worker beenden
            for q in self._input_queues:
                q.put(None)  # Sentinel
            for worker in self._workers:
                worker.join(timeout=60)
                if worker.is_alive():
                    worker.kill()
            self._workers.clear()
            self._input_queues.clear()
            self._output_queues.clear()
        elif self._llm is not None:
            # Single-Process
            del self._llm
            self._llm = None
            self._sampling_params = None

        self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Modell entladen: {self.model_name}")

    @property
    def is_loaded(self) -> bool:
        """Prüft ob das Modell geladen ist."""
        return self._llm is not None or len(self._workers) > 0
