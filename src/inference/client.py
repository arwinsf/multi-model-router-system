"""HTTP-Client für vLLM OpenAI-kompatible API.

Ersetzt die offline VLLMInference.generate()/generate_batch() durch HTTP-Calls
an laufende vLLM-Server. Unterstützt per-Request Thinking/Reasoning-Toggle.

Thinking-Steuerung per Request:
    extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from src.inference.llm import InferenceResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Read-Timeout als Inaktivitätsgrenze zwischen zwei Streaming-Events.
# Lange Generationen dürfen beliebig weiterlaufen, solange der Server Tokens
# weiter streamt. Nur echte Stalls sollen hier abbrechen.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
HTTP_CONNECT_TIMEOUT_SECONDS = 30.0
HTTP_WRITE_TIMEOUT_SECONDS = 30.0
HTTP_POOL_TIMEOUT_SECONDS = 30.0
HTTP_ERROR_BODY_MAX_CHARS = 4000
SERVER_LOG_EXCERPT_MAX_CHARS = 16000
DEFAULT_MAX_CONCURRENT_REQUESTS = 32


class VLLMClient:
    """HTTP-Client für Inferenz gegen einen vLLM-Server.

    Nutzt die OpenAI-kompatible Chat Completions API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        temperature_nothinking: float | None = None,
        top_p_nothinking: float | None = None,
        presence_penalty_nothinking: float | None = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        server_log_path: str | Path | None = None,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        # Non-Thinking-Modus: gleiche Temperature, aber top_p=1.0 und presence_penalty=2.0
        self.temperature_nothinking = (
            temperature_nothinking
            if temperature_nothinking is not None
            else temperature
        )
        self.top_p_nothinking = (
            top_p_nothinking if top_p_nothinking is not None else top_p
        )
        self.presence_penalty_nothinking = (
            presence_penalty_nothinking
            if presence_penalty_nothinking is not None
            else presence_penalty
        )
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.server_log_path = Path(server_log_path) if server_log_path else None
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))
        self._client = httpx.Client(timeout=self._build_timeout())

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    @classmethod
    def _format_server_log_excerpt(cls, text: str) -> str:
        if not text:
            return ""

        lines = text.splitlines()
        root_needles = (
            "traceback",
            "runtimeerror",
            "outofmemory",
            "out of memory",
            "cuda error",
            "assertion",
            "valueerror",
            "no available memory",
            "available kv cache memory",
            "failed",
            "exception",
        )
        secondary_needles = (
            "enginedeaderror",
            "enginecore encountered an issue",
            "internal server error",
        )

        root_indices = []
        secondary_indices = []
        for idx, line in enumerate(lines):
            lower = line.lower()
            if any(needle in lower for needle in root_needles):
                if "apisserver" not in lower and "apiserver" not in lower:
                    root_indices.append(idx)
                else:
                    secondary_indices.append(idx)
            elif any(needle in lower for needle in secondary_needles):
                secondary_indices.append(idx)

        selected_indices = (root_indices or secondary_indices)[:40]
        if not selected_indices:
            return cls._truncate_text(text, SERVER_LOG_EXCERPT_MAX_CHARS)

        selected: list[str] = []
        seen: set[int] = set()
        for idx in selected_indices:
            start = max(0, idx - 3)
            end = min(len(lines), idx + 8)
            for line_no in range(start, end):
                if line_no in seen:
                    continue
                seen.add(line_no)
                selected.append(lines[line_no])

        tail = "\n".join(lines[-80:])
        excerpt = "\n".join(selected)
        if tail and tail not in excerpt:
            excerpt = f"{excerpt}\n\n--- Log-Ende ---\n{tail}"

        return cls._truncate_text(excerpt, SERVER_LOG_EXCERPT_MAX_CHARS)

    def _server_log_tail(self) -> str:
        if self.server_log_path is None or not self.server_log_path.exists():
            return ""

        try:
            return self._format_server_log_excerpt(
                self.server_log_path.read_text(errors="replace")
            )
        except Exception:
            return ""

    def _http_error_message(
        self,
        response: httpx.Response,
        prompt: str,
        enable_thinking: bool,
        max_tokens: int | None,
    ) -> str:
        body_excerpt = self._truncate_text(
            response.text.strip(),
            HTTP_ERROR_BODY_MAX_CHARS,
        )
        server_log_tail = self._server_log_tail()

        parts = [
            f"vLLM HTTP {response.status_code} für {self.model_name} "
            f"({response.request.method} {response.request.url})",
            f"thinking={enable_thinking}, prompt_chars={len(prompt)}, max_tokens={max_tokens}",
        ]
        if body_excerpt:
            parts.append(f"Response-Body:\n{body_excerpt}")
        if server_log_tail:
            parts.append(f"vLLM-Serverlog-Ende:\n{server_log_tail}")

        return "\n".join(parts)

    def _request_error_message(
        self,
        request_error: httpx.RequestError,
        prompt: str,
        enable_thinking: bool,
        max_tokens: int | None,
    ) -> str:
        server_log_tail = self._server_log_tail()
        parts = [
            f"vLLM-Verbindungsfehler für {self.model_name} "
            f"({self.base_url}/v1/chat/completions): {request_error}",
            f"thinking={enable_thinking}, prompt_chars={len(prompt)}, max_tokens={max_tokens}",
        ]
        if server_log_tail:
            parts.append(f"vLLM-Serverlog-Ende:\n{server_log_tail}")

        return "\n".join(parts)

    def _build_timeout(self) -> httpx.Timeout:
        """Baut HTTPX-Timeouts mit Read-Timeout als Inaktivitätsgrenze."""
        return httpx.Timeout(
            connect=HTTP_CONNECT_TIMEOUT_SECONDS,
            read=self.request_timeout_seconds,
            write=HTTP_WRITE_TIMEOUT_SECONDS,
            pool=HTTP_POOL_TIMEOUT_SECONDS,
        )

    def _build_request_body(
        self,
        prompt: str,
        enable_thinking: bool,
        max_tokens: int | None,
    ) -> dict:
        """Erzeugt den OpenAI-kompatiblen Request-Body."""
        messages = [{"role": "user", "content": prompt}]

        temp = self.temperature if enable_thinking else self.temperature_nothinking
        tp = self.top_p if enable_thinking else self.top_p_nothinking
        pp = (
            self.presence_penalty
            if enable_thinking
            else self.presence_penalty_nothinking
        )

        body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "top_p": tp,
            "presence_penalty": pp,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {
                "repetition_penalty": self.repetition_penalty,
                "top_k": self.top_k if self.top_k > 0 else -1,
                "min_p": self.min_p,
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                },
            },
        }

        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        return body

    @staticmethod
    def _extract_delta_text(choice: dict) -> str:
        """Extrahiert sichtbaren Assistant-Text aus einem Streaming-Chunk."""
        delta = choice.get("delta") or {}
        content = delta.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)

        message = choice.get("message") or {}
        message_content = message.get("content")
        if isinstance(message_content, str):
            return message_content

        return ""

    def generate(
        self,
        prompt: str,
        enable_thinking: bool = False,
        max_tokens: int | None = None,
    ) -> InferenceResult:
        """Generiert eine Antwort für einen einzelnen Prompt.

        Args:
            prompt: Der Eingabe-Prompt.
            enable_thinking: Ob Thinking/Reasoning aktiviert werden soll.
            max_tokens: Maximale Output-Tokens (None = Modell-Default).

        Returns:
            InferenceResult mit generiertem Text und Metriken.
        """
        return self._single_request(prompt, enable_thinking, max_tokens)

    def generate_batch(
        self,
        prompts: list[str],
        enable_thinking: bool | list[bool] = False,
        max_tokens: int | None = None,
    ) -> list[InferenceResult]:
        """Generiert Antworten für mehrere Prompts.

        Sendet alle Requests an den vLLM-Server, der intern
        Continuous Batching nutzt.

        Args:
            prompts: Liste von Prompts.
            enable_thinking: Bool (für alle gleich) oder Liste pro Prompt.
            max_tokens: Maximale Output-Tokens pro Antwort.

        Returns:
            Liste von InferenceResults (gleiche Reihenfolge wie Eingabe).
        """
        if isinstance(enable_thinking, bool):
            thinking_flags = [enable_thinking] * len(prompts)
        else:
            thinking_flags = enable_thinking

        start_time = time.perf_counter()

        # Parallele Requests an den Server (vLLM batcht intern)
        results: list[InferenceResult | None] = [None] * len(prompts)

        max_workers = min(len(prompts), self.max_concurrent_requests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (prompt, thinking) in enumerate(zip(prompts, thinking_flags)):
                future = executor.submit(
                    self._single_request, prompt, thinking, max_tokens
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        end_time = time.perf_counter()
        total_latency = end_time - start_time

        # Latenz auf Batch-Level setzen (wie bei offline-Modus)
        for r in results:
            if r is not None:
                r.latency_seconds = total_latency
                if total_latency > 0:
                    r.tokens_per_second = r.output_tokens / total_latency

        return results

    def _single_request(
        self,
        prompt: str,
        enable_thinking: bool,
        max_tokens: int | None,
    ) -> InferenceResult:
        """Einzelner HTTP-Request (für ThreadPool).

        Nutzt Streaming, damit der Read-Timeout nur auf echte Inaktivität
        zwischen Server-Events wirkt statt auf die gesamte Generierungsdauer.
        """
        body = self._build_request_body(prompt, enable_thinking, max_tokens)

        output_parts: list[str] = []
        usage: dict = {}
        received_chunks = 0
        received_chars = 0
        start_time = time.perf_counter()

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=body,
                timeout=self._build_timeout(),
            ) as response:
                if response.is_error:
                    response.read()
                    message = self._http_error_message(
                        response,
                        prompt,
                        enable_thinking,
                        max_tokens,
                    )
                    logger.error(message)
                    raise httpx.HTTPStatusError(
                        message,
                        request=response.request,
                        response=response,
                    )

                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    if payload == "[DONE]":
                        break

                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Überspringe ungültiges Streaming-Event: %s", payload
                        )
                        continue

                    chunk_usage = data.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = chunk_usage

                    for choice in data.get("choices", []):
                        chunk_text = self._extract_delta_text(choice)
                        if chunk_text:
                            output_parts.append(chunk_text)
                            received_chunks += 1
                            received_chars += len(chunk_text)

        except httpx.TimeoutException:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Stream-Inaktivitäts-Timeout nach %.0fs für %s (thinking=%s, elapsed=%.1fs, chunks=%d, chars=%d).",
                self.request_timeout_seconds,
                self.model_name,
                enable_thinking,
                elapsed,
                received_chunks,
                received_chars,
            )
            raise
        except httpx.RequestError as request_error:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "%s\nelapsed=%.1fs, chunks=%d, chars=%d",
                self._request_error_message(
                    request_error,
                    prompt,
                    enable_thinking,
                    max_tokens,
                ),
                elapsed,
                received_chunks,
                received_chars,
            )
            raise

        end_time = time.perf_counter()
        output_text = "".join(output_parts)

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        if output_text and output_tokens == 0:
            logger.warning(
                "Kein usage.completion_tokens im Streaming-Response für %s erhalten.",
                self.model_name,
            )

        latency = end_time - start_time
        tokens_per_second = output_tokens / latency if latency > 0 else 0

        return InferenceResult(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            tokens_per_second=tokens_per_second,
            model_name=self.model_name,
        )

    def close(self) -> None:
        """Schliesst den HTTP-Client."""
        self._client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
