"""Regressionstest: Request-Body des vLLM-HTTP-Clients.

Hintergrund: Der Client sendete ``chat_template_kwargs`` (und ``top_k``,
``min_p``, ``repetition_penalty``) verschachtelt unter einem Schlüssel
``"extra_body"``. ``extra_body`` ist nur ein Parameter des OpenAI-Python-SDK;
per httpx roh gesendet kommt es bei vLLM als unbekanntes Feld an und wird
ignoriert. Dadurch erreichte ``enable_thinking=False`` die Zielmodelle nie, und
alle Router-Ziele liefen mit dem Template-Default (Reasoning an).

Der Test braucht weder GPU noch Server noch vLLM/torch.

Ausführen (aus dem Repo-Root):
    python -m unittest tests/test_client_request_body.py -v
"""

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_client_module():
    """Lädt nur ``src/inference/client.py``.

    Das Paket ``src.inference`` importiert in seinem ``__init__`` auch den
    Offline-Pfad (torch/vLLM) und den Server-Manager. Für diesen Test wird nur
    der Client gebraucht; fehlt torch (z. B. lokal ohne GPU), ersetzt ein
    minimaler Stub das Modul ``src.inference.llm``.
    """
    import src  # noqa: F401  (Paket-Bootstrap, leichtgewichtig)

    if "src.inference" not in sys.modules:
        pkg = types.ModuleType("src.inference")
        pkg.__path__ = [str(REPO_ROOT / "src" / "inference")]
        sys.modules["src.inference"] = pkg

    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("src.inference.llm")

        @dataclass
        class InferenceResult:  # gleiche Felder wie im Original
            output_text: str
            input_tokens: int
            output_tokens: int
            latency_seconds: float
            tokens_per_second: float
            model_name: str

        stub.InferenceResult = InferenceResult
        sys.modules.setdefault("src.inference.llm", stub)

    spec = importlib.util.spec_from_file_location(
        "src.inference.client", REPO_ROOT / "src" / "inference" / "client.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["src.inference.client"] = module
    spec.loader.exec_module(module)
    return module


client_module = _load_client_module()

# Felder, die vLLMs ChatCompletionRequest auf oberster Ebene kennt und die
# dieser Client verwendet. Ein neues, unbekanntes Feld würde von vLLM still
# ignoriert -- deshalb schlägt der Test dann fehl, statt es durchzulassen.
KNOWN_TOP_LEVEL_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "presence_penalty",
    "stream",
    "stream_options",
    "max_tokens",
    "repetition_penalty",
    "top_k",
    "min_p",
    "chat_template_kwargs",
}


class RequestBodyTest(unittest.TestCase):
    def setUp(self):
        self.client = client_module.VLLMClient(
            base_url="http://127.0.0.1:1",
            model_name="test-model",
            temperature=0.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            temperature_nothinking=0.0,
            top_p_nothinking=1.0,
            presence_penalty_nothinking=2.0,
        )

    def tearDown(self):
        self.client._client.close()

    def body(self, enable_thinking, max_tokens=None):
        return self.client._build_request_body("prompt", enable_thinking, max_tokens)

    def test_no_extra_body_key(self):
        for mode in (True, False):
            self.assertNotIn("extra_body", self.body(mode))

    def test_enable_thinking_reaches_top_level(self):
        for mode in (True, False):
            body = self.body(mode)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": mode})

    def test_vllm_sampling_fields_at_top_level(self):
        body = self.body(True)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["min_p"], 0.0)
        self.assertEqual(body["repetition_penalty"], 1.0)

    def test_top_k_disabled_maps_to_minus_one(self):
        self.client.top_k = 0
        self.assertEqual(self.body(True)["top_k"], -1)

    def test_mode_specific_sampling(self):
        thinking, direct = self.body(True), self.body(False)
        self.assertEqual((thinking["top_p"], thinking["presence_penalty"]), (0.95, 1.5))
        self.assertEqual((direct["top_p"], direct["presence_penalty"]), (1.0, 2.0))

    def test_only_known_fields(self):
        for mode in (True, False):
            unknown = set(self.body(mode, max_tokens=64)) - KNOWN_TOP_LEVEL_FIELDS
            self.assertFalse(unknown, f"vLLM would silently ignore: {sorted(unknown)}")


if __name__ == "__main__":
    unittest.main()
