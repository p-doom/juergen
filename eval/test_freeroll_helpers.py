from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


def _install_import_stubs() -> None:
    """Let these helper tests run without syncing the heavy eval venv."""
    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        image = types.ModuleType("PIL.Image")
        image.Image = object
        pil.Image = image
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image
    if "requests" not in sys.modules:
        requests = types.ModuleType("requests")
        requests.RequestException = Exception
        requests.Session = lambda: None
        requests.get = lambda *args, **kwargs: None
        requests.post = lambda *args, **kwargs: None
        sys.modules["requests"] = requests


sys.path.insert(0, str(Path(__file__).resolve().parent))
_install_import_stubs()

import freeroll  # noqa: E402
import osworld_runtime  # noqa: E402
import osworld_vm_client  # noqa: E402


class FreerollHelperTests(unittest.TestCase):
    def test_parse_instructions_splits_nonempty_noncomment_lines(self) -> None:
        self.assertEqual(
            freeroll._parse_instructions("first\n\n# skip\nsecond\n"),
            ["first", "second"],
        )

    def test_parse_instructions_preserves_legacy_empty_behavior(self) -> None:
        self.assertEqual(freeroll._parse_instructions(None), [None])
        self.assertEqual(freeroll._parse_instructions("\n# only comments"), [None])

    def test_terminate_is_detected_as_first_line_token(self) -> None:
        self.assertTrue(freeroll._is_terminate("TERMINATE"))
        self.assertTrue(freeroll._is_terminate(" TERMINATE\nignored"))
        self.assertFalse(freeroll._is_terminate("NO_OP"))

    def test_rdev_key_mapping_covers_typing_punctuation_and_digits(self) -> None:
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("KeyA"), "a")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Digit1"), "1")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Comma"), ",")
        self.assertEqual(osworld_vm_client._rdev_to_pyautogui("Quote"), "'")


class SamplingOverrideTests(unittest.TestCase):
    """The request body must carry exactly the knobs that were pinned.

    Anything omitted is resolved by sglang from the checkpoint's
    generation_config.json, so an accidentally-sent default (e.g. top_p=1.0)
    would silently override the model's own recommendation.
    """

    def _post_body(self, **call_kwargs) -> dict:
        captured: dict = {}

        class _Resp:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"choices": [{"message": {"content": "0 0 0"}}]}

        def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            captured["body"] = json
            return _Resp()

        real_post = osworld_runtime.requests.post
        osworld_runtime.requests.post = _fake_post
        try:
            osworld_runtime._call_model(
                sglang_url="http://localhost:30000/v1", api_key="k", model="m",
                system_prompt="sys", instruction=None, recent_frames=[],
                **call_kwargs,
            )
        finally:
            osworld_runtime.requests.post = real_post
        return captured["body"]

    def test_unset_knobs_are_absent_from_the_request(self) -> None:
        body = self._post_body(max_tokens=64, temperature=0.0)
        self.assertEqual(
            set(body), {"model", "messages", "max_tokens", "temperature"})

    def test_pinned_knobs_are_sent_verbatim(self) -> None:
        sampling = osworld_runtime.SamplingOverrides(
            top_p=0.8, top_k=20, repetition_penalty=1.0, presence_penalty=1.5)
        body = self._post_body(
            max_tokens=512, temperature=0.7, sampling=sampling)
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["repetition_penalty"], 1.0)
        self.assertEqual(body["presence_penalty"], 1.5)
        # Never-set knobs stay inherited even when siblings are pinned.
        self.assertNotIn("min_p", body)
        self.assertNotIn("frequency_penalty", body)

    def test_falsy_values_are_pinned_not_dropped(self) -> None:
        fields = osworld_runtime.SamplingOverrides(
            top_k=-1, presence_penalty=0.0).to_request_fields()
        self.assertEqual(fields, {"top_k": -1, "presence_penalty": 0.0})
        self.assertEqual(osworld_runtime.SamplingOverrides().to_request_fields(), {})


if __name__ == "__main__":
    unittest.main()
