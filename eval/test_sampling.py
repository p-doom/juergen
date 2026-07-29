"""Unit tests for the Qwen-recommended sampling source of truth.

Pure-stdlib (dataclasses/argparse only), so it runs without the heavy eval venv:

    python -m unittest eval.test_sampling      # from the repo root
    python test_sampling.py                    # from eval/
"""

from __future__ import annotations

import argparse
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sampling  # noqa: E402


class TestRecommendedTuples(unittest.TestCase):
    def test_instruct_tuple(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT)
        self.assertEqual(sp.mode, "instruct")
        self.assertAlmostEqual(sp.temperature, 0.7)
        self.assertAlmostEqual(sp.top_p, 0.8)
        self.assertEqual(sp.top_k, 20)
        self.assertAlmostEqual(sp.repetition_penalty, 1.0)
        self.assertAlmostEqual(sp.presence_penalty, 1.5)
        self.assertFalse(sp.greedy)

    def test_thinking_tuple(self):
        sp = sampling.qwen_sampling(sampling.THINKING)
        self.assertEqual(sp.mode, "thinking")
        self.assertAlmostEqual(sp.temperature, 1.0)
        self.assertAlmostEqual(sp.top_p, 0.95)
        self.assertEqual(sp.top_k, 20)
        self.assertAlmostEqual(sp.repetition_penalty, 1.0)
        self.assertAlmostEqual(sp.presence_penalty, 0.0)

    def test_instruct_and_thinking_differ(self):
        i = sampling.qwen_sampling(sampling.INSTRUCT)
        t = sampling.qwen_sampling(sampling.THINKING)
        self.assertNotEqual((i.temperature, i.top_p), (t.temperature, t.top_p))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            sampling.qwen_sampling("chatty")


class TestModeDetection(unittest.TestCase):
    def test_explicit_wins(self):
        self.assertEqual(
            sampling.detect_mode(model_path="Qwen3-VL-8B-Thinking", mode="instruct"),
            "instruct")

    def test_thinking_from_checkpoint_name(self):
        self.assertEqual(
            sampling.detect_mode(model_path="/ckpt/qwen3vl-8b-thinking-16k"),
            "thinking")

    def test_thinking_from_think_tag_in_prompt(self):
        self.assertEqual(
            sampling.detect_mode(system_prompt="Reason step by step.\n<think>"),
            "thinking")

    def test_defaults_to_instruct(self):
        self.assertEqual(
            sampling.detect_mode(model_path="Qwen/Qwen3-VL-8B-Instruct"),
            "instruct")

    def test_think_carefully_prompt_not_misflagged(self):
        # A prose "think" in an Instruct prompt must NOT flip the regime.
        self.assertEqual(
            sampling.detect_mode(
                model_path="Qwen/Qwen3-VL-8B-Instruct",
                system_prompt="Please think carefully before acting."),
            "instruct")


class TestOverrides(unittest.TestCase):
    def test_presence_penalty_override_to_zero(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT, presence_penalty=0.0)
        self.assertAlmostEqual(sp.presence_penalty, 0.0)
        # everything else stays on the Qwen tuple
        self.assertAlmostEqual(sp.temperature, 0.7)
        self.assertAlmostEqual(sp.top_p, 0.8)

    def test_max_tokens_default_and_override(self):
        self.assertEqual(sampling.qwen_sampling(sampling.INSTRUCT).max_tokens,
                         sampling.DEFAULT_MAX_TOKENS)
        self.assertEqual(
            sampling.qwen_sampling(sampling.INSTRUCT, max_tokens=256).max_tokens, 256)


class TestWireFormats(unittest.TestCase):
    def test_request_json_is_full_tuple(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT)
        body = sp.as_request_json()
        self.assertEqual(
            set(body),
            {"max_tokens", "temperature", "top_p", "top_k",
             "repetition_penalty", "presence_penalty"})
        self.assertEqual(body["top_k"], 20)
        self.assertAlmostEqual(body["presence_penalty"], 1.5)

    def test_openai_kwargs_route_via_extra_body(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT)
        kw = sp.as_openai_kwargs()
        # top-level = OpenAI-schema-legal only
        self.assertEqual(set(kw), {"max_tokens", "temperature", "top_p", "extra_body"})
        self.assertEqual(
            set(kw["extra_body"]), {"top_k", "repetition_penalty", "presence_penalty"})
        self.assertEqual(kw["extra_body"]["top_k"], 20)

    def test_greedy_drops_sampling_knobs(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT, greedy=True)
        self.assertEqual(sp.as_request_json(), {"max_tokens": sp.max_tokens, "temperature": 0.0})
        self.assertEqual(sp.as_openai_kwargs(), {"max_tokens": sp.max_tokens, "temperature": 0.0})


class _UnpatchedAgent:
    """Mimics the stock vendored Qwen3VLAgent constructor signature."""
    def __init__(self, platform="ubuntu", model="m", max_tokens=32768,
                 top_p=0.9, temperature=0.0, action_space="pyautogui",
                 api_backend="openai"):
        pass


class _PatchedAgent:
    """Mimics the checkout-patched constructor (accepts the extra knobs)."""
    def __init__(self, platform="ubuntu", model="m", max_tokens=32768,
                 top_p=0.9, temperature=0.0, top_k=None,
                 repetition_penalty=None, presence_penalty=None,
                 action_space="pyautogui", api_backend="openai"):
        pass


class TestOpenAIAgentKwargs(unittest.TestCase):
    def test_patched_agent_gets_full_tuple(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT, max_tokens=2048)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning expected
            kw = sampling.openai_agent_kwargs(
                _PatchedAgent, sp, base={"model": "x", "api_backend": "openai"})
        self.assertEqual(kw["top_k"], 20)
        self.assertAlmostEqual(kw["presence_penalty"], 1.5)
        self.assertAlmostEqual(kw["temperature"], 0.7)
        self.assertEqual(kw["max_tokens"], 2048)
        _PatchedAgent(**kw)  # must not raise

    def test_unpatched_agent_drops_extras_and_warns(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT)
        with self.assertWarns(UserWarning):
            kw = sampling.openai_agent_kwargs(
                _UnpatchedAgent, sp, base={"model": "x", "api_backend": "openai"})
        # extras dropped so the stock constructor does not TypeError
        for k in ("top_k", "repetition_penalty", "presence_penalty"):
            self.assertNotIn(k, kw)
        self.assertAlmostEqual(kw["temperature"], 0.7)
        _UnpatchedAgent(**kw)  # must not raise

    def test_greedy_unpatched_does_not_warn(self):
        sp = sampling.qwen_sampling(sampling.INSTRUCT, greedy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kw = sampling.openai_agent_kwargs(
                _UnpatchedAgent, sp, base={"model": "x"})
        self.assertEqual(kw["temperature"], 0.0)


class TestCli(unittest.TestCase):
    def _parse(self, argv):
        p = argparse.ArgumentParser()
        sampling.add_sampling_cli(p, default_max_tokens=256)
        return p.parse_args(argv)

    def test_defaults_are_qwen_not_greedy(self):
        args = self._parse([])
        sp = sampling.from_cli(args, model_path="Qwen/Qwen3-VL-8B-Instruct")
        self.assertFalse(sp.greedy)
        self.assertAlmostEqual(sp.temperature, 0.7)  # NOT 0.0
        self.assertEqual(sp.max_tokens, 256)

    def test_auto_detects_thinking(self):
        args = self._parse([])
        sp = sampling.from_cli(args, model_path="/ckpt/qwen3vl-thinking")
        self.assertEqual(sp.mode, "thinking")
        self.assertAlmostEqual(sp.temperature, 1.0)

    def test_flags_take_effect(self):
        args = self._parse(["--temperature", "0.3", "--presence_penalty", "0", "--top_k", "40"])
        sp = sampling.from_cli(args, model_path="Qwen/Qwen3-VL-8B-Instruct")
        self.assertAlmostEqual(sp.temperature, 0.3)
        self.assertAlmostEqual(sp.presence_penalty, 0.0)
        self.assertEqual(sp.top_k, 40)

    def test_greedy_flag(self):
        args = self._parse(["--greedy"])
        sp = sampling.from_cli(args, model_path="Qwen/Qwen3-VL-8B-Instruct")
        self.assertTrue(sp.greedy)


if __name__ == "__main__":
    unittest.main()
