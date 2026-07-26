"""Tests for offline_thinking_score.py.

Three layers, none of which loads model weights or touches a GPU:
  1. tokenizer-boundary tests with the REAL Qwen3-VL tokenizer (offline HF
     cache) — span extraction is exact at special-token boundaries;
  2. parity tests against the ACTUAL omegalax training source (imported by
     path): build_chatml_text byte-identity and loss-mask identity with
     collator_qwen3._build_assistant_loss_mask;
  3. end-to-end metric plumbing with deterministic stub models on synthetic
     conversations with real (tiny) images — known NLL, known top1 /
     terminate outcomes, day filtering, report shape, --smoke wiring.

Run:
    cd /fast/project/HFMI_SynergyUnit/yll/juergen
    uv run --no-sync --with pytest python -m pytest eval/test_offline_thinking_score.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import ast
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _purge_import_stubs() -> None:
    """test_freeroll_helpers.py installs fake ``PIL``/``requests`` modules at
    import time when the real ones are absent; under a combined pytest run its
    collection can precede ours and the fakes (``__spec__`` is None) then break
    the real transformers/Pillow imports these tests need. Evict any fake and
    eagerly import the real packages so later stub installs no-op."""
    for name in [m for m in sys.modules
                 if m in ("PIL", "requests") or m.startswith(("PIL.", "requests."))]:
        if getattr(sys.modules[name], "__spec__", None) is None:
            del sys.modules[name]
    import PIL.Image  # noqa: F401, PLC0415
    import requests  # noqa: F401, PLC0415


_purge_import_stubs()

import offline_thinking_score as ots  # noqa: E402

TOKENIZER_ID = "Qwen/Qwen3-VL-2B-Instruct"
OMEGALAX_DATA = Path("/fast/project/HFMI_SynergyUnit/yll/omegalax/omegalax/data")
QWEN3_ENCODING = OMEGALAX_DATA / "qwen3_encoding.py"
COLLATOR = OMEGALAX_DATA / "collator_qwen3.py"


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@lru_cache(maxsize=1)
def _image_processor():
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(TOKENIZER_ID)


def _text(t):
    return {"type": "text", "text": t}


def _image(p):
    return {"type": "image", "image": str(p)}


def _sample_messages():
    """system / user(text) / assistant(think+action) / user(text) / assistant(action)."""
    return [
        {"role": "system", "content": [_text("You operate a desktop computer.")]},
        {"role": "user", "content": [_text("GOAL: install the plugin\nSo far: opened the site")]},
        {"role": "assistant",
         "content": [_text("<think>\nswitch to firefox\n</think>\n-3 4 0")]},
        {"role": "user", "content": [_text("(frame 2)")]},
        {"role": "assistant", "content": [_text("NO_OP")]},
    ]


# ---------------------------------------------------------------------------
# 1. tokenizer-boundary tests
# ---------------------------------------------------------------------------


class SpanBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()
        cls.sp = ots.SpecialIds.from_tokenizer(cls.tok)

    def test_pivot_tokens_are_single_specials(self):
        # SpecialIds.from_tokenizer raises unless each is a single token; also
        # pin the ids so silent tokenizer swaps are caught.
        self.assertEqual(self.sp.im_start, 151644)
        self.assertEqual(self.sp.im_end, 151645)
        self.assertEqual(self.sp.think_open, 151667)
        self.assertEqual(self.sp.think_close, 151668)
        self.assertEqual(self.sp.assistant, 77091)

    def test_terminate_openings(self):
        self.assertGreater(len(self.sp.terminate_ids), 0)
        # "\nTERMINATE" must not merge across the newline: its tail is
        # exactly the plain TERMINATE tokenization.
        self.assertEqual(self.sp.terminate_after_think_ids[1:], self.sp.terminate_ids)

    def test_span_extraction_decodes_exactly(self):
        text = ots.build_chatml_text(_sample_messages(), [], 1)
        ids = self.tok.encode(text, add_special_tokens=False)
        turns = ots.find_assistant_turns(ids, self.sp)
        self.assertEqual(len(turns), 2)

        t0, t1 = turns
        self.assertIsNotNone(t0.thought)
        s, e = t0.thought
        self.assertEqual(self.tok.decode(ids[s:e]), "<think>\nswitch to firefox\n</think>")
        s, e = t0.action
        self.assertEqual(self.tok.decode(ids[s:e]), "\n-3 4 0<|im_end|>")

        self.assertIsNone(t1.thought)
        s, e = t1.action
        self.assertEqual(self.tok.decode(ids[s:e]), "NO_OP<|im_end|>")

    def test_content_starts_after_three_header_tokens(self):
        text = ots.build_chatml_text(_sample_messages(), [], 1)
        ids = self.tok.encode(text, add_special_tokens=False)
        for turn in ots.find_assistant_turns(ids, self.sp):
            start = turn.content_start - 3
            self.assertEqual(ids[start], self.sp.im_start)
            self.assertEqual(ids[start + 1], self.sp.assistant)
            self.assertEqual(self.tok.decode([ids[start + 2]]), "\n")

    def test_action_span_token_count_no_op_turn(self):
        text = ots.build_chatml_text(_sample_messages(), [], 1)
        ids = self.tok.encode(text, add_special_tokens=False)
        turn = ots.find_assistant_turns(ids, self.sp)[1]
        n_expected = len(self.tok.encode("NO_OP", add_special_tokens=False)) + 1  # + <|im_end|>
        self.assertEqual(turn.action[1] - turn.action[0], n_expected)

    def test_unterminated_think_raises(self):
        messages = [
            {"role": "user", "content": [_text("x")]},
            {"role": "assistant", "content": [_text("<think>\nnever closed")]},
        ]
        text = ots.build_chatml_text(messages, [], 1)
        ids = self.tok.encode(text, add_special_tokens=False)
        with self.assertRaises(ValueError):
            ots.find_assistant_turns(ids, self.sp)

    def test_first_divergence(self):
        self.assertEqual(ots.first_divergence((1, 2, 3), (1, 9)), 1)
        self.assertEqual(ots.first_divergence((5,), (1, 2)), 0)
        self.assertIsNone(ots.first_divergence((1, 2), (1, 2, 3)))


class EncodeRecordGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()
        cls.sp = ots.SpecialIds.from_tokenizer(cls.tok)

    def test_terminate_metadata_mismatch_raises(self):
        rec = {
            "conversation_id": "c0",
            "day_tag": "d0",
            "terminate": "clean",
            "messages": [
                {"role": "user", "content": [_text("x")]},
                {"role": "assistant", "content": [_text("NO_OP")]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "terminate"):
            ots.encode_record(rec, self.tok, _image_processor(), self.sp)

    def test_terminate_with_thought_passes_guard(self):
        rec = {
            "conversation_id": "c1",
            "day_tag": "d0",
            "terminate": "verified",
            "messages": [
                {"role": "user", "content": [_text("x")]},
                {"role": "assistant", "content": [_text("<think>\ndone\n</think>\nTERMINATE")]},
            ],
        }
        enc = ots.encode_record(rec, self.tok, _image_processor(), self.sp)
        self.assertEqual([k for k, _ in enc.turns], [ots.TURN_TERMINATE])

    def test_memory_update_kind_on_final_turn(self):
        rec = {
            "conversation_id": "c2",
            "day_tag": "d0",
            "terminate": None,
            "memory_update": True,
            "messages": [
                {"role": "user", "content": [_text("x")]},
                {"role": "assistant", "content": [_text("3 3 0")]},
                {"role": "user", "content": [_text("MEMORY UPDATE: ...")]},
                {"role": "assistant", "content": [_text("Opened the editor.")]},
            ],
        }
        enc = ots.encode_record(rec, self.tok, _image_processor(), self.sp)
        self.assertEqual([k for k, _ in enc.turns], [ots.TURN_ACTION, ots.TURN_MEMORY])


# ---------------------------------------------------------------------------
# 2. parity with the omegalax training source
# ---------------------------------------------------------------------------


@unittest.skipUnless(QWEN3_ENCODING.is_file(), "omegalax checkout not present")
class OmegalaxRenderParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_qenc_parity", QWEN3_ENCODING)
        cls.qenc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.qenc)

    def test_build_chatml_text_byte_identical(self):
        messages = [
            {"role": "system", "content": "plain string content"},
            {"role": "user",
             "content": [_text("GOAL: x"), {"type": "image", "image": "ignored.jpg"},
                         {"type": "image", "image": "ignored2.jpg"}]},
            {"role": "assistant", "content": [_text("<think>\nt\n</think>\n1 2 0")]},
            {"role": "user", "content": [{"type": "image", "image": "ignored3.jpg"}]},
            {"role": "assistant", "content": [_text("TERMINATE")]},
        ]
        grids = [(1, 8, 8), (1, 4, 4), (1, 6, 8)]
        for merge in (1, 2):
            self.assertEqual(
                ots.build_chatml_text(messages, grids, merge),
                self.qenc.build_chatml_text(messages, grids, merge),
            )


@unittest.skipUnless(COLLATOR.is_file(), "omegalax checkout not present")
class OmegalaxLossMaskParityTests(unittest.TestCase):
    """The span logic must imply EXACTLY the training loss mask. The reference
    is the actual _build_assistant_loss_mask source, extracted by ast (the
    module itself imports ml_dtypes, absent from this venv)."""

    @classmethod
    def setUpClass(cls):
        import numpy as np

        tree = ast.parse(COLLATOR.read_text())
        fn = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_build_assistant_loss_mask"
        )
        module = ast.Module(body=[fn], type_ignores=[])
        ns = {"np": np}
        exec(compile(ast.fix_missing_locations(module), str(COLLATOR), "exec"), ns)  # noqa: S102
        cls.reference = staticmethod(ns["_build_assistant_loss_mask"])
        cls.np = np
        cls.tok = _tokenizer()
        cls.sp = ots.SpecialIds.from_tokenizer(cls.tok)

    def _assert_mask_parity(self, messages):
        text = ots.build_chatml_text(messages, [], 1)
        ids = self.tok.encode(text, add_special_tokens=False)
        ref = self.reference(
            self.np.asarray(ids), self.sp.im_start, self.sp.im_end, self.sp.assistant
        )
        mine = ots.loss_mask_from_turns(len(ids), ots.find_assistant_turns(ids, self.sp))
        self.assertEqual(mine, ref.tolist())

    def test_mask_parity_thinking_conversation(self):
        self._assert_mask_parity(_sample_messages())

    def test_mask_parity_terminate_and_memory_turns(self):
        self._assert_mask_parity([
            {"role": "system", "content": [_text("sys")]},
            {"role": "user", "content": [_text("GOAL: g")]},
            {"role": "assistant", "content": [_text("12 0 0 ; +LMB -LMB")]},
            {"role": "user", "content": [_text("MEMORY UPDATE: summarize")]},
            {"role": "assistant", "content": [_text("Did the thing; two steps left.")]},
            {"role": "user", "content": [_text("(frame)")]},
            {"role": "assistant", "content": [_text("<think>\ndone\n</think>\nTERMINATE")]},
        ])

    def test_mask_parity_no_assistant_turn(self):
        self._assert_mask_parity([
            {"role": "system", "content": [_text("sys")]},
            {"role": "user", "content": [_text("only a prompt")]},
        ])


# ---------------------------------------------------------------------------
# 3. stub-model end-to-end plumbing
# ---------------------------------------------------------------------------


def _write_jpeg(path: Path, color):
    from PIL import Image

    Image.new("RGB", (64, 64), color).save(path, format="JPEG")


def _make_records(img_a: Path, img_b: Path):
    """Three synthetic 2-frame conversations across two days:
    A: plain action turns (one with a thought);
    B: terminate window (final turn TERMINATE);
    C: memory-update window (final turn = memory text)."""
    rec_a = {
        "conversation_id": "dayA_g0001_r00_w000",
        "day_tag": "2026-05-01",
        "terminate": None,
        "memory_update": False,
        "messages": [
            {"role": "system", "content": [_text("You operate a desktop computer.")]},
            {"role": "user", "content": [_text("GOAL: write the report"), _image(img_a)]},
            {"role": "assistant",
             "content": [_text("<think>\nopen the editor\n</think>\n-3 4 0")]},
            {"role": "user", "content": [_image(img_b)]},
            {"role": "assistant", "content": [_text("12 0 0 ; +LMB -LMB")]},
        ],
    }
    rec_b = {
        "conversation_id": "dayB_g0002_r00_w000",
        "day_tag": "2026-05-02",
        "terminate": "clean",
        "memory_update": False,
        "messages": [
            {"role": "system", "content": [_text("You operate a desktop computer.")]},
            {"role": "user", "content": [_text("GOAL: send the mail"), _image(img_a)]},
            {"role": "assistant", "content": [_text("NO_OP")]},
            {"role": "user", "content": [_image(img_b)]},
            {"role": "assistant", "content": [_text("TERMINATE")]},
        ],
    }
    rec_c = {
        "conversation_id": "dayB_g0003_r00_w000",
        "day_tag": "2026-05-02",
        "terminate": None,
        "memory_update": True,
        "messages": [
            {"role": "system", "content": [_text("You operate a desktop computer.")]},
            {"role": "user",
             "content": [_text("GOAL: file the ticket\nSo far: form open"), _image(img_a)]},
            {"role": "assistant", "content": [_text("5 5 0")]},
            {"role": "user", "content": [_text("MEMORY UPDATE: Summarize your progress.")]},
            {"role": "assistant",
             "content": [_text("Filled the form; submission still pending.")]},
        ],
    }
    return [rec_a, rec_b, rec_c]


class StubEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()
        cls.imgproc = _image_processor()
        cls.sp = ots.SpecialIds.from_tokenizer(cls.tok)
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        img_a, img_b = base / "a.jpg", base / "b.jpg"
        _write_jpeg(img_a, (200, 30, 30))
        _write_jpeg(img_b, (30, 30, 200))
        cls.records = _make_records(img_a, img_b)
        cls.vocab = len(cls.tok)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _score(self, model, records=None, batch_size=2):
        return ots.score_records(
            records if records is not None else self.records,
            model, self.tok, self.imgproc, device="cpu", batch_size=batch_size,
        )

    def test_oracle_stub_known_metrics(self):
        # peak=8: the closed-form NLL (~3.9) is large enough that fp32
        # log-softmax accumulation over the 152k vocab is exact to ~1e-6;
        # at the default peak=20 the true NLL (~3e-4) sits below fp32
        # summation accuracy and only order-of-magnitude checks make sense.
        rows = self._score(ots.make_stub_model(self.vocab, "oracle", peak=8.0))
        agg = ots.aggregate(rows)
        m = agg["aggregate"]

        expected = ots.oracle_expected_nll(self.vocab, peak=8.0)
        for name in ("action_nll", "thought_nll", "memory_nll", "terminate_nll"):
            self.assertTrue(
                math.isclose(m[name]["mean"], expected, rel_tol=1e-3),
                f"{name}: {m[name]['mean']} != {expected}",
            )
        self.assertEqual(m["action_top1"]["rate"], 1.0)
        self.assertEqual(m["action_top1"]["n_turns"], 4)      # A1 A2 B1 C1
        self.assertEqual(m["terminate_recall"]["rate"], 1.0)
        self.assertEqual(m["terminate_recall"]["n_turns"], 1)
        self.assertEqual(m["terminate_false_alarm"]["rate"], 0.0)
        self.assertEqual(m["terminate_false_alarm"]["n_turns"], 4)
        self.assertEqual(m["memory_nll"]["n_turns"], 1)
        self.assertEqual(m["thought_nll"]["n_turns"], 1)
        self.assertEqual(m["n_turns_total"], 6)

    def test_oracle_token_counts_are_exact(self):
        rows = self._score(ots.make_stub_model(self.vocab, "oracle"))
        by_conv = {}
        for r in rows:
            by_conv.setdefault(r["conversation_id"], []).append(r)

        think_ids = self.tok.encode(
            "<think>\nopen the editor\n</think>", add_special_tokens=False)
        a1 = by_conv["dayA_g0001_r00_w000"][0]
        self.assertEqual(a1["n_thought_tokens"], len(think_ids))
        act_ids = self.tok.encode("\n-3 4 0", add_special_tokens=False)
        self.assertEqual(a1["n_action_tokens"], len(act_ids) + 1)  # + <|im_end|>

        b_term = by_conv["dayB_g0002_r00_w000"][1]
        self.assertEqual(b_term["kind"], ots.TURN_TERMINATE)
        term_ids = self.tok.encode("TERMINATE", add_special_tokens=False)
        self.assertEqual(b_term["n_action_tokens"], len(term_ids) + 1)

    def test_uniform_stub_nll_is_log_vocab(self):
        rows = self._score(ots.make_stub_model(self.vocab, "uniform"))
        m = ots.aggregate(rows)["aggregate"]
        for name in ("action_nll", "thought_nll", "memory_nll"):
            self.assertTrue(math.isclose(m[name]["mean"], math.log(self.vocab), rel_tol=1e-4))
        self.assertEqual(m["action_top1"]["rate"], 0.0)
        self.assertEqual(m["terminate_recall"]["rate"], 0.0)
        self.assertEqual(m["terminate_false_alarm"]["rate"], 0.0)

    def test_terminate_biased_stub_false_alarms(self):
        const = self.sp.terminate_ids[0]
        rows = self._score(ots.make_stub_model(self.vocab, "const", const_token_id=const))
        m = ots.aggregate(rows)["aggregate"]
        # Every non-terminate action turn greedily opens with the TERMINATE
        # token at its divergence position (thought turns diverge after the
        # shared "\n": position 1 of "\nTERMINATE").
        self.assertEqual(m["terminate_false_alarm"]["rate"], 1.0)
        self.assertEqual(m["terminate_false_alarm"]["n_turns"], 4)
        # But it cannot reproduce the full multi-token TERMINATE span.
        self.assertEqual(m["terminate_recall"]["rate"], 0.0)
        self.assertEqual(m["action_top1"]["rate"], 0.0)

    def test_batch_size_invariance(self):
        model = ots.make_stub_model(self.vocab, "oracle")
        rows_b1 = self._score(model, batch_size=1)
        rows_b3 = self._score(model, batch_size=3)
        self.assertEqual(len(rows_b1), len(rows_b3))
        for r1, r3 in zip(rows_b1, rows_b3):
            self.assertEqual(r1["conversation_id"], r3["conversation_id"])
            self.assertEqual(r1["exact"], r3["exact"])
            self.assertAlmostEqual(r1["action_nll_sum"], r3["action_nll_sum"], places=4)

    def test_per_day_grouping(self):
        rows = self._score(ots.make_stub_model(self.vocab, "oracle"))
        agg = ots.aggregate(rows)
        self.assertEqual(set(agg["per_day"]), {"2026-05-01", "2026-05-02"})
        self.assertEqual(agg["per_day"]["2026-05-01"]["n_turns_total"], 2)
        self.assertEqual(agg["per_day"]["2026-05-02"]["n_turns_total"], 4)
        self.assertEqual(agg["per_day"]["2026-05-02"]["terminate_recall"]["n_turns"], 1)


class CliAndReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        img_a, img_b = base / "a.jpg", base / "b.jpg"
        _write_jpeg(img_a, (200, 30, 30))
        _write_jpeg(img_b, (30, 30, 200))
        cls.data_dir = base / "stage04_out"
        cls.data_dir.mkdir()
        with (cls.data_dir / "conversations.jsonl").open("w") as fh:
            for rec in _make_records(img_a, img_b):
                fh.write(json.dumps(rec) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parse_days_comma_list(self):
        self.assertEqual(ots.parse_days("d1, d2,d3"), {"d1", "d2", "d3"})
        self.assertIsNone(ots.parse_days(None))
        with self.assertRaises(ValueError):
            ots.parse_days(" , ")

    def test_parse_days_file(self):
        p = Path(self.tmp.name) / "days.txt"
        p.write_text("# held-out\n2026-05-02\n\n2026-05-03\n")
        self.assertEqual(ots.parse_days(str(p)), {"2026-05-02", "2026-05-03"})

    def test_load_records_day_filter_and_limit(self):
        recs = ots.load_records(self.data_dir, {"2026-05-02"}, None)
        self.assertEqual([r["day_tag"] for r in recs], ["2026-05-02", "2026-05-02"])
        recs = ots.load_records(self.data_dir, None, 1)
        self.assertEqual(len(recs), 1)
        # dir and direct-file addressing are equivalent
        recs2 = ots.load_records(self.data_dir / "conversations.jsonl", None, 1)
        self.assertEqual(recs, recs2)

    def test_smoke_main_end_to_end(self):
        out = Path(self.tmp.name) / "report.json"
        report = ots.main([
            "--checkpoint", TOKENIZER_ID,
            "--conversations", str(self.data_dir),
            "--smoke",
            "--output", str(out),
        ])
        self.assertTrue(out.is_file())
        on_disk = json.loads(out.read_text())
        self.assertEqual(on_disk["task"], "offline_thinking_score")
        self.assertTrue(on_disk["smoke"])
        self.assertEqual(on_disk["n_records"], 3)
        self.assertEqual(set(on_disk["per_day"]), {"2026-05-01", "2026-05-02"})
        # stub is the oracle: perfect imitation
        self.assertEqual(on_disk["scores"]["offline_thinking/action_top1"], 1.0)
        self.assertEqual(on_disk["scores"]["offline_thinking/terminate_recall"], 1.0)
        self.assertEqual(on_disk["scores"]["offline_thinking/terminate_false_alarm"], 0.0)
        self.assertLess(on_disk["scores"]["offline_thinking/action_nll"], 0.01)
        for key in ("aggregate", "per_day", "n_records_per_day", "params", "elapsed_s"):
            self.assertIn(key, report)

    def test_smoke_main_day_filter(self):
        out = Path(self.tmp.name) / "report_day.json"
        report = ots.main([
            "--checkpoint", TOKENIZER_ID,
            "--conversations", str(self.data_dir),
            "--days", "2026-05-01",
            "--smoke",
            "--output", str(out),
        ])
        self.assertEqual(report["n_records"], 1)
        self.assertEqual(set(report["per_day"]), {"2026-05-01"})


if __name__ == "__main__":
    unittest.main()
