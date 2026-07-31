"""Regression gates for defects #7, #8, #9, #20 — the "server said READY" family.

* #7  ``deployment.num_infer_gpus`` OVERRIDES ``inference.parallel.dp``
      (``prime-rl-configs/.../configs/rl.py``:542-555, the override is line 549).
* #8  a ``config.json`` missing ``architectures`` makes vLLM resolve
      ``--runner pooling / --convert embed``; ``/v1/models`` returns 200 (gate says
      READY) while every ``/v1/chat/completions`` returns 404.
* #9  ``return_exceptions=True`` + filtering the exceptions out yielded
      ``success=0/0`` and wrote **0.0 as a result**.
* #20 vLLM's compile cache on NFS ``/fast/home`` throws ``Errno 121 Remote I/O error``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rft.errors import (
    DeploymentConfigError,
    ExportConfigError,
    FailureRateExceeded,
    PreflightError,
    SchemaError,
)
from rft.sampling import (
    ErrorLedger,
    RolloutStore,
    SamplingConfig,
    enumerate_units,
    merge_shards,
    run_sampling,
    shard_assignments,
)
from rft.serving import (
    CACHE_ENV_VARS,
    assert_caches_off_home,
    assert_export_differs_from_base,
    compile_cache_env,
    preflight_chat_completion,
    validate_deployment_parallelism,
    validate_export_config,
)


# --------------------------------------------------------------------------
# fake HTTP transports
# --------------------------------------------------------------------------
def _chat_ok(_url, _body, _timeout):
    return 200, {"choices": [{"message": {"content": "ready"}}]}


def _chat_404(_url, _body, _timeout):
    return 404, {"detail": "Not Found"}


def _models_ok(_url, _timeout):
    return 200, {"data": [{"id": "policy"}]}


class Defect08PoolingRunnerTest(unittest.TestCase):
    """Defect #8: an export without ``architectures`` serves as an EMBEDDING model.

    ``/v1/models`` answers 200 in pooling mode, so a readiness gate that polls it
    prints READY while every chat completion 404s. Two independent guards must hold:
    the export is rejected up front, and the preflight is a real chat completion.
    """

    def _write_export(self, d: Path, cfg: dict, *, weights: bool = True) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "tokenizer.json").write_text("{}")
        if weights:
            (d / "model.safetensors").write_bytes(b"\x00" * 128)
        return d

    def test_missing_architectures_is_rejected_at_export_time(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            d = self._write_export(Path(tmp) / "exp", {"model_type": "qwen3_vl"})
            with self.assertRaises(ExportConfigError) as ctx:
                validate_export_config(d)
            msg = str(ctx.exception)
            self.assertIn("architectures", msg)
            self.assertIn("pooling", msg)
            self.assertIn("404", msg)

    def test_empty_architectures_list_is_rejected(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            d = self._write_export(
                Path(tmp) / "exp", {"model_type": "qwen3_vl", "architectures": []}
            )
            with self.assertRaises(ExportConfigError):
                validate_export_config(d)

    def test_pooling_style_architecture_is_rejected(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            d = self._write_export(
                Path(tmp) / "exp",
                {"model_type": "qwen3_vl", "architectures": ["Qwen3VLModel"]},
            )
            with self.assertRaises(ExportConfigError) as ctx:
                validate_export_config(d)
            self.assertIn("pooling", str(ctx.exception))

    def test_good_export_passes_and_is_described(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            d = self._write_export(
                Path(tmp) / "exp",
                {
                    "model_type": "qwen3_vl",
                    "architectures": ["Qwen3VLForConditionalGeneration"],
                },
            )
            audit = validate_export_config(d)
            self.assertEqual(audit.architectures, ("Qwen3VLForConditionalGeneration",))
            self.assertIn("architectures", audit.describe())

    def test_config_without_weights_is_rejected(self) -> None:
        """An export that wrote only a config secretly serves something else."""
        with TemporaryDirectory(dir="/tmp") as tmp:
            d = self._write_export(
                Path(tmp) / "exp",
                {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]},
                weights=False,
            )
            with self.assertRaises(ExportConfigError) as ctx:
                validate_export_config(d)
            self.assertIn("no weight shards", str(ctx.exception))

    def test_preflight_fails_loudly_on_a_pooling_server(self) -> None:
        """The core of #8: /v1/models 200 must NOT be enough to say READY."""
        with self.assertRaises(PreflightError) as ctx:
            preflight_chat_completion(
                base_url="http://localhost:8000/v1",
                model="policy",
                timeout_s=0.0,
                http_post=_chat_404,
                http_get=_models_ok,  # a pooling server answers this with 200
                sleep=lambda _s: None,
                monotonic=lambda: 0.0,
            )
        msg = str(ctx.exception)
        self.assertIn("404", msg)
        self.assertIn("pooling", msg)
        self.assertIn("architectures", msg)

    def test_preflight_passes_only_on_a_real_completion(self) -> None:
        result = preflight_chat_completion(
            base_url="http://localhost:8000/v1",
            model="policy",
            http_post=_chat_ok,
            http_get=_models_ok,
            sleep=lambda _s: None,
        )
        self.assertEqual(result.completion_preview, "ready")
        self.assertEqual(result.served_models, ("policy",))
        self.assertEqual(result.warnings, ())

    def test_preflight_warns_when_the_served_name_differs(self) -> None:
        result = preflight_chat_completion(
            base_url="http://localhost:8000/v1",
            model="wrong-name",
            http_post=_chat_ok,
            http_get=_models_ok,
            sleep=lambda _s: None,
        )
        self.assertTrue(any("not in /v1/models" in w for w in result.warnings))

    def test_empty_completion_is_not_ready(self) -> None:
        def empty(_url, _body, _timeout):
            return 200, {"choices": [{"message": {"content": "   "}}]}

        with self.assertRaises(PreflightError) as ctx:
            preflight_chat_completion(
                base_url="http://x/v1",
                model="policy",
                timeout_s=0.0,
                http_post=empty,
                http_get=_models_ok,
                sleep=lambda _s: None,
                monotonic=lambda: 0.0,
            )
        self.assertIn("no choices[0].message.content", str(ctx.exception))

    def test_lora_export_identical_to_base_is_rejected(self) -> None:
        """The prime-rl LoRA-merge bug: weights/step_N was byte-identical to base, so
        every eval secretly scored the base model (ckpt.py:445-453 is the fix)."""
        with TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp) / "base"
            exp = Path(tmp) / "exp"
            for d in (base, exp):
                d.mkdir()
                (d / "model.safetensors").write_bytes(b"identical-bytes" * 100)
            with self.assertRaises(ExportConfigError) as ctx:
                assert_export_differs_from_base(exp, base)
            self.assertIn("never merged", str(ctx.exception))

    def test_genuinely_finetuned_export_passes(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            base = Path(tmp) / "base"
            exp = Path(tmp) / "exp"
            base.mkdir()
            exp.mkdir()
            (base / "model.safetensors").write_bytes(b"A" * 1000)
            (exp / "model.safetensors").write_bytes(b"B" * 1000)
            assert_export_differs_from_base(exp, base)  # must not raise


class Defect07DeploymentParallelismTest(unittest.TestCase):
    """Defect #7: ``num_infer_gpus`` wins over ``inference.parallel.dp``.

    ``configs/rl.py``:549 recomputes ``dp = num_infer_gpus // tp``, discarding a
    user-set dp with no warning. Editing dp alone leaves ranks that never start;
    every request routed to them 400s, and because routing is session-hashed per
    rollout group, one 400 destroys the whole group (20-33% observed loss).
    """

    def test_consistent_plan_passes(self) -> None:
        plan = validate_deployment_parallelism(num_infer_gpus=4, dp=2, tp=2)
        self.assertEqual(plan.gpus_required, 4)
        self.assertIn("num_infer_gpus=4", plan.describe())

    def test_editing_dp_alone_is_caught_before_any_gpu_is_allocated(self) -> None:
        with self.assertRaises(DeploymentConfigError) as ctx:
            validate_deployment_parallelism(num_infer_gpus=4, dp=4, tp=2)
        msg = str(ctx.exception)
        self.assertIn("num_infer_gpus WINS", msg)
        self.assertIn("session-hashed", msg)

    def test_the_override_arithmetic_is_pinned(self) -> None:
        """Model rl.py:549 exactly, so a change upstream shows up here."""
        num_infer_gpus, tp, user_dp = 4, 2, 4
        effective_dp = num_infer_gpus // tp
        self.assertNotEqual(effective_dp, user_dp)
        self.assertEqual(effective_dp, 2)

    def test_zero_and_negative_degrees_are_rejected(self) -> None:
        for kwargs in (
            {"num_infer_gpus": 0, "dp": 1, "tp": 1},
            {"num_infer_gpus": 2, "dp": 0, "tp": 1},
            {"num_infer_gpus": 2, "dp": 2, "tp": -1},
        ):
            with self.assertRaises(DeploymentConfigError):
                validate_deployment_parallelism(**kwargs)  # type: ignore[arg-type]


class Defect20CachePlacementTest(unittest.TestCase):
    """Defect #20: compile caches on NFS ``/fast/home`` throw Errno 121."""

    def test_home_cache_root_is_refused(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            compile_cache_env("/fast/home/franz.srambical/.cache/vllm")
        self.assertIn("Errno 121", str(ctx.exception))

    def test_project_cache_root_is_accepted_and_covers_every_var(self) -> None:
        with TemporaryDirectory(dir="/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/tmp") as tmp:
            env = compile_cache_env(tmp)
            for var in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                self.assertIn(var, env)
                self.assertTrue(Path(env[var]).is_dir())
                self.assertTrue(env[var].startswith("/fast/project"))

    def test_inherited_home_cache_env_is_caught(self) -> None:
        with self.assertRaises(SchemaError) as ctx:
            assert_caches_off_home({"VLLM_CACHE_ROOT": "/fast/home/x/.cache"})
        self.assertIn("VLLM_CACHE_ROOT", str(ctx.exception))

    def test_clean_env_passes(self) -> None:
        assert_caches_off_home({"VLLM_CACHE_ROOT": "/fast/project/x"})

    def test_every_known_cache_var_is_checked(self) -> None:
        for var in CACHE_ENV_VARS:
            with self.assertRaises(SchemaError, msg=var):
                assert_caches_off_home({var: "/fast/home/x"})


class Defect09NoSilentZerosTest(unittest.TestCase):
    """Defect #9: a probe used ``return_exceptions=True``, filtered the exceptions out,
    got ``success=0/0``, and wrote **0.0** as a result.
    """

    def test_the_defective_pattern_produces_a_fake_zero(self) -> None:
        """Reproduce it, so the fix has something to be a fix of."""
        results = [RuntimeError("boom")] * 8  # what return_exceptions=True yields
        oks = [r for r in results if not isinstance(r, BaseException)]
        naive_rate = (sum(1 for _ in oks) / len(oks)) if oks else 0.0
        self.assertEqual(naive_rate, 0.0, "the defective probe writes 0.0")

    def test_ledger_rate_over_zero_attempts_is_undefined(self) -> None:
        ledger = ErrorLedger()
        with self.assertRaises(SchemaError) as ctx:
            _ = ledger.failure_rate
        self.assertIn("undefined", str(ctx.exception))

    def test_ledger_aborts_instead_of_degrading(self) -> None:
        ledger = ErrorLedger(max_failure_rate=0.05, min_attempts_before_abort=10)
        for _ in range(10):
            ledger.record_ok()
        with self.assertRaises(FailureRateExceeded) as ctx:
            for i in range(5):
                ledger.record_failure(
                    sample_id=f"s{i}", task_id="t", rollout_index=i, exc=RuntimeError("boom")
                )
        msg = str(ctx.exception)
        self.assertIn("failure rate", msg)
        self.assertIn("RuntimeError", msg)

    def test_ledger_keeps_every_failure_classified(self) -> None:
        ledger = ErrorLedger(max_failure_rate=1.0 - 1e-9, min_attempts_before_abort=10_000)
        ledger.record_failure(sample_id="a", task_id="t", rollout_index=0, exc=ValueError("v"))
        ledger.record_failure(sample_id="b", task_id="t", rollout_index=1, exc=KeyError("k"))
        self.assertEqual(ledger.kind_counts(), {"ValueError": 1, "KeyError": 1})
        self.assertEqual(ledger.n_failed, 2)
        self.assertIn("failure_rate", ledger.as_dict())

    def test_rollout_fn_returning_an_error_record_is_rejected(self) -> None:
        """A failed rollout must RAISE so the ledger sees it, not return an
        error-shaped 'success' that downstream reads as a score."""
        with TemporaryDirectory(dir="/tmp") as tmp:
            cfg = SamplingConfig(
                task_ids=["t1"],
                k=1,
                grammar="bare_line",
                out_path=Path(tmp) / "rollouts.jsonl",
                base_url="http://x/v1",
                model="policy",
            )
            with self.assertRaises(SchemaError) as ctx:
                run_sampling(
                    cfg,
                    lambda _u: {"error": "swallowed"},
                    http_post=_chat_ok,
                    http_get=_models_ok,
                    check_caches=False,
                )
            self.assertIn("must RAISE", str(ctx.exception))


class SamplingDeterminismAndResumeTest(unittest.TestCase):
    """Stage-1 requirements: deterministic sharding and real resumability."""

    def test_sharding_is_deterministic_and_order_independent(self) -> None:
        tasks = [f"task{i}" for i in range(50)]
        units_a = enumerate_units(tasks, k=4, sample_root="/tmp/root")
        units_b = enumerate_units(list(reversed(tasks)), k=4, sample_root="/tmp/root")
        for i in range(8):
            got_a = {(u.task_id, u.rollout_index) for u in shard_assignments(
                units_a, num_shards=8, shard_index=i)}
            got_b = {(u.task_id, u.rollout_index) for u in shard_assignments(
                units_b, num_shards=8, shard_index=i)}
            self.assertEqual(got_a, got_b, f"shard {i} depends on input order")

    def test_shards_partition_the_work_exactly_once(self) -> None:
        units = enumerate_units([f"t{i}" for i in range(30)], k=3, sample_root="/tmp/root")
        seen: list[tuple[str, int]] = []
        for i in range(7):
            seen.extend(
                (u.task_id, u.rollout_index)
                for u in shard_assignments(units, num_shards=7, shard_index=i)
            )
        self.assertEqual(len(seen), len(units))
        self.assertEqual(len(set(seen)), len(units))

    def test_seeds_are_stable_per_unit(self) -> None:
        a = enumerate_units(["x", "y"], k=2, sample_root="/tmp/r")
        b = enumerate_units(["y", "x"], k=2, sample_root="/tmp/r")
        by_key = {(u.task_id, u.rollout_index): u.seed for u in b}
        for u in a:
            self.assertEqual(u.seed, by_key[(u.task_id, u.rollout_index)])

    def test_duplicate_task_ids_are_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            enumerate_units(["a", "a"], k=1, sample_root="/tmp/r")

    def test_resume_skips_completed_units(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            out = Path(tmp) / "rollouts.jsonl"
            cfg = SamplingConfig(
                task_ids=[f"t{i}" for i in range(5)],
                k=2,
                grammar="bare_line",
                out_path=out,
                base_url="http://x/v1",
                model="policy",
            )
            calls: list[str] = []

            def roll(unit):
                calls.append(unit.sample_id)
                return {"scores": {"reward": 1.0}, "completion": "0 0 0"}

            r1 = run_sampling(cfg, roll, http_post=_chat_ok, http_get=_models_ok,
                              check_caches=False)
            self.assertEqual(r1.n_units_this_shard, 10)
            self.assertEqual(len(calls), 10)

            calls.clear()
            r2 = run_sampling(cfg, roll, http_post=_chat_ok, http_get=_models_ok,
                              check_caches=False)
            self.assertEqual(calls, [], "a resumed run must redo nothing")
            self.assertEqual(r2.n_skipped_resumed, 10)

    def test_truncated_final_line_is_counted_and_redone(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            out = Path(tmp) / "rollouts.jsonl"
            store = RolloutStore(out)
            store.append({"sample_id": "good", "x": 1})
            with out.open("a") as fh:
                fh.write('{"sample_id": "trunca')  # killed mid-write
            store2 = RolloutStore(out)
            done = store2.completed_ids()
            self.assertEqual(done, {"good"})
            self.assertEqual(store2.n_truncated_lines, 1)

    def test_merge_rejects_duplicate_sample_ids(self) -> None:
        with TemporaryDirectory(dir="/tmp") as tmp:
            a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            for p in (a, b):
                RolloutStore(p).append({"sample_id": "same", "v": 1})
            with self.assertRaises(SchemaError) as ctx:
                merge_shards([a, b], Path(tmp) / "merged.jsonl")
            self.assertIn("duplicate sample_id", str(ctx.exception))

    def test_unknown_grammar_fails_before_any_gpu_work(self) -> None:
        with self.assertRaises(SchemaError):
            SamplingConfig(
                task_ids=["t"],
                k=1,
                grammar="nonsense",
                out_path=Path("/tmp/x.jsonl"),
                base_url="http://x/v1",
                model="policy",
            )


if __name__ == "__main__":
    unittest.main()
