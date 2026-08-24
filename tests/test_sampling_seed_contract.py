"""Paired sampling uses one attested seed path or the run is not causal."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_seed_identity_is_arm_independent_and_distinct_across_trials() -> None:
    from evals.signoflife.__main__ import _sampling_seed

    identity = {
        "suite_manifest_sha256": "a" * 64,
        "cell_id": "terminal_submit_only",
    }
    first = _sampling_seed(**identity, trial=1)
    assert first == _sampling_seed(**identity, trial=1)
    assert 0 < first <= 0x7FFFFFFF
    assert len({_sampling_seed(**identity, trial=trial) for trial in range(1, 5)}) == 4


def test_eval_seed_reaches_the_wire_record_from_ctx_sampling(tmp_path) -> None:
    import verifiers.v1 as vf

    from agent.agent import resolve_sampling
    from evals.signoflife.__main__ import _eval_config

    config = _eval_config(
        arm="ordered",
        tier="candidate",
        task_ids=["terminal_submit_only"],
        artifacts=tmp_path / "artifacts",
        traces_dir=tmp_path / "traces",
        pool={},
        base_url="http://127.0.0.1:1/v1",
        temperature=0.7,
        top_p=1.0,
        max_tokens=256,
        served_model="sign-of-life-sha256-test",
        seed=1234567,
    )
    ctx = vf.ModelContext(model=config.model, client=None, sampling=config.sampling)
    wire, effective = resolve_sampling(ctx, {"messages": []})

    assert wire["seed"] == 1234567
    assert effective.as_dict()["seed"] == 1234567
    assert "seed" in effective.wire_body_keys


def test_local_launch_enables_deterministic_sampling_without_an_api_key(
    monkeypatch,
) -> None:
    from evals.signoflife.__main__ import _sglang_command, _sglang_environment

    monkeypatch.setenv("SIGN_OF_LIFE_API_KEY", "never-in-the-server-child")

    command = _sglang_command(
        python="/sealed/venv/bin/python",
        model_path="/sealed/model",
        port=19000,
        mem_fraction_static=0.65,
        served_model="sign-of-life-sha256-test",
    )

    assert "--enable-deterministic-inference" in command
    assert "--api-key" not in command
    assert not any("secret" in value for value in command)
    assert "SIGN_OF_LIFE_API_KEY" not in _sglang_environment()
    assert "never-in-the-server-child" not in json.dumps(_sglang_environment())


def test_local_launch_refuses_unbound_semantic_environment(monkeypatch) -> None:
    from evals.signoflife.__main__ import _sglang_environment

    monkeypatch.setenv("TORCH_LOGS", "recompiles")
    with pytest.raises(RuntimeError, match="TORCH_LOGS"):
        _sglang_environment()


def test_local_child_process_and_log_never_receive_parent_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    from evals.signoflife.__main__ import _sglang_environment

    secret = "credential-that-must-not-cross-the-launch-boundary"
    monkeypatch.setenv("SIGN_OF_LIFE_API_KEY", secret)
    command = [
        sys.executable,
        "-c",
        "import os,time; print(dict(os.environ), flush=True); time.sleep(30)",
    ]
    log_path = tmp_path / "child.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=_sglang_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            process_cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
            process_environment = Path(f"/proc/{process.pid}/environ").read_bytes()
        finally:
            process.terminate()
            process.wait(timeout=10)

    observed = process_cmdline + process_environment + log_path.read_bytes()
    assert secret.encode() not in observed
    assert b"SIGN_OF_LIFE_API_KEY" not in observed


class _Response:
    status = 200

    def __init__(self, value) -> None:
        self._value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, limit):
        assert limit == 65537
        return json.dumps(self._value).encode()


def test_local_server_identity_requires_seeded_pytorch_mode(tmp_path, monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher
    from test_model_attestation import _register_model

    model, _ = _register_model(tmp_path)
    artifact = dispatcher._verify_model_artifact(model)

    def urlopen(request, timeout):
        assert timeout == 10.0
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/v1/models"):
            return _Response({"data": [{"id": artifact.served_model}]})
        assert url.endswith("/server_info")
        return _Response(
            {
                "version": "0.5.10.post1",
                "enable_deterministic_inference": True,
                "sampling_backend": "pytorch",
                "attention_backend": "fa3",
                "api_key": "must not be copied into the result",
            }
        )

    monkeypatch.setattr(dispatcher.urllib.request, "urlopen", urlopen)
    record = dispatcher._attest_local_server(
        "http://127.0.0.1:19000/v1", artifact=artifact
    )

    assert record["server"] == {
        "version": "0.5.10.post1",
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "attention_backend": "fa3",
    }
    assert "api_key" not in json.dumps(record)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"enable_deterministic_inference": False}, "deterministic server mismatch"),
        ({"sampling_backend": "flashinfer"}, "deterministic server mismatch"),
        ({"attention_backend": "trtllm"}, "deterministic server mismatch"),
    ],
)
def test_local_server_refuses_unseeded_semantics(
    tmp_path, monkeypatch, change, message
) -> None:
    import evals.signoflife.__main__ as dispatcher
    from test_model_attestation import _register_model

    model, _ = _register_model(tmp_path)
    artifact = dispatcher._verify_model_artifact(model)
    server = {
        "version": "0.5.10.post1",
        "enable_deterministic_inference": True,
        "sampling_backend": "pytorch",
        "attention_backend": "fa3",
        **change,
    }

    def urlopen(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        return (
            _Response({"data": [{"id": artifact.served_model}]})
            if url.endswith("/v1/models")
            else _Response(server)
        )

    monkeypatch.setattr(dispatcher.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match=message):
        dispatcher._attest_local_server(
            "http://127.0.0.1:19000/v1", artifact=artifact
        )


def test_seed_conformance_probe_repeats_and_varies_controls(monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    calls = []

    def complete(base_url, *, served_model, seed, timeout_s):
        calls.append(seed)
        return ("same" if seed == dispatcher._SEED_PROBE_SEEDS[0] else str(seed), "stop")

    monkeypatch.setattr(dispatcher, "_seed_probe_completion", complete)
    record = dispatcher._probe_seeded_sampling(
        "http://127.0.0.1:19000/v1",
        served_model="sign-of-life-sha256-test",
        timeout_s=180.0,
    )

    assert calls[0] == calls[1]
    assert calls[2:] == list(dispatcher._SEED_PROBE_SEEDS[1:])
    assert record["request_count"] == 5


def test_seed_conformance_probe_rejects_ignored_seeds(monkeypatch) -> None:
    import evals.signoflife.__main__ as dispatcher

    monkeypatch.setattr(
        dispatcher,
        "_seed_probe_completion",
        lambda *args, **kwargs: ("same", "stop"),
    )
    with pytest.raises(RuntimeError, match="different-seed controls were all identical"):
        dispatcher._probe_seeded_sampling(
            "http://127.0.0.1:19000/v1",
            served_model="sign-of-life-sha256-test",
            timeout_s=180.0,
        )
