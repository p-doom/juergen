"""Paired sampling uses one attested seed path or the run is not causal."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


def _completion(content: str) -> dict:
    return {
        "id": content,
        "object": "chat.completion",
        "created": 0,
        "model": "seed-consumer-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


@contextlib.contextmanager
def _capture_server(content: str):
    requests: list[dict] = []
    response = json.dumps(_completion(content)).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            requests.append(json.loads(body or b"{}"))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_seed_identity_is_arm_independent_and_distinct_across_trials() -> None:
    from evals.signoflife.__main__ import _SAMPLING_SEED_DOMAIN, _sampling_seed

    identity = {
        "suite_manifest_sha256": "a" * 64,
        "cell_id": "terminal_submit_only",
    }
    first = _sampling_seed(**identity, trial=1)
    material = (
        _SAMPLING_SEED_DOMAIN
        + identity["suite_manifest_sha256"].encode()
        + b"\0"
        + identity["cell_id"].encode()
        + b"\0"
        + b"1"
    )
    expected = int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF
    assert first == _sampling_seed(**identity, trial=1)
    assert first == (expected or 1)
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


def test_both_seed_consumers_send_every_planned_step_to_the_real_eval_client(
    monkeypatch,
) -> None:
    import verifiers.v1 as vf
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.clients.eval import EvalClient
    from verifiers.v1.interception import InterceptionServer
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.task import TaskData
    from verifiers.v1.trace import Trace, TraceTask

    from agent.agent import ContextTransport, EndpointTransport
    from evals.signoflife.__main__ import _sampling_seed
    from evals.signoflife.suite import load_suite

    task = load_suite().by_id("terminal_ls")
    planned = [
        (
            trial,
            step,
            _sampling_seed(
                suite_manifest_sha256=load_suite().manifest_sha256,
                cell_id=task.id,
                trial=trial,
            ),
        )
        for trial in range(1, 4)
        for step in range(1, task.max_steps + 1)
    ]

    async def through_interception(upstream_url: str) -> None:
        client = EvalClient(upstream_url, "test-key")
        server = InterceptionServer()
        try:
            async with server:
                for trial, step, seed in planned:
                    ctx = ModelContext(
                        model="seed-consumer-test",
                        client=client,
                        sampling=vf.Sampling(
                            temperature=0.7,
                            top_p=1.0,
                            max_tokens=256,
                            seed=seed,
                        ),
                    )
                    trace = Trace(
                        task=TraceTask(
                            type="SeedConsumerTask",
                            data=TaskData(
                                idx=trial,
                                name=task.id,
                                prompt=f"trial={trial} step={step}",
                            ),
                        )
                    )
                    session = RolloutSession(ctx=ctx, trace=trace)
                    async with server.acquire(session) as (base_url, secret):
                        transport = EndpointTransport(
                            endpoint=f"{base_url}/v1", secret=secret
                        )
                        try:
                            await transport.complete(
                                ctx,
                                {
                                    "messages": [
                                        vf.UserMessage(
                                            content=f"trial={trial} step={step}"
                                        )
                                    ]
                                },
                                session_id=trace.id,
                            )
                        finally:
                            await transport.close()
                    assert trace.calls[0].sampling.seed == seed
        finally:
            await client.close()

    async def through_context(upstream_url: str) -> None:
        client = EvalClient(upstream_url, "test-key")
        try:
            for trial, step, seed in planned:
                ctx = ModelContext(
                    model="seed-consumer-test",
                    client=client,
                    sampling=vf.Sampling(
                        temperature=0.7,
                        top_p=1.0,
                        max_tokens=256,
                        seed=seed,
                    ),
                )
                await ContextTransport().complete(
                    ctx,
                    {
                        "messages": [
                            vf.UserMessage(content=f"trial={trial} step={step}")
                        ]
                    },
                    session_id=f"trial-{trial}",
                )
        finally:
            await client.close()

    with _capture_server("upstream") as (upstream_url, upstream_requests):
        asyncio.run(through_interception(upstream_url))
        endpoint_requests = list(upstream_requests)
        upstream_requests.clear()

        with _capture_server("proxy") as (proxy_url, proxy_requests):
            for name in (
                "HTTP_PROXY",
                "http_proxy",
                "HTTPS_PROXY",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
            ):
                monkeypatch.setenv(name, proxy_url.removesuffix("/v1"))
            for name in ("NO_PROXY", "no_proxy"):
                monkeypatch.delenv(name, raising=False)

            async def prove_proxy_is_live() -> None:
                async with httpx.AsyncClient(trust_env=True) as client:
                    response = await client.post(f"{upstream_url}/chat/completions")
                    assert response.json()["id"] == "proxy"

            asyncio.run(prove_proxy_is_live())
            proxy_requests.clear()
            asyncio.run(through_context(upstream_url))

            assert proxy_requests == []
        context_requests = list(upstream_requests)

    expected_seeds = [seed for _trial, _step, seed in planned]
    assert [request["seed"] for request in endpoint_requests] == expected_seeds
    assert [request["seed"] for request in context_requests] == expected_seeds
    assert len(endpoint_requests) == len(context_requests) == len(planned)


def test_local_client_uses_only_the_fixed_unexported_no_auth_sentinel(
    tmp_path, monkeypatch
) -> None:
    from verifiers.v1.clients.config import resolve_api_key

    import evals.signoflife.__main__ as dispatcher

    monkeypatch.delenv(dispatcher.API_KEY_VAR, raising=False)
    monkeypatch.delenv(dispatcher._LOCAL_NO_AUTH_API_KEY_VAR, raising=False)
    config = dispatcher._eval_config(
        arm="ordered",
        tier="candidate",
        task_ids=["terminal_submit_only"],
        artifacts=tmp_path / "artifacts",
        traces_dir=tmp_path / "traces",
        pool={},
        base_url="http://127.0.0.1:19000/v1",
        temperature=0.7,
        top_p=1.0,
        max_tokens=256,
        served_model="sign-of-life-sha256-test",
        seed=1234567,
    )

    assert config.client.api_key_var == dispatcher._LOCAL_NO_AUTH_API_KEY_VAR
    assert resolve_api_key(config.client) == "EMPTY"
    assert dispatcher.API_KEY_VAR not in os.environ
    assert dispatcher._LOCAL_NO_AUTH_API_KEY_VAR not in os.environ


def test_local_launch_enables_deterministic_sampling_without_an_api_key(
    monkeypatch,
) -> None:
    from evals.signoflife.__main__ import _sglang_command, _sglang_environment
    from evals.signoflife.sglang_server import _server_arguments

    monkeypatch.setenv("SIGN_OF_LIFE_API_KEY", "never-in-the-server-child")

    command = _sglang_command(
        python="/sealed/venv/bin/python",
        model_path=Path("/sealed/model"),
        listener_fd=7,
        mem_fraction_static=0.65,
    )

    server_arguments = _server_arguments(
        SimpleNamespace(
            model_path="/sealed/model",
            mem_fraction_static=0.65,
        ),
        {"host": "127.0.0.1", "port": 19000},
    )
    assert "--enable-deterministic-inference" in server_arguments
    assert command[:3] == ["/sealed/venv/bin/python", "-I", "-B"]
    assert "--listener-fd" in command
    assert "--api-key" not in command
    assert not any("secret" in value for value in command)
    assert "SIGN_OF_LIFE_API_KEY" not in _sglang_environment()
    assert "never-in-the-server-child" not in json.dumps(_sglang_environment())


def test_local_launch_drops_unbound_semantic_environment(monkeypatch) -> None:
    from evals.signoflife.__main__ import _sglang_environment

    monkeypatch.setenv("TORCH_LOGS", "recompiles")
    assert "TORCH_LOGS" not in _sglang_environment()


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
