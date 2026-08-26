from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.types import SamplingConfig

from evals.signoflife.__main__ import (
    _attest_local_server,
    _sglang,
    _verify_model_artifact,
)


@pytest.mark.skipif(
    "JUERGEN_SGLANG_MODEL" not in os.environ,
    reason="JUERGEN_SGLANG_MODEL is not set",
)
def test_local_frontdoor_serves_a_real_completion(tmp_path) -> None:
    artifact = _verify_model_artifact(Path(os.environ["JUERGEN_SGLANG_MODEL"]))
    runtime = os.environ["JUERGEN_SGLANG_PYTHON"]

    with _sglang(
        python=runtime,
        model_path=artifact.model_path,
        log_path=tmp_path / "sglang.log",
        mem_fraction_static=0.4,
        ready_timeout_s=900,
    ) as server:
        attestation = _attest_local_server(server.base_url, artifact=artifact)

        async def consume():
            client = EvalClient(server.base_url, "EMPTY")
            try:
                return await client.get_response(
                    ChatDialect(),
                    {"messages": [{"role": "user", "content": "Reply with OK."}]},
                    artifact.served_model,
                    SamplingConfig(
                        temperature=0,
                        max_tokens=8,
                        seed=7,
                    ),
                )
            finally:
                await client.close()

        response = asyncio.run(consume())

    assert attestation["served_model"] == artifact.served_model
    assert isinstance(response.message.content, str)
    assert response.message.content
