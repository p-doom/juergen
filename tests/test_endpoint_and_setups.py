"""The default transport.

`EndpointTransport` is the default: it posts to verifiers' interception endpoint,
the path that commits the turn to the trace graph, so losing it loses tokens,
logprobs and branch structure. It is driven here against a fake OpenAI-shaped HTTP
server rather than mocked.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import verifiers.v1 as vf

from agent.agent import EndpointTransport, ModelCallError, load_codec
from agent.history import History, ImageBudget, StatelessSingleTurn
from juergen_doubles import make_ctx, png


class FakeOpenAI:
    """Serves `/chat/completions` and records every request body."""

    def __init__(
        self, *, reply: str = "0 0 0 ;", status: int = 200, finish_reason: str = "stop"
    ) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                outer.headers.append(dict(self.headers))
                try:
                    outer.requests.append(json.loads(raw))
                except json.JSONDecodeError:
                    outer.requests.append({"__raw__": raw.decode("utf-8", "replace")})
                if status != 200:
                    body = json.dumps({"error": {"message": "upstream exploded"}}).encode()
                    self.send_response(status)
                else:
                    body = json.dumps(
                        {
                            "id": "cmpl-1",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "test-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": reply},
                                    "finish_reason": finish_reason,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def endpoint():
    servers: list[FakeOpenAI] = []

    def build(**kwargs) -> FakeOpenAI:
        server = FakeOpenAI(**kwargs)
        servers.append(server)
        return server

    yield build
    for server in servers:
        server.close()


def _geometry():
    from desktop.geometry import DisplayGeometry

    return DisplayGeometry(desktop_width=1920, desktop_height=1080)


def _history() -> History:
    history = History(n_history_frames=4)
    history.start(png())
    return history


def test_the_endpoint_transport_posts_and_returns_the_content(endpoint) -> None:
    server = endpoint(reply="10 20 0 ; +LMB -LMB")

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="sk-test")
        try:
            return await transport.complete(
                make_ctx(model="my-model"),
                {"messages": [vf.UserMessage(content="hi")], "max_tokens": 32},
                session_id="t1",
            )
        finally:
            await transport.close()

    assert asyncio.run(body()) == ("10 20 0 ; +LMB -LMB", "stop")
    request = server.requests[0]
    assert request["model"] == "my-model", "ctx.model is what goes on the wire"
    assert request["max_tokens"] == 32
    assert request["messages"][0]["role"] == "user"


def test_the_secret_is_sent_as_a_bearer_token(endpoint) -> None:
    server = endpoint()

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="sk-abc123")
        try:
            await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    asyncio.run(body())
    assert server.headers[0]["Authorization"] == "Bearer sk-abc123"


def test_only_the_unset_knobs_reach_the_wire_through_the_endpoint(endpoint) -> None:
    """The proxy applies ctx.sampling on top, so sending ours would be a lie."""
    from agent.agent import Agent

    server = endpoint()
    agent = Agent(
        codec=load_codec("deltatype_v2"),
        policy=StatelessSingleTurn(),
        budget=ImageBudget(max_images=1),
        transport=EndpointTransport(endpoint=server.base_url, secret="s"),
        max_tokens=64,
        temperature=0.0,
    )
    async def body():
        try:
            return await agent.step(
                make_ctx(model="m", temperature=0.9),
                history=_history(),
                instruction="do it",
                step=1,
                geometry=_geometry(),
                cursor=(0, 0),
            )
        finally:
            await agent.close()

    decision = asyncio.run(body())
    request = server.requests[0]
    assert "temperature" not in request, "the eval set it, so the harness must not send it"
    assert request["max_tokens"] == 64, "the eval left this unset, so the harness fills in"
    assert decision.sampling.temperature == 0.9
    assert decision.sampling.temperature_source == "ctx.sampling"


def test_an_image_rides_the_endpoint_body_as_a_data_url(endpoint) -> None:
    from agent.agent import Agent

    server = endpoint()
    agent = Agent(
        codec=load_codec("deltatype_v2"),
        policy=StatelessSingleTurn(),
        budget=ImageBudget(max_images=1),
        transport=EndpointTransport(endpoint=server.base_url, secret="s"),
    )
    async def body():
        try:
            await agent.step(
                make_ctx(),
                history=_history(),
                instruction="do it",
                step=1,
                geometry=_geometry(),
                cursor=(0, 0),
            )
        finally:
            await agent.close()

    asyncio.run(body())
    parts = server.requests[0]["messages"][1]["content"]
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_an_http_failure_becomes_a_model_call_error(endpoint) -> None:
    server = endpoint(status=500)

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        try:
            await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    with pytest.raises(ModelCallError):
        asyncio.run(body())


def test_an_unreachable_endpoint_becomes_a_model_call_error() -> None:
    async def body():
        transport = EndpointTransport(
            endpoint="http://127.0.0.1:1/v1", secret="s", timeout_s=2.0
        )
        try:
            await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    with pytest.raises(ModelCallError):
        asyncio.run(body())


def test_the_client_is_reused_across_turns_and_closed_once(endpoint) -> None:
    server = endpoint()
    seen = {}

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        ctx = make_ctx()
        await transport.complete(ctx, {"messages": []}, session_id=None)
        seen["first"] = transport._client
        await transport.complete(ctx, {"messages": []}, session_id=None)
        seen["second"] = transport._client
        await transport.close()
        seen["after_close"] = transport._client
        await transport.close()  # idempotent

    asyncio.run(body())
    assert seen["second"] is seen["first"], "one AsyncOpenAI per transport, not per turn"
    assert seen["after_close"] is None


def test_an_empty_completion_is_an_empty_string_not_none(endpoint) -> None:
    server = endpoint(reply="")

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        try:
            return await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    assert asyncio.run(body()) == ("", "stop")


def test_a_length_finish_is_reported_by_the_transport_not_swallowed(endpoint) -> None:
    """A turn cut off at `max_tokens` reaches the caller as a distinct fact. Reading
    only the content makes it a fake parse error, or a dispatched fragment."""
    server = endpoint(reply="10 20 0 ; +LM", finish_reason="length")

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        try:
            return await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    assert asyncio.run(body()) == ("10 20 0 ; +LM", "length")
