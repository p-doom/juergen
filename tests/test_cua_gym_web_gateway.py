from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from urllib.parse import parse_qs, urlsplit

import pytest

import evals.cua_gym.web.gateway as gateway_module
from evals.cua_gym.models import EndpointName
from evals.cua_gym.web.gateway import (
    CuaGymEpisodeGateway,
    CuaGymGatewayConfig,
    GatewayPhase,
)
from evals.cua_gym.web.hub import CuaGymHubDescriptor
from evals.cua_gym.web.manifest import (
    WEB_RUNTIME_MANIFEST_VERSION,
    CuaGymWebRuntimeManifest,
)

_MAX_RAW_HTTP_BYTES = 1024 * 1024
_RAW_HTTP_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class _RecordedRequest:
    method: str
    target: str
    headers: MappingProxyType[str, str]
    body: bytes


class _Backend:
    def __init__(self) -> None:
        self.requests: list[_RecordedRequest] = []
        self._token_lock = threading.Lock()
        self._next_token = 1
        self.block_path: str | None = None
        self.request_started = threading.Event()
        self.release_request = threading.Event()
        self.oversized_response: bytes | None = None
        self.response_written = threading.Event()
        self.release_response = threading.Event()
        self.expose_admin = False
        backend = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                backend._handle(self)

            def do_POST(self) -> None:
                backend._handle(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.release_request.set()
        self.release_response.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length) if length else b""
        request = _RecordedRequest(
            method=handler.command,
            target=handler.path,
            headers=MappingProxyType(
                {name.lower(): value for name, value in handler.headers.items()}
            ),
            body=body,
        )
        self.requests.append(request)
        path = urlsplit(handler.path).path
        if path == self.block_path:
            self.request_started.set()
            self.release_request.wait(timeout=5)
        if path == "/oversized":
            assert self.oversized_response is not None
            handler.send_response(200)
            handler.end_headers()
            handler.wfile.write(self.oversized_response)
            handler.wfile.flush()
            self.response_written.set()
            self.release_response.wait(timeout=5)
            return
        if path == "/post":
            payload = json.loads(body)
            response: dict[str, object] = {
                "status": "ok",
                "action": payload["action"],
            }
            if payload["action"] == "set":
                response["launch_url"] = (
                    f"/_cua_session?token={self._new_launch_token()}"
                )
            self._json(handler, response)
            return
        if path == "/state":
            self._json(handler, {"ok": True, "target": handler.path})
            return
        if path == "/_cua_session":
            handler.send_response(302)
            handler.send_header("Set-Cookie", "cua_mock_session=cookie; Path=/")
            handler.send_header("Location", "/?sid=__cua_session__")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        if path == "/opaque":
            payload = handler.path.encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/octet-stream")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
            return
        payload = {"ok": True, "target": handler.path}
        if self.expose_admin:
            payload["received_admin"] = request.headers.get("x-cua-admin-token")
        self._json(
            handler,
            payload,
            extra_headers=(
                ("Authorization", "backend-secret"),
                ("X-CUA-Admin-Token", "backend-admin"),
                ("Connection", "X-Backend-Hop"),
                ("X-Backend-Hop", "remove-me"),
            ),
        )

    def _new_launch_token(self) -> str:
        with self._token_lock:
            token = f"{self._next_token:064x}"
            self._next_token += 1
            return token

    @staticmethod
    def _json(
        handler: BaseHTTPRequestHandler,
        payload: object,
        *,
        extra_headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        body = json.dumps(payload).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        for name, value in extra_headers:
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


@pytest.fixture
def backends() -> Iterator[tuple[_Backend, _Backend]]:
    values = (_Backend(), _Backend())
    for backend in values:
        backend.start()
    try:
        yield values
    finally:
        for backend in values:
            backend.close()


def _manifest(*endpoints: EndpointName) -> CuaGymWebRuntimeManifest:
    return CuaGymWebRuntimeManifest(
        manifest_version=WEB_RUNTIME_MANIFEST_VERSION,
        dataset_revision="d" * 40,
        hub_revision="e" * 40,
        endpoint_apps={endpoint: f"{endpoint}_mock" for endpoint in endpoints},
        writable_directories=(".mock-states",),
        unsupported_endpoints={},
        unsupported_tasks={},
        task_scratch_paths=("/tmp/cua-task",),
        supported_task_count=1,
    )


def _free_port() -> int:
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        return int(available.getsockname()[1])


def _descriptor(
    tmp_path: Path,
    manifest: CuaGymWebRuntimeManifest,
    backend_urls: dict[EndpointName, str],
) -> CuaGymHubDescriptor:
    return CuaGymHubDescriptor(
        dataset_revision=manifest.dataset_revision,
        hub_revision=manifest.hub_revision,
        backend_urls=backend_urls,
        state_root=tmp_path / "hub-state",
        request_timeout_s=2.0,
        admin_token="a" * 64,
    )


def _gateway(
    tmp_path: Path,
    manifest: CuaGymWebRuntimeManifest,
    backend_urls: dict[EndpointName, str],
    *,
    episode_id: str,
    port: int | None = None,
    namespace_key: bytes = b"k" * 32,
    endpoints: tuple[EndpointName, ...] | None = None,
) -> CuaGymEpisodeGateway:
    selected_endpoints = endpoints or tuple(sorted(manifest.endpoint_apps))
    return CuaGymEpisodeGateway(
        config=CuaGymGatewayConfig(
            bind_host="127.0.0.1",
            port=port or _free_port(),
            endpoints=selected_endpoints,
            episode_id=episode_id,
            namespace_key=namespace_key,
            hub=_descriptor(tmp_path, manifest, backend_urls),
        ),
        manifest=manifest,
    )


def _request(
    gateway: CuaGymEpisodeGateway,
    method: str,
    target: str,
    *,
    endpoint: EndpointName,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    host: str | None = None,
) -> tuple[HTTPResponse, bytes]:
    connection = HTTPConnection("127.0.0.1", gateway.config.port, timeout=5)
    request_headers = {
        "Host": host or urlsplit(gateway.gateway_urls[endpoint]).netloc,
        **(headers or {}),
    }
    encoded = json.dumps(body).encode() if body is not None else None
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    connection.request(method, target, body=encoded, headers=request_headers)
    response = connection.getresponse()
    response_body = response.read()
    connection.close()
    return response, response_body


def _raw_request(gateway: CuaGymEpisodeGateway, request: bytes) -> tuple[int, bytes]:
    if len(request) > _MAX_RAW_HTTP_BYTES:
        raise ValueError("raw test request is too large")
    deadline = time.monotonic() + _RAW_HTTP_TIMEOUT_S
    with socket.create_connection(
        (gateway.config.bind_host, gateway.config.port),
        timeout=_RAW_HTTP_TIMEOUT_S,
    ) as connection:
        connection.settimeout(max(0.001, deadline - time.monotonic()))
        connection.sendall(request)
        response = bytearray()
        while len(response) <= _MAX_RAW_HTTP_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("raw test response timed out")
            connection.settimeout(remaining)
            chunk = connection.recv(
                min(65_536, _MAX_RAW_HTTP_BYTES + 1 - len(response))
            )
            if not chunk:
                break
            response.extend(chunk)
    if len(response) > _MAX_RAW_HTTP_BYTES:
        raise RuntimeError("raw test response is too large")
    status = int(response.split(b"\r\n", 1)[0].split()[1])
    return status, bytes(response)


def _setup_launch(
    gateway: CuaGymEpisodeGateway,
    backend: _Backend,
    endpoint: EndpointName,
    raw_sid: str,
) -> tuple[str, str]:
    response, body = _request(
        gateway,
        "POST",
        f"/post?sid={raw_sid}",
        endpoint=endpoint,
        body={"action": "set", "state": {}},
    )
    assert response.status == 200
    launch_url = json.loads(body)["launch_url"]
    token = parse_qs(urlsplit(launch_url).query)["token"][0]
    private_sid = parse_qs(urlsplit(backend.requests[-1].target).query)["sid"][0]
    return token, private_sid


def test_gateway_enforces_setup_rollout_and_evaluation(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="episode-1",
    )
    gateway.start()
    try:
        gateway.assert_healthy()
        empty_sid, _ = _request(
            gateway,
            "POST",
            "/post?sid=",
            endpoint=endpoint,
            body={"action": "set", "state": {}},
        )
        assert empty_sid.status == 400
        assert backend.requests == []
        response, body = _request(
            gateway,
            "POST",
            "/post?sid=raw-task-sid",
            endpoint=endpoint,
            body={"action": "set", "state": {}},
            headers={
                "Authorization": "Bearer attacker",
                "X-CUA-Admin-Token": "attacker",
                "Connection": "X-Attacker-Hop",
                "X-Attacker-Hop": "remove-me",
            },
        )
        assert response.status == 200
        token = parse_qs(urlsplit(json.loads(body)["launch_url"]).query)["token"][0]
        setup = backend.requests[-1]
        private_sid = parse_qs(urlsplit(setup.target).query)["sid"][0]
        assert len(private_sid) == 64
        assert "raw-task-sid" not in setup.target
        assert setup.headers["x-cua-admin-token"] == "a" * 64
        assert "authorization" not in setup.headers
        assert "x-attacker-hop" not in setup.headers

        gateway.transition(GatewayPhase.ROLLOUT)
        raw_request, _ = _request(
            gateway,
            "GET",
            "/upload?sid=raw-task-sid",
            endpoint=endpoint,
        )
        assert raw_request.status == 403

        navigation, _ = _request(
            gateway,
            "GET",
            "/mail?sid=raw-task-sid&view=compact",
            endpoint=endpoint,
            headers={"Accept": "text/html"},
        )
        assert navigation.status == 302
        assert navigation.getheader("Location") == f"/_cua_session?token={token}"
        aliased_exchange, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={token}&extra=1",
            endpoint=endpoint,
        )
        assert aliased_exchange.status == 403
        wrong_method, _ = _request(
            gateway,
            "POST",
            f"/_cua_session?token={token}",
            endpoint=endpoint,
        )
        assert wrong_method.status == 405
        exchange, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={token}",
            endpoint=endpoint,
        )
        assert exchange.status == 302
        assert exchange.getheader("Location") == (
            "/mail?sid=__cua_session__&view=compact"
        )
        reused, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={token}",
            endpoint=endpoint,
        )
        assert reused.status == 403
        gateway.wait_for_browser_session(1.0)

        denied_reset, _ = _request(
            gateway,
            "POST",
            "/post?sid=__cua_session__",
            endpoint=endpoint,
            body={"action": "reset"},
        )
        assert denied_reset.status == 403
        update, _ = _request(
            gateway,
            "POST",
            "/post?sid=__cua_session__",
            endpoint=endpoint,
            body={"action": "set_current", "state": {"messages": []}},
        )
        assert update.status == 200
        assert backend.requests[-1].headers.get("x-cua-admin-token") is None

        state, state_body = _request(
            gateway,
            "GET",
            "/state?sid=__cua_session__",
            endpoint=endpoint,
        )
        assert state.status == 200
        assert json.loads(state_body) == {
            "stored_state": {
                "ok": True,
                "target": "/state?sid=__cua_session__",
            },
            "has_custom_state": True,
            "sid": "__cua_session__",
        }

        upload, upload_body = _request(
            gateway,
            "GET",
            "/upload?sid=__cua_session__",
            endpoint=endpoint,
        )
        assert upload.status == 200
        assert upload.getheader("Authorization") is None
        assert upload.getheader("X-CUA-Admin-Token") is None
        assert upload.getheader("X-Backend-Hop") is None
        assert private_sid.encode() not in upload_body
        assert b"__cua_session__" in upload_body
        assert parse_qs(urlsplit(backend.requests[-1].target).query)["sid"] == [
            private_sid
        ]

        gateway.transition(GatewayPhase.EVALUATE)
        rollout_write, _ = _request(
            gateway,
            "POST",
            "/post?sid=raw-task-sid",
            endpoint=endpoint,
            body={"action": "reset"},
        )
        assert rollout_write.status == 403
        reward, _ = _request(
            gateway,
            "GET",
            "/go?sid=raw-task-sid",
            endpoint=endpoint,
        )
        assert reward.status == 200
        assert backend.requests[-1].headers["x-cua-admin-token"] == "a" * 64
        assert parse_qs(urlsplit(backend.requests[-1].target).query)["sid"] == [
            private_sid
        ]
        assert gateway.private_sessions == ((endpoint, private_sid),)
        gateway.transition(GatewayPhase.CLOSED)
    finally:
        gateway.close()
    with pytest.raises(RuntimeError, match="not running"):
        gateway.assert_healthy()


def test_endpoint_topology_is_exact_and_routes_by_host(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    drive = EndpointName("drive")
    gmail = EndpointName("gmail")
    manifest = _manifest(drive, gmail)
    gateway = _gateway(
        tmp_path,
        manifest,
        {drive: backends[0].url, gmail: backends[1].url},
        episode_id="endpoint-routing",
        endpoints=(gmail,),
    )
    gateway.start()
    try:
        allowed, _ = _request(gateway, "GET", "/", endpoint=gmail)
        episode_label = urlsplit(gateway.gateway_urls[gmail]).hostname.split(".", 1)[0]
        rejected, _ = _request(
            gateway,
            "GET",
            "/",
            endpoint=gmail,
            host=f"{episode_label}.drive.cua.internal:{gateway.config.port}",
        )
        assert allowed.status == 200
        assert rejected.status == 401
        assert len(backends[0].requests) == 0
        assert len(backends[1].requests) == 1
        assert set(gateway.gateway_urls) == {gmail}
        hostname = urlsplit(gateway.gateway_urls[gmail]).hostname
        assert hostname is not None
        assert hostname.endswith(".gmail.cua.internal")
        assert len(hostname.split(".", 1)[0]) == 52
    finally:
        gateway.close()


def test_host_authority_is_unambiguous(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="host-canonicalization",
    )
    gateway.start()
    expected = urlsplit(gateway.gateway_urls[endpoint]).netloc
    try:
        accepted, _ = _request(
            gateway, "GET", "/", endpoint=endpoint, host=expected.upper()
        )
        assert accepted.status == 200

        for invalid in (
            expected.rsplit(":", 1)[0],
            expected.rsplit(":", 1)[0] + ":1",
            f"[::1]:{gateway.config.port}",
            expected + ".",
        ):
            rejected, _ = _request(gateway, "GET", "/", endpoint=endpoint, host=invalid)
            assert rejected.status == 401

        missing_status, _ = _raw_request(
            gateway,
            b"GET / HTTP/1.1\r\nConnection: close\r\n\r\n",
        )
        assert missing_status == 401

        for first, second in (
            (expected, "attacker.invalid"),
            ("attacker.invalid", expected),
        ):
            duplicate_status, _ = _raw_request(
                gateway,
                (
                    "GET / HTTP/1.1\r\n"
                    f"Host: {first}\r\n"
                    f"Host: {second}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii"),
            )
            assert duplicate_status == 401

        assert len(backend.requests) == 1
    finally:
        gateway.close()


def test_request_framing_is_unambiguous_and_rejected_before_body_or_backend(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="request-framing",
    )
    gateway.start()
    host = urlsplit(gateway.gateway_urls[endpoint]).netloc
    try:
        for first, second in (("0", "1"), ("1", "0")):
            duplicate_status, _ = _raw_request(
                gateway,
                (
                    "POST /post?sid=raw HTTP/1.1\r\n"
                    f"Host: {host}\r\n"
                    f"Content-Length: {first}\r\n"
                    f"Content-Length: {second}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii"),
            )
            assert duplicate_status == 400

        transfer_status, _ = _raw_request(
            gateway,
            (
                "POST /post?sid=raw HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Transfer-Encoding: chunked\r\n"
                "Content-Length: 4\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii"),
        )
        assert transfer_status == 400
        assert backends[0].requests == []
    finally:
        gateway.close()


def test_private_session_namespaces_differ_across_episodes(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    private_sids: list[str] = []
    gateway_urls: list[str] = []
    for index, backend in enumerate(backends):
        gateway = _gateway(
            tmp_path,
            manifest,
            {endpoint: backend.url},
            episode_id=f"episode-{index}",
        )
        with gateway:
            _, private_sid = _setup_launch(gateway, backend, endpoint, "shared-raw-sid")
            private_sids.append(private_sid)
            gateway_urls.append(gateway.gateway_urls[endpoint])
    assert len(set(private_sids)) == 2
    assert len(set(gateway_urls)) == 2


def test_gateway_rejects_missing_and_foreign_episode_credentials_before_backend_io(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    first = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="first",
        namespace_key=b"a" * 32,
    )
    second = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[1].url},
        episode_id="second",
        namespace_key=b"b" * 32,
    )
    first.start()
    second.start()
    try:
        missing, _ = _request(
            first,
            "POST",
            "/post",
            endpoint=endpoint,
            host=f"gmail.cua.internal:{first.config.port}",
            headers={"Content-Length": str(64 * 1024 * 1024 + 1)},
        )
        foreign_host = urlsplit(second.gateway_urls[endpoint]).hostname
        assert foreign_host is not None
        foreign, _ = _request(
            first,
            "GET",
            "/",
            endpoint=endpoint,
            host=f"{foreign_host}:{first.config.port}",
        )
        assert missing.status == 401
        assert foreign.status == 401
        assert backends[0].requests == []
        assert backends[1].requests == []
    finally:
        first.close()
        second.close()


def test_secondary_endpoint_claims_the_selected_session(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    drive = EndpointName("drive")
    gmail = EndpointName("gmail")
    manifest = _manifest(drive, gmail)
    gateway = _gateway(
        tmp_path,
        manifest,
        {drive: backends[0].url, gmail: backends[1].url},
        episode_id="cross-endpoint",
    )
    gateway.start()
    try:
        drive_token, drive_private = _setup_launch(
            gateway, backends[0], drive, "selected-sid"
        )
        gmail_token, _ = _setup_launch(gateway, backends[1], gmail, "selected-sid")
        distractor_token, _ = _setup_launch(
            gateway, backends[0], drive, "distractor-sid"
        )
        gateway.transition(GatewayPhase.ROLLOUT)

        wrong_endpoint, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={distractor_token}",
            endpoint=gmail,
        )
        assert wrong_endpoint.status == 403
        primary_navigation, _ = _request(
            gateway,
            "GET",
            "/mail?sid=selected-sid",
            endpoint=gmail,
            headers={"Accept": "text/html"},
        )
        assert primary_navigation.getheader("Location") == (
            f"/_cua_session?token={gmail_token}"
        )
        _request(
            gateway,
            "GET",
            f"/_cua_session?token={gmail_token}",
            endpoint=gmail,
        )

        secondary_navigation, _ = _request(
            gateway,
            "GET",
            "/files?sid=__cua_session__",
            endpoint=drive,
            headers={"Accept": "text/html"},
        )
        assert secondary_navigation.getheader("Location") == (
            f"/_cua_session?token={drive_token}"
        )
        rejected, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={distractor_token}",
            endpoint=drive,
        )
        assert rejected.status == 403
        exchange, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={drive_token}",
            endpoint=drive,
        )
        assert exchange.getheader("Location") == "/files?sid=__cua_session__"
        file_response, _ = _request(
            gateway,
            "GET",
            "/files/__cua_session__/document.txt",
            endpoint=drive,
        )
        assert file_response.status == 200
        assert urlsplit(backends[0].requests[-1].target).path == (
            f"/files/{drive_private}/document.txt"
        )
    finally:
        gateway.close()


def test_private_sid_in_opaque_backend_data_is_not_exposed(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="opaque-data",
    )
    gateway.start()
    try:
        token, _ = _setup_launch(gateway, backends[0], endpoint, "selected-sid")
        gateway.transition(GatewayPhase.ROLLOUT)
        _request(
            gateway,
            "GET",
            f"/_cua_session?token={token}",
            endpoint=endpoint,
        )
        response, body = _request(
            gateway,
            "GET",
            "/opaque?sid=__cua_session__",
            endpoint=endpoint,
        )
        assert response.status == 502
        assert b"__cua_session__" not in body
        assert b"gateway backend failure" in body
    finally:
        gateway.close()


def test_admin_token_in_backend_payload_is_not_exposed(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    backend.expose_admin = True
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="admin-token-redaction",
    )
    gateway.start()
    try:
        gateway.transition(GatewayPhase.ROLLOUT)
        gateway.transition(GatewayPhase.EVALUATE)
        response, body = _request(
            gateway,
            "GET",
            "/go?sid=raw-sid",
            endpoint=endpoint,
        )
        assert response.status == 502
        assert body == b'{"error": "gateway backend failure"}'
        assert gateway.config.hub.admin_token.encode() not in body
        assert backend.requests[-1].headers["x-cua-admin-token"] == "a" * 64
    finally:
        gateway.close()


def test_transition_waits_for_requests_from_the_previous_phase(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    backend.block_path = "/slow"
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="drain",
    )
    gateway.start()
    executor = ThreadPoolExecutor(max_workers=3)
    request = executor.submit(_request, gateway, "GET", "/slow", endpoint=endpoint)
    try:
        assert backend.request_started.wait(timeout=2)
        transition = executor.submit(gateway.transition, GatewayPhase.ROLLOUT)
        with pytest.raises(TimeoutError):
            transition.result(timeout=0.05)
        target_request = executor.submit(
            _request,
            gateway,
            "GET",
            "/target-phase",
            endpoint=endpoint,
        )
        with pytest.raises(TimeoutError):
            target_request.result(timeout=0.05)
        assert gateway.phase is GatewayPhase.SETUP
        assert [urlsplit(item.target).path for item in backend.requests] == ["/slow"]
        backend.release_request.set()
        assert request.result(timeout=2)[0].status == 200
        transition.result(timeout=2)
        assert target_request.result(timeout=2)[0].status == 200
        assert gateway.phase is GatewayPhase.ROLLOUT
        assert [urlsplit(item.target).path for item in backend.requests] == [
            "/slow",
            "/target-phase",
        ]
    finally:
        backend.release_request.set()
        executor.shutdown(wait=True)
        gateway.close()


def test_close_wins_over_a_transition_waiting_for_the_previous_phase(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    backend.block_path = "/slow"
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="close-during-transition",
    )
    gateway.start()
    executor = ThreadPoolExecutor(max_workers=3)
    request = executor.submit(_request, gateway, "GET", "/slow", endpoint=endpoint)
    try:
        assert backend.request_started.wait(timeout=2)
        transition = executor.submit(gateway.transition, GatewayPhase.ROLLOUT)
        with pytest.raises(TimeoutError):
            transition.result(timeout=0.05)
        assert transition.running()
        with gateway._condition:
            assert gateway._transitioning

        close = executor.submit(gateway.close)
        with pytest.raises(TimeoutError):
            close.result(timeout=0.05)
        assert close.running()
        assert gateway.phase is GatewayPhase.CLOSED

        backend.release_request.set()
        assert request.result(timeout=2)[0].status == 200
        close.result(timeout=2)
        with pytest.raises(RuntimeError, match="closed during phase transition"):
            transition.result(timeout=2)
        assert gateway.phase is GatewayPhase.CLOSED
    finally:
        backend.release_request.set()
        executor.shutdown(wait=True)
        gateway.close()


def test_launch_token_is_claimed_before_backend_io(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="token-race",
    )
    gateway.start()
    token, _ = _setup_launch(gateway, backend, endpoint, "raw-sid")
    gateway.transition(GatewayPhase.ROLLOUT)
    backend.block_path = "/_cua_session"
    executor = ThreadPoolExecutor(max_workers=2)
    first = executor.submit(
        _request,
        gateway,
        "GET",
        f"/_cua_session?token={token}",
        endpoint=endpoint,
    )
    try:
        assert backend.request_started.wait(timeout=2)
        duplicate, _ = _request(
            gateway,
            "GET",
            f"/_cua_session?token={token}",
            endpoint=endpoint,
        )
        assert duplicate.status == 403
        assert (
            sum(
                urlsplit(request.target).path == "/_cua_session"
                for request in backend.requests
            )
            == 1
        )
        backend.release_request.set()
        assert first.result(timeout=2)[0].status == 302
    finally:
        backend.release_request.set()
        executor.shutdown(wait=True)
        gateway.close()


def test_close_disconnects_keepalive_clients_and_releases_the_port(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    port = _free_port()
    first = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="first",
        port=port,
    )
    first.start()
    browser = HTTPConnection("127.0.0.1", port, timeout=2)
    browser.request(
        "GET",
        "/",
        headers={"Host": urlsplit(first.gateway_urls[endpoint]).netloc},
    )
    response = browser.getresponse()
    response.read()
    first.close()
    with pytest.raises((OSError, HTTPException)):
        browser.request(
            "GET",
            "/",
            headers={"Host": urlsplit(first.gateway_urls[endpoint]).netloc},
        )
        browser.getresponse()
    browser.close()

    second = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[1].url},
        episode_id="second",
        port=port,
    )
    with second:
        response, _ = _request(second, "GET", "/", endpoint=endpoint)
        assert response.status == 200


def test_close_interrupts_a_client_stalled_in_the_request_body(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="stalled-body",
    )
    gateway.start()
    client = socket.create_connection(
        (gateway.config.bind_host, gateway.config.port), timeout=2.0
    )
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        host = urlsplit(gateway.gateway_urls[endpoint]).netloc
        client.sendall(
            (
                "POST /post?sid=raw HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Content-Length: 100\r\n\r\n"
                "{"
            ).encode("ascii")
        )
        close = executor.submit(gateway.close)
        close.result(timeout=2.0)
        assert backends[0].requests == []
    finally:
        client.close()
        executor.shutdown(wait=True)
        gateway.close()


def test_invalid_configuration_fails_before_socket_open(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    port = _free_port()
    with pytest.raises(ValueError, match="absent from the web manifest"):
        _gateway(
            tmp_path,
            manifest,
            {endpoint: backends[0].url},
            episode_id="bad-topology",
            port=port,
            endpoints=(EndpointName("unknown"),),
        )
    with socket.socket() as available:
        available.bind(("127.0.0.1", port))

    descriptor = _descriptor(tmp_path, manifest, {endpoint: backends[0].url})
    with pytest.raises(ValueError, match="at least 32 bytes"):
        CuaGymGatewayConfig(
            bind_host="127.0.0.1",
            port=_free_port(),
            endpoints=(endpoint,),
            episode_id="bad-secret",
            namespace_key=b"short",
            hub=descriptor,
        )


def test_oversized_requests_are_rejected_before_body_read(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="body-limit",
    )
    gateway.start()
    try:
        connection = HTTPConnection("127.0.0.1", gateway.config.port, timeout=2)
        connection.putrequest("POST", "/post", skip_host=True)
        connection.putheader("Host", urlsplit(gateway.gateway_urls[endpoint]).netloc)
        connection.putheader("Content-Length", str(64 * 1024 * 1024 + 1))
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        connection.close()
        assert response.status == 400
        assert backends[0].requests == []
    finally:
        gateway.close()


def test_unauthorized_phase_requests_are_rejected_before_body_read(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="request-framing",
    )
    gateway.start()
    host = urlsplit(gateway.gateway_urls[endpoint]).netloc
    try:
        gateway.transition(GatewayPhase.ROLLOUT)
        gateway.transition(GatewayPhase.EVALUATE)
        denied_status, _ = _raw_request(
            gateway,
            (
                "POST /post?sid=raw HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Content-Length: 67108864\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii"),
        )
        assert denied_status == 403
        assert backends[0].requests == []
    finally:
        gateway.close()


def test_oversized_backend_response_is_bounded_and_rejected(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = EndpointName("gmail")
    backend = backends[0]
    monkeypatch.setattr(gateway_module, "_MAX_RESPONSE_BYTES", 1024)
    backend.oversized_response = b"x" * 1025
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backend.url},
        episode_id="backend-body-limit",
    )
    gateway.start()
    executor = ThreadPoolExecutor(max_workers=1)
    request = executor.submit(
        _request,
        gateway,
        "GET",
        "/oversized",
        endpoint=endpoint,
    )
    try:
        assert backend.response_written.wait(timeout=2)
        response, body = request.result(timeout=2)
        assert response.status == 502
        assert json.loads(body) == {"error": "gateway backend failure"}
    finally:
        backend.release_response.set()
        executor.shutdown(wait=True)
        gateway.close()


def test_phase_transitions_are_strict(
    backends: tuple[_Backend, _Backend],
    tmp_path: Path,
) -> None:
    endpoint = EndpointName("gmail")
    manifest = _manifest(endpoint)
    gateway = _gateway(
        tmp_path,
        manifest,
        {endpoint: backends[0].url},
        episode_id="phases",
    )
    with pytest.raises(RuntimeError, match="setup -> evaluate"):
        gateway.transition(GatewayPhase.EVALUATE)
    gateway.transition(GatewayPhase.ROLLOUT)
    with pytest.raises(RuntimeError, match="rollout -> setup"):
        gateway.transition(GatewayPhase.SETUP)
    gateway.close()
