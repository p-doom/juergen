from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Self
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

from ..models import EndpointName
from .hub import CuaGymHubDescriptor
from .manifest import CuaGymWebRuntimeManifest

_ADMIN_HEADER = "x-cua-admin-token"
_PLACEHOLDER_SID = "__cua_session__"
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_PRIVILEGED_PATHS = {"/post", "/state", "/go", "/state-inspector"}
_INSPECTOR_PATHS = {"/go", "/state-inspector"}


class GatewayPhase(StrEnum):
    SETUP = "setup"
    ROLLOUT = "rollout"
    EVALUATE = "evaluate"
    CLOSED = "closed"


@dataclass(frozen=True)
class CuaGymGatewayConfig:
    bind_host: str
    port: int
    endpoints: tuple[EndpointName, ...]
    episode_id: str
    namespace_key: bytes = field(repr=False)
    hub: CuaGymHubDescriptor = field(repr=False)

    def __post_init__(self) -> None:
        try:
            bind_address = ipaddress.ip_address(self.bind_host)
        except ValueError as error:
            raise ValueError("gateway bind_host must be an IPv4 address") from error
        if bind_address.version != 4:
            raise ValueError("gateway bind_host must be an IPv4 address")
        if type(self.port) is not int or not 1 <= self.port <= 65_535:
            raise ValueError("gateway port must be within 1-65535")
        if (
            not self.endpoints
            or tuple(sorted(self.endpoints)) != self.endpoints
            or len(set(self.endpoints)) != len(self.endpoints)
            or any(
                not isinstance(endpoint, str) or not endpoint
                for endpoint in self.endpoints
            )
        ):
            raise ValueError("gateway endpoints must be sorted, unique, and non-empty")
        if (
            not isinstance(self.episode_id, str)
            or not self.episode_id
            or "\0" in self.episode_id
        ):
            raise ValueError(
                "gateway episode_id must be a non-empty string without NUL"
            )
        if not isinstance(self.namespace_key, bytes) or len(self.namespace_key) < 32:
            raise ValueError("gateway namespace_key must contain at least 32 bytes")


@dataclass(frozen=True)
class _GatewayResponse:
    status: int
    reason: str | None
    headers: tuple[tuple[str, str], ...]
    body: bytes


@dataclass
class _PendingLaunch:
    endpoint: EndpointName
    raw_sid: str
    launch_url: str
    token: str
    target: str


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def get_request(self) -> tuple[socket.socket, object]:
        connection, address = super().get_request()
        with self._connections_lock:
            self._connections.add(connection)
        return connection, address

    def shutdown_request(self, request: socket.socket) -> None:  # type: ignore[override]
        with self._connections_lock:
            self._connections.discard(request)
        super().shutdown_request(request)

    def disconnect_clients(self) -> None:
        with self._connections_lock:
            for connection in tuple(self._connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


class CuaGymEpisodeGateway:
    def __init__(
        self,
        *,
        config: CuaGymGatewayConfig,
        manifest: CuaGymWebRuntimeManifest,
    ) -> None:
        config.hub.validate_manifest(manifest)
        unknown_endpoints = set(config.endpoints) - set(manifest.endpoint_apps)
        if unknown_endpoints:
            raise ValueError(
                "gateway endpoints are absent from the web manifest: "
                + ", ".join(sorted(unknown_endpoints))
            )
        self.config = config
        self.manifest = manifest
        episode_label = base64.b32encode(
            hmac.new(
                config.namespace_key,
                f"gateway\0{config.episode_id}".encode(),
                hashlib.sha256,
            ).digest()
        ).decode("ascii").lower().rstrip("=")
        self.gateway_hostnames = MappingProxyType(
            {
                endpoint: f"{episode_label}.{manifest.hostname(endpoint)}"
                for endpoint in config.endpoints
            }
        )
        self.gateway_urls = MappingProxyType(
            {
                endpoint: f"http://{self.gateway_hostnames[endpoint]}:{config.port}"
                for endpoint in config.endpoints
            }
        )
        self._phase = GatewayPhase.SETUP
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._transitioning = False
        self._active_requests = {phase: 0 for phase in GatewayPhase}
        self._private_sids: dict[tuple[EndpointName, str], str] = {}
        self._pending_launches_by_sid: dict[
            tuple[EndpointName, str], _PendingLaunch
        ] = {}
        self._pending_launches_by_token: dict[
            tuple[EndpointName, str], _PendingLaunch
        ] = {}
        self._selected_raw_sids: dict[EndpointName, str] = {}
        self._completed_browser_sessions = 0
        self._server: _GatewayServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def phase(self) -> GatewayPhase:
        with self._lock:
            return self._phase

    @property
    def private_sessions(self) -> tuple[tuple[EndpointName, str], ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        (endpoint, private_sid)
                        for (endpoint, _), private_sid in self._private_sids.items()
                    ),
                    key=lambda pair: (pair[0], pair[1]),
                )
            )

    def start(self) -> None:
        if self._server is not None or self._thread is not None:
            raise RuntimeError("CUA-Gym gateway has already been started")
        if self.phase is not GatewayPhase.SETUP:
            raise RuntimeError("CUA-Gym gateway can only start in setup phase")
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                gateway._handle(self)

            def do_HEAD(self) -> None:
                gateway._handle(self)

            def do_POST(self) -> None:
                gateway._handle(self)

            def do_PUT(self) -> None:
                gateway._handle(self)

            def do_PATCH(self) -> None:
                gateway._handle(self)

            def do_DELETE(self) -> None:
                gateway._handle(self)

            def do_OPTIONS(self) -> None:
                gateway._handle(self)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = _GatewayServer(
            (self.config.bind_host, self.config.port),
            Handler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"cua-gym-gateway-{self.config.port}",
            daemon=True,
        )
        self._thread.start()

    def assert_healthy(self) -> None:
        thread = self._thread
        server = self._server
        if (
            server is None
            or thread is None
            or not thread.is_alive()
            or server.fileno() < 0
            or self.phase is GatewayPhase.CLOSED
        ):
            raise RuntimeError("CUA-Gym gateway is not running")

    def transition(self, phase: GatewayPhase) -> None:
        target = GatewayPhase(phase)
        allowed = {
            GatewayPhase.SETUP: GatewayPhase.ROLLOUT,
            GatewayPhase.ROLLOUT: GatewayPhase.EVALUATE,
            GatewayPhase.EVALUATE: GatewayPhase.CLOSED,
        }
        with self._condition:
            if self._transitioning:
                raise RuntimeError("gateway phase transition is already in progress")
            previous = self._phase
            expected = allowed.get(previous)
            if target is not expected:
                raise RuntimeError(
                    f"invalid gateway phase transition: {previous} -> {target}"
                )
            self._transitioning = True
            try:
                while self._active_requests[previous] > 0:
                    self._condition.wait()
                if self._phase is GatewayPhase.CLOSED:
                    raise RuntimeError("gateway closed during phase transition")
                self._phase = target
            finally:
                self._transitioning = False
                self._condition.notify_all()

    def wait_for_browser_session(self, timeout_s: float) -> None:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._completed_browser_sessions == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "browser session exchange did not finish: "
                        f"pending navigations={len(self._pending_launches_by_sid)}, "
                        f"pending tokens={len(self._pending_launches_by_token)}"
                    )
                self._condition.wait(timeout=remaining)

    def close(self) -> None:
        with self._condition:
            previous = self._phase
            self._phase = GatewayPhase.CLOSED
            self._condition.notify_all()
        server = self._server
        thread = self._thread
        if server is not None:
            server.shutdown()
            server.disconnect_clients()
        with self._condition:
            while (
                previous is not GatewayPhase.CLOSED
                and self._active_requests[previous] > 0
            ):
                self._condition.wait()
        if server is not None:
            server.server_close()
            self._server = None
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError("CUA-Gym gateway thread did not stop")
            self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        with self._condition:
            while self._transitioning:
                self._condition.wait()
            phase = self._phase
            self._active_requests[phase] += 1
        try:
            try:
                response = self._route(handler, phase)
            except ValueError as error:
                response = self._error(HTTPStatus.BAD_REQUEST, str(error))
            except Exception:  # noqa: BLE001
                response = self._error(
                    HTTPStatus.BAD_GATEWAY, "gateway backend failure"
                )
            handler.send_response(response.status, response.reason)
            for name, value in response.headers:
                handler.send_header(name, value)
            if not any(
                name.lower() == "content-length" for name, _ in response.headers
            ):
                handler.send_header("Content-Length", str(len(response.body)))
            handler.end_headers()
            if handler.command != "HEAD":
                handler.wfile.write(response.body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self._condition:
                self._active_requests[phase] -= 1
                self._condition.notify_all()

    def _route(
        self,
        handler: BaseHTTPRequestHandler,
        phase: GatewayPhase,
    ) -> _GatewayResponse:
        original_close_connection = handler.close_connection
        handler.close_connection = True
        host_headers = handler.headers.get_all("Host", [])
        endpoint = (
            self._endpoint_for_host(host_headers[0]) if len(host_headers) == 1 else None
        )
        if endpoint is None:
            return self._error(
                HTTPStatus.UNAUTHORIZED,
                "episode authentication failed",
            )
        split = urlsplit(handler.path)
        if split.scheme or split.netloc or split.fragment:
            raise ValueError("gateway request target must be an origin-form path")
        if phase is GatewayPhase.CLOSED:
            return self._error(HTTPStatus.GONE, "episode is closed")
        raw_sid = self._query_sid(split.query)
        content_length = self._content_length(handler)
        if phase in {GatewayPhase.SETUP, GatewayPhase.ROLLOUT} and self._is_navigation(
            handler
        ):
            if content_length:
                return self._error(
                    HTTPStatus.BAD_REQUEST,
                    "navigation requests must not contain a body",
                )
            bootstrap: _GatewayResponse | None = None
            if raw_sid is not None and raw_sid != _PLACEHOLDER_SID:
                bootstrap = self._bootstrap_raw_navigation(endpoint, raw_sid, split)
            elif raw_sid == _PLACEHOLDER_SID:
                bootstrap = self._bootstrap_placeholder_navigation(endpoint, split)
            if bootstrap is not None:
                handler.close_connection = original_close_connection
                return bootstrap

        pending_launch_token = (
            self._launch_token(handler.path) if split.path == "/_cua_session" else None
        )
        if phase is GatewayPhase.ROLLOUT and pending_launch_token is None:
            denied = self._rollout_denial(handler.command, split.path, split.query)
            if denied is not None:
                return denied
        elif phase is GatewayPhase.EVALUATE:
            if handler.command != "GET" or split.path not in {"/go", "/state"}:
                return self._error(
                    HTTPStatus.FORBIDDEN,
                    "only verifier reads are allowed during evaluation",
                )

        claimed_launch: _PendingLaunch | None = None
        if split.path == "/_cua_session":
            if handler.command != "GET":
                return self._error(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "session exchange requires GET",
                )
            if content_length:
                return self._error(
                    HTTPStatus.BAD_REQUEST,
                    "session exchange must not contain a body",
                )
            if pending_launch_token is None:
                return self._error(HTTPStatus.FORBIDDEN, "unknown setup session token")
            claimed_launch = self._claim_launch(endpoint, pending_launch_token)
            if claimed_launch is None:
                return self._error(HTTPStatus.FORBIDDEN, "unknown setup session token")

        body = handler.rfile.read(content_length) if content_length else b""
        if len(body) != content_length:
            raise ValueError("request body ended before its declared length")
        handler.close_connection = original_close_connection
        if (
            phase is GatewayPhase.ROLLOUT
            and pending_launch_token is None
            and split.path == "/post"
        ):
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._error(HTTPStatus.BAD_REQUEST, "invalid state update")
            if not isinstance(payload, dict) or payload.get("action") != "set_current":
                return self._error(
                    HTTPStatus.FORBIDDEN,
                    "only current-state updates are allowed during rollout",
                )

        rewritten = self._rewrite_request_target(endpoint, split)
        inject_admin = split.path in _PRIVILEGED_PATHS and (
            (
                phase is GatewayPhase.SETUP
                and raw_sid is not None
                and raw_sid != _PLACEHOLDER_SID
            )
            or phase is GatewayPhase.EVALUATE
        )
        response = self._proxy(
            handler=handler,
            endpoint=endpoint,
            target=rewritten,
            body=body,
            inject_admin=inject_admin,
        )
        response = self._redact_private_sids(endpoint, response)
        if split.path == "/state":
            response = self._restore_state_response(response)
        if phase is GatewayPhase.SETUP and split.path == "/post":
            self._capture_launch(endpoint, raw_sid, response)
        if claimed_launch is not None:
            response = self._rewrite_launch_response(claimed_launch, response)
        return response

    @staticmethod
    def _restore_state_response(response: _GatewayResponse) -> _GatewayResponse:
        """Adapt the hardened Hub's raw state to the pinned client API."""

        if response.status != HTTPStatus.OK:
            return response
        try:
            state = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return response
        if not isinstance(state, dict) or {
            "stored_state",
            "has_custom_state",
        } <= set(state):
            return response
        body = json.dumps(
            {
                "stored_state": state,
                "has_custom_state": bool(state),
                "sid": _PLACEHOLDER_SID,
            }
        ).encode()
        return _GatewayResponse(
            status=response.status,
            reason=response.reason,
            headers=response.headers,
            body=body,
        )

    def _rollout_denial(
        self,
        method: str,
        path: str,
        query: str,
    ) -> _GatewayResponse | None:
        if path in _INSPECTOR_PATHS or path == "/_cua_session":
            return self._error(HTTPStatus.FORBIDDEN, "verifier endpoint is unavailable")
        sid = self._query_sid(query)
        if sid not in {None, _PLACEHOLDER_SID}:
            return self._error(HTTPStatus.FORBIDDEN, "raw session IDs are unavailable")
        if path == "/state" and method != "GET":
            return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "state is read-only")
        if path == "/post" and method != "POST":
            return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "invalid state method")
        return None

    def _rewrite_request_target(
        self,
        endpoint: EndpointName,
        split: SplitResult,
    ) -> str:
        path = split.path
        pairs = parse_qsl(split.query, keep_blank_values=True)
        rewritten_pairs: list[tuple[str, str]] = []
        for key, value in pairs:
            if key != "sid":
                rewritten_pairs.append((key, value))
                continue
            if value == _PLACEHOLDER_SID and path not in {"/post", "/state"}:
                value = self._active_private_sid(endpoint)
            elif value != _PLACEHOLDER_SID:
                value = self._private_sid(endpoint, value or "default")
            rewritten_pairs.append((key, value))
        if path.startswith(f"/files/{_PLACEHOLDER_SID}/"):
            path = path.replace(
                f"/files/{_PLACEHOLDER_SID}/",
                f"/files/{self._active_private_sid(endpoint)}/",
                1,
            )
        return urlunsplit(("", "", path, urlencode(rewritten_pairs), ""))

    def _proxy(
        self,
        *,
        handler: BaseHTTPRequestHandler,
        endpoint: EndpointName,
        target: str,
        body: bytes,
        inject_admin: bool,
    ) -> _GatewayResponse:
        backend = urlsplit(self.config.hub.backend_urls[endpoint])
        connection_options = {
            option.strip().lower()
            for value in handler.headers.get_all("Connection", [])
            for option in value.split(",")
            if option.strip()
        }
        stripped_headers = (
            _HOP_BY_HOP_HEADERS
            | connection_options
            | {
                _ADMIN_HEADER,
                "accept-encoding",
                "authorization",
                "content-length",
                "host",
            }
        )
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in stripped_headers
        }
        headers["Host"] = backend.netloc
        headers["Accept-Encoding"] = "identity"
        headers["Content-Length"] = str(len(body))
        if inject_admin:
            headers["X-CUA-Admin-Token"] = self.config.hub.admin_token
        connection = http.client.HTTPConnection(
            backend.hostname,
            backend.port,
            timeout=self.config.hub.request_timeout_s,
        )
        try:
            connection.request(handler.command, target, body=body, headers=headers)
            backend_response = connection.getresponse()
            response_body = backend_response.read(_MAX_RESPONSE_BYTES + 1)
            if len(response_body) > _MAX_RESPONSE_BYTES:
                raise RuntimeError("Hub backend response exceeds the size limit")
            raw_response_headers = tuple(backend_response.getheaders())
            response_connection_options = {
                option.strip().lower()
                for name, value in raw_response_headers
                if name.lower() == "connection"
                for option in value.split(",")
                if option.strip()
            }
            response_headers = tuple(
                (name, value)
                for name, value in raw_response_headers
                if name.lower()
                not in _HOP_BY_HOP_HEADERS
                | response_connection_options
                | {"content-length", _ADMIN_HEADER, "authorization"}
            )
            content_encodings = [
                value
                for name, value in response_headers
                if name.lower() == "content-encoding"
            ]
            if any(value.lower() != "identity" for value in content_encodings):
                raise RuntimeError("Hub backend returned an encoded response")
            admin_token = self.config.hub.admin_token
            if (
                admin_token.encode("ascii") in response_body
                or (
                    backend_response.reason is not None
                    and admin_token in backend_response.reason
                )
                or any(admin_token in value for _, value in response_headers)
            ):
                raise RuntimeError("Hub backend exposed the admin token")
            return _GatewayResponse(
                status=backend_response.status,
                reason=backend_response.reason,
                headers=response_headers,
                body=response_body,
            )
        finally:
            connection.close()

    def _capture_launch(
        self,
        endpoint: EndpointName,
        raw_sid: str | None,
        response: _GatewayResponse,
    ) -> None:
        if raw_sid is None or raw_sid == _PLACEHOLDER_SID or response.status != 200:
            return
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or "launch_url" not in payload:
            return
        launch_url = payload["launch_url"]
        if not isinstance(launch_url, str):
            raise TypeError("Hub backend returned an invalid launch URL")
        token = self._launch_token(launch_url)
        if token is None:
            raise RuntimeError("Hub backend returned an invalid launch token")
        with self._lock:
            selected_raw_sid = self._selected_raw_sids.get(endpoint)
            if selected_raw_sid not in {None, raw_sid}:
                raise RuntimeError("Hub backend launched a conflicting session")
            sid_key = (endpoint, raw_sid)
            token_key = (endpoint, token)
            previous = self._pending_launches_by_sid.get(sid_key)
            target = (
                previous.target if previous is not None else f"/?sid={_PLACEHOLDER_SID}"
            )
            if previous is not None:
                self._remove_pending_launch_locked(previous)
            token_collision = self._pending_launches_by_token.get(token_key)
            if token_collision is not None:
                self._remove_pending_launch_locked(token_collision)
            launch = _PendingLaunch(
                endpoint=endpoint,
                raw_sid=raw_sid,
                launch_url=launch_url,
                token=token,
                target=target,
            )
            self._pending_launches_by_sid[sid_key] = launch
            self._pending_launches_by_token[token_key] = launch

    def _rewrite_launch_response(
        self,
        launch: _PendingLaunch,
        response: _GatewayResponse,
    ) -> _GatewayResponse:
        if response.status not in {HTTPStatus.FOUND, HTTPStatus.SEE_OTHER}:
            raise RuntimeError("Hub session exchange did not redirect")
        location_count = sum(name.lower() == "location" for name, _ in response.headers)
        if location_count != 1:
            raise RuntimeError("Hub session exchange returned an invalid redirect")
        with self._condition:
            self._completed_browser_sessions += 1
            self._condition.notify_all()
        headers = tuple(
            (name, launch.target if name.lower() == "location" else value)
            for name, value in response.headers
        )
        return _GatewayResponse(
            response.status, response.reason, headers, response.body
        )

    def _private_sid(self, endpoint: EndpointName, raw_sid: str) -> str:
        key = (endpoint, raw_sid)
        with self._lock:
            existing = self._private_sids.get(key)
            if existing is not None:
                return existing
            message = f"{self.config.episode_id}\0{endpoint}\0{raw_sid}".encode()
            private_sid = hmac.new(
                self.config.namespace_key, message, hashlib.sha256
            ).hexdigest()
            self._private_sids[key] = private_sid
            return private_sid

    def _redact_private_sids(
        self,
        endpoint: EndpointName,
        response: _GatewayResponse,
    ) -> _GatewayResponse:
        with self._lock:
            private_sids = {
                private_sid
                for (candidate_endpoint, _), private_sid in self._private_sids.items()
                if candidate_endpoint == endpoint
            }
        body = response.body
        headers = response.headers
        content_types = [
            value.lower() for name, value in headers if name.lower() == "content-type"
        ]
        textual = len(content_types) == 1 and (
            content_types[0].startswith("text/")
            or any(
                marker in content_types[0] for marker in ("json", "javascript", "xml")
            )
        )
        for private_sid in private_sids:
            private_bytes = private_sid.encode("ascii")
            if private_bytes in body:
                if not textual:
                    raise RuntimeError(
                        "Hub backend exposed a private SID in opaque data"
                    )
                body = body.replace(private_bytes, _PLACEHOLDER_SID.encode("ascii"))
            headers = tuple(
                (name, value.replace(private_sid, _PLACEHOLDER_SID))
                for name, value in headers
            )
        if body == response.body and headers == response.headers:
            return response
        return _GatewayResponse(response.status, response.reason, headers, body)

    def _active_private_sid(self, endpoint: EndpointName) -> str:
        with self._lock:
            selected_raw_sid = self._selected_raw_sids.get(endpoint)
            private_sid = (
                None
                if selected_raw_sid is None
                else self._private_sids.get((endpoint, selected_raw_sid))
            )
        if selected_raw_sid is None or private_sid is None:
            raise RuntimeError(
                f"selected session is not initialized for endpoint {endpoint}"
            )
        return private_sid

    def _bootstrap_raw_navigation(
        self,
        endpoint: EndpointName,
        raw_sid: str,
        target: SplitResult,
    ) -> _GatewayResponse | None:
        with self._lock:
            selected_raw_sid = self._selected_raw_sids.get(endpoint)
            if selected_raw_sid not in {None, raw_sid}:
                return self._error(
                    HTTPStatus.CONFLICT,
                    f"endpoint {endpoint} is already bound to another session",
                )
            launch = self._pending_launches_by_sid.get((endpoint, raw_sid))
            if launch is None:
                return None
            self._select_launch_locked(launch)
            launch.target = self._placeholder_target(target)
            return self._launch_redirect(launch)

    def _bootstrap_placeholder_navigation(
        self,
        endpoint: EndpointName,
        target: SplitResult,
    ) -> _GatewayResponse | None:
        with self._lock:
            selected_raw_sid = self._selected_raw_sids.get(endpoint)
            if selected_raw_sid is not None:
                launch = self._pending_launches_by_sid.get((endpoint, selected_raw_sid))
                if launch is None:
                    return None
            else:
                candidates = [
                    launch
                    for (
                        candidate_endpoint,
                        _,
                    ), launch in self._pending_launches_by_sid.items()
                    if candidate_endpoint == endpoint
                ]
                selected_elsewhere = {
                    raw_sid
                    for candidate_endpoint, raw_sid in self._selected_raw_sids.items()
                    if candidate_endpoint != endpoint
                }
                matching = [
                    launch
                    for launch in candidates
                    if launch.raw_sid in selected_elsewhere
                ]
                if len(matching) == 1:
                    launch = matching[0]
                elif len(candidates) == 1:
                    launch = candidates[0]
                else:
                    return self._error(
                        HTTPStatus.CONFLICT,
                        f"cannot choose a session for endpoint {endpoint}",
                    )
            self._select_launch_locked(launch)
            launch.target = self._placeholder_target(target)
            return self._launch_redirect(launch)

    def _claim_launch(
        self,
        endpoint: EndpointName,
        token: str,
    ) -> _PendingLaunch | None:
        with self._lock:
            launch = self._pending_launches_by_token.get((endpoint, token))
            if launch is None:
                return None
            selected_raw_sid = self._selected_raw_sids.get(endpoint)
            if selected_raw_sid not in {None, launch.raw_sid}:
                self._remove_pending_launch_locked(launch)
                return None
            self._select_launch_locked(launch)
            self._remove_pending_launch_locked(launch)
            return launch

    def _select_launch_locked(self, launch: _PendingLaunch) -> None:
        self._selected_raw_sids[launch.endpoint] = launch.raw_sid
        siblings = [
            sibling
            for (endpoint, _), sibling in self._pending_launches_by_sid.items()
            if endpoint == launch.endpoint and sibling is not launch
        ]
        for sibling in siblings:
            self._remove_pending_launch_locked(sibling)

    def _remove_pending_launch_locked(self, launch: _PendingLaunch) -> None:
        sid_key = (launch.endpoint, launch.raw_sid)
        if self._pending_launches_by_sid.get(sid_key) is launch:
            self._pending_launches_by_sid.pop(sid_key)
        token_key = (launch.endpoint, launch.token)
        if self._pending_launches_by_token.get(token_key) is launch:
            self._pending_launches_by_token.pop(token_key)

    @staticmethod
    def _launch_redirect(launch: _PendingLaunch) -> _GatewayResponse:
        return _GatewayResponse(
            status=HTTPStatus.FOUND,
            reason=None,
            headers=(("Location", launch.launch_url), ("Cache-Control", "no-store")),
            body=b"",
        )

    def _endpoint_for_host(self, host_header: str) -> EndpointName | None:
        if not host_header.isascii():
            return None
        for endpoint in self.config.endpoints:
            expected = f"{self.gateway_hostnames[endpoint]}:{self.config.port}"
            if hmac.compare_digest(host_header.lower(), expected):
                return endpoint
        return None

    @staticmethod
    def _content_length(handler: BaseHTTPRequestHandler) -> int:
        if handler.headers.get_all("Transfer-Encoding", []):
            raise ValueError("chunked requests are unsupported")
        raw_lengths = handler.headers.get_all("Content-Length", [])
        if not raw_lengths:
            return 0
        if len(raw_lengths) != 1:
            raise ValueError("requests must contain at most one content length")
        raw_length = raw_lengths[0]
        if not raw_length or any(
            character not in "0123456789" for character in raw_length
        ):
            raise ValueError("invalid request content length")
        length = int(raw_length)
        if not 0 <= length <= _MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        return length

    @staticmethod
    def _query_sid(query: str) -> str | None:
        values = [
            value
            for key, value in parse_qsl(query, keep_blank_values=True)
            if key == "sid"
        ]
        if not values:
            return None
        if len(values) != 1:
            raise ValueError("requests must contain at most one sid")
        if not values[0]:
            raise ValueError("session IDs must not be empty")
        return values[0]

    @staticmethod
    def _is_navigation(handler: BaseHTTPRequestHandler) -> bool:
        if handler.command != "GET":
            return False
        fetch_navigation = handler.headers.get(
            "Sec-Fetch-Mode", ""
        ).lower() == "navigate" and handler.headers.get(
            "Sec-Fetch-Dest", ""
        ).lower() in {"", "document"}
        accepts_html = "text/html" in handler.headers.get("Accept", "").lower()
        return fetch_navigation or accepts_html

    @staticmethod
    def _launch_token(url: str) -> str | None:
        split = urlsplit(url)
        if (
            split.scheme
            or split.netloc
            or split.path != "/_cua_session"
            or split.fragment
        ):
            return None
        pairs = parse_qsl(split.query, keep_blank_values=True)
        if len(pairs) != 1 or pairs[0][0] != "token" or len(pairs[0][1]) != 64:
            return None
        token = pairs[0][1]
        if any(character not in "0123456789abcdef" for character in token):
            return None
        return token

    @staticmethod
    def _placeholder_target(split: SplitResult) -> str:
        pairs = [
            (key, _PLACEHOLDER_SID if key == "sid" else value)
            for key, value in parse_qsl(split.query, keep_blank_values=True)
        ]
        if not any(key == "sid" for key, _ in pairs):
            pairs.append(("sid", _PLACEHOLDER_SID))
        return urlunsplit(("", "", split.path or "/", urlencode(pairs), ""))

    @staticmethod
    def _error(status: HTTPStatus, message: str) -> _GatewayResponse:
        body = json.dumps({"error": message}).encode("utf-8")
        return _GatewayResponse(
            status=status,
            reason=None,
            headers=(
                ("Content-Type", "application/json"),
                ("Cache-Control", "no-store"),
            ),
            body=body,
        )
