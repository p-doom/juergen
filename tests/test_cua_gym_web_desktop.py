from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from evals.cua_gym.web import desktop
from evals.cua_gym.web.desktop import (
    BrowserUnavailable,
    CuaGymDesktopBrowser,
    CuaGymDesktopConfig,
)

_EPISODE_LABEL = "a" * 52
_DOCS_HOST = f"{_EPISODE_LABEL}.docs.cua.internal"
_GMAIL_HOST = f"{_EPISODE_LABEL}.gmail.cua.internal"

_IDENTITY = "ws://localhost:9222/devtools/browser/browser-1"


@dataclass(frozen=True)
class _Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class _Guest:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.execute_result: dict[str, Any] = {
            "status": "success",
            "returncode": 1,
            "output": "",
            "error": "",
        }
        self.execute_results: list[dict[str, Any]] = []
        self.secret_error: BaseException | None = None
        self.detached_result = _Result(0)
        self.on_detached = lambda: None

    def write_file(self, path: str, content: bytes) -> None:
        self.calls.append(("write_file", path, content))

    def execute(
        self,
        argv: list[str],
        *,
        check: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("execute", argv, check, timeout_s))
        result = (
            self.execute_results.pop(0) if self.execute_results else self.execute_result
        )
        return dict(result)

    def execute_with_secret_stdin(
        self,
        argv: Sequence[str],
        *,
        secret: bytes,
        timeout_s: float | None = None,
    ) -> None:
        self.calls.append(("execute_with_secret_stdin", tuple(argv), secret, timeout_s))
        if self.secret_error is not None:
            raise self.secret_error

    def execute_detached(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> _Result:
        self.calls.append(("execute_detached", tuple(argv), timeout_s, env))
        self.on_detached()
        return self.detached_result


class _VersionServer(ThreadingHTTPServer):
    identity: str | None = _IDENTITY
    malformed = False


class _VersionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/json/version":
            self.send_response(404)
            self.end_headers()
            return
        if self.server.identity is None:  # type: ignore[attr-defined]
            self.send_response(503)
            self.end_headers()
            return
        body = (
            b"not-json"
            if self.server.malformed  # type: ignore[attr-defined]
            else json.dumps(
                {"webSocketDebuggerUrl": self.server.identity}  # type: ignore[attr-defined]
            ).encode("utf-8")
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def version_server():
    server = _VersionServer(("127.0.0.1", 0), _VersionHandler)
    server.identity = _IDENTITY
    server.malformed = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _config(version_server: _VersionServer, **overrides: object) -> CuaGymDesktopConfig:
    values: dict[str, object] = {
        "browser_debugging_url": f"http://127.0.0.1:{version_server.server_port}",
        "guest_gateway_host": "10.0.2.2",
        "guest_hostnames": (_DOCS_HOST, _GMAIL_HOST),
        "guest_password": "password",
        "browser_ready_timeout_s": 1.0,
    }
    values.update(overrides)
    return CuaGymDesktopConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("browser_debugging_url", "http://localhost:9222", "127.0.0.1"),
        ("browser_debugging_url", "https://127.0.0.1:9222", "127.0.0.1"),
        ("guest_gateway_host", "10.0.2.02", "IPv4"),
        (
            "guest_hostnames",
            (_GMAIL_HOST, _DOCS_HOST),
            "sorted, unique",
        ),
        ("guest_hostnames", ("example.com",), "sorted, unique"),
        ("guest_password", "line\nbreak", "single-line"),
        ("browser_ready_timeout_s", 0.0, "positive"),
    ],
)
def test_configuration_rejects_invalid_inputs_before_io(
    version_server: _VersionServer, field: str, value: object, message: str
) -> None:
    guest = _Guest()
    with pytest.raises(ValueError, match=message):
        CuaGymDesktopBrowser(
            guest=guest,
            config=_config(version_server, **{field: value}),
        )
    assert guest.calls == []


def test_guest_hosts_use_one_fixed_privileged_operation(
    version_server: _VersionServer,
) -> None:
    guest = _Guest()
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))

    browser.configure_guest_hosts()

    assert len(guest.calls) == 1
    operation, argv, secret, timeout_s = guest.calls[0]
    assert operation == "execute_with_secret_stdin"
    assert secret == b"password\n"
    assert timeout_s == 1.0
    assert list(argv[:6]) == [
        "sudo",
        "--stdin",
        "--prompt=",
        "--",
        "python3",
        "-c",
    ]
    assert list(argv[-3:]) == [
        "10.0.2.2",
        _DOCS_HOST,
        _GMAIL_HOST,
    ]
    assert "password" not in argv


def test_guest_host_failure_is_not_ignored(version_server: _VersionServer) -> None:
    guest = _Guest()
    guest.secret_error = RuntimeError("sudo denied")
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))

    with pytest.raises(RuntimeError, match="sudo denied"):
        browser.configure_guest_hosts()


def test_browser_identity_reads_the_real_version_endpoint(
    version_server: _VersionServer,
) -> None:
    browser = CuaGymDesktopBrowser(guest=_Guest(), config=_config(version_server))
    assert browser.browser_identity() == _IDENTITY


def test_browser_identity_distinguishes_absence_from_bad_contract(
    version_server: _VersionServer,
) -> None:
    browser = CuaGymDesktopBrowser(guest=_Guest(), config=_config(version_server))
    version_server.identity = None
    with pytest.raises(BrowserUnavailable, match="unavailable"):
        browser.browser_identity()

    version_server.identity = _IDENTITY
    version_server.malformed = True
    with pytest.raises(RuntimeError, match="invalid JSON"):
        browser.browser_identity()


def test_existing_browser_is_reused_without_guest_io(
    version_server: _VersionServer,
) -> None:
    guest = _Guest()
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))
    assert browser.ensure_browser() == _IDENTITY
    assert guest.calls == []


def test_absent_browser_is_launched_once(version_server: _VersionServer) -> None:
    guest = _Guest()
    version_server.identity = None
    guest.on_detached = lambda: setattr(version_server, "identity", _IDENTITY)
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))

    assert browser.ensure_browser() == _IDENTITY
    assert [call[0] for call in guest.calls] == [
        "execute",
        "write_file",
        "execute_detached",
    ]
    assert guest.calls[1][1] == "/tmp/cua_gym_launch_chrome.py"
    assert b"--remote-debugging-port=9222" in guest.calls[1][2]


def test_launch_failure_fails_before_polling(version_server: _VersionServer) -> None:
    guest = _Guest()
    version_server.identity = None
    guest.detached_result = _Result(7, stderr="chrome missing")
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))

    with pytest.raises(RuntimeError, match="launcher failed with 7: chrome missing"):
        browser.ensure_browser()
    assert [call[0] for call in guest.calls] == [
        "execute",
        "write_file",
        "execute_detached",
    ]


def test_readiness_timeout_includes_bounded_guest_log_tail(
    version_server: _VersionServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guest = _Guest()
    guest.execute_results = [
        dict(guest.execute_result),
        {
            "status": "success",
            "returncode": 0,
            "output": "Chrome failed to bind\n",
        },
    ]
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))

    def unavailable() -> str:
        raise BrowserUnavailable("still unavailable")

    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(browser, "browser_identity", unavailable)
    monkeypatch.setattr(desktop.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(desktop.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="guest Chrome log:\\nChrome failed to bind"):
        browser.ensure_browser()
    assert guest.calls[-1] == (
        "execute",
        ["tail", "-c", "65536", "/tmp/cua_gym_chrome.log"],
        False,
        1.0,
    )


@pytest.mark.parametrize(
    "log_result",
    [
        {"status": "success", "returncode": 0},
        {"status": "success", "returncode": 0, "output": None},
        {"status": "success", "returncode": True, "output": ""},
        {"status": "success", "returncode": 1, "output": ""},
        {"status": "failure", "returncode": 0, "output": ""},
        {"status": "success", "returncode": 0, "output": "x" * 65_537},
    ],
)
def test_readiness_timeout_rejects_malformed_log_result(
    version_server: _VersionServer,
    monkeypatch: pytest.MonkeyPatch,
    log_result: dict[str, Any],
) -> None:
    guest = _Guest()
    guest.execute_results = [dict(guest.execute_result), log_result]
    browser = CuaGymDesktopBrowser(guest=guest, config=_config(version_server))

    def unavailable() -> str:
        raise BrowserUnavailable("still unavailable")

    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(browser, "browser_identity", unavailable)
    monkeypatch.setattr(desktop.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(desktop.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="log retrieval returned an invalid result"):
        browser.ensure_browser()


def test_setup_identity_replacement_fails_loudly(
    version_server: _VersionServer,
) -> None:
    browser = CuaGymDesktopBrowser(guest=_Guest(), config=_config(version_server))
    version_server.identity = "ws://localhost:9222/devtools/browser/browser-2"
    with pytest.raises(RuntimeError, match="setup replaced"):
        browser.verify_after_setup(_IDENTITY)


def test_cleanup_validates_then_clears_and_preserves_identity(
    version_server: _VersionServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = CuaGymDesktopBrowser(guest=_Guest(), config=_config(version_server))
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        desktop,
        "_cleanup_over_cdp",
        lambda url, origins, timeout: calls.append((url, origins, timeout)),
    )
    origins = (
        f"http://{_DOCS_HOST}:45000",
        f"http://{_GMAIL_HOST}:45000",
    )

    browser.cleanup_browser(origins=origins, expected_identity=_IDENTITY)

    assert calls == [
        (
            f"http://127.0.0.1:{version_server.server_port}",
            origins,
            1.0,
        )
    ]


def test_cleanup_rejects_invalid_origin_before_browser_io(
    version_server: _VersionServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = CuaGymDesktopBrowser(guest=_Guest(), config=_config(version_server))
    calls: list[object] = []
    monkeypatch.setattr(
        desktop,
        "_cleanup_over_cdp",
        lambda *_args: calls.append(object()),
    )
    with pytest.raises(ValueError, match="authenticated"):
        browser.cleanup_browser(
            origins=(f"https://{_GMAIL_HOST}:45000",),
            expected_identity=_IDENTITY,
        )
    assert calls == []


def test_cleanup_identity_replacement_fails_loudly(
    version_server: _VersionServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = CuaGymDesktopBrowser(guest=_Guest(), config=_config(version_server))
    monkeypatch.setattr(
        desktop,
        "_cleanup_over_cdp",
        lambda *_args: setattr(
            version_server,
            "identity",
            "ws://localhost:9222/devtools/browser/browser-2",
        ),
    )
    with pytest.raises(RuntimeError, match="cleanup replaced"):
        browser.cleanup_browser(
            origins=(f"http://{_GMAIL_HOST}:45000",),
            expected_identity=_IDENTITY,
        )


def test_cdp_cleanup_clears_storage_cache_cookies_and_old_tabs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    class Page:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(("close-page", self.name))

        def goto(self, url: str, *, timeout: float) -> None:
            events.append(("goto", self.name, url, timeout))

    class Session:
        def send(self, method: str, payload: object = None) -> None:
            events.append(("send", method, payload))

    old_page = Page("old")
    keep_page = Page("keep")

    class Context:
        def __init__(self) -> None:
            self.pages = [old_page]

        def new_page(self) -> Page:
            self.pages.append(keep_page)
            return keep_page

        def new_cdp_session(self, page: Page) -> Session:
            assert page is keep_page
            return Session()

        def clear_cookies(self) -> None:
            events.append(("clear-cookies",))

    context = Context()

    class Browser:
        def __init__(self) -> None:
            self.contexts = [context]

        def close(self) -> None:
            events.append(("disconnect",))

    class Chromium:
        def connect_over_cdp(self, url: str, **kwargs: object) -> Browser:
            events.append(("connect", url, kwargs))
            return Browser()

    class Manager:
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(chromium=Chromium())

        def __exit__(self, *_exc: object) -> None:
            pass

    package = ModuleType("playwright")
    sync_api = ModuleType("playwright.sync_api")
    sync_api.sync_playwright = Manager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)

    desktop._cleanup_over_cdp(
        "http://127.0.0.1:9222",
        (f"http://{_GMAIL_HOST}:45000",),
        2.5,
    )

    assert events == [
        (
            "connect",
            "http://127.0.0.1:9222",
            {"timeout": 2500.0, "no_defaults": True},
        ),
        (
            "send",
            "Storage.clearDataForOrigin",
            {
                "origin": f"http://{_GMAIL_HOST}:45000",
                "storageTypes": "all",
            },
        ),
        ("send", "Network.clearBrowserCache", None),
        ("clear-cookies",),
        ("close-page", "old"),
        ("goto", "keep", "about:blank", 2500.0),
        ("disconnect",),
    ]
