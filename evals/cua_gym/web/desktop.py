from __future__ import annotations

import ipaddress
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

_CHROME_DEBUGGING_PORT = 9222
_CHROME_LAUNCH_PATH = "/tmp/cua_gym_launch_chrome.py"
_CHROME_LOG_PATH = "/tmp/cua_gym_chrome.log"
_MAX_CHROME_LOG_BYTES = 65_536
_CUA_HOSTNAME_RE = re.compile(
    r"[a-z2-7]{52}\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.cua\.internal"
)
_BROWSER_IDENTITY_PATH_RE = re.compile(r"/devtools/browser/[A-Za-z0-9._-]+")
_MAX_VERSION_RESPONSE_BYTES = 1_048_576

_CONFIGURE_CUA_HOSTS_SOURCE = """\
import sys
from pathlib import Path

path = Path("/etc/hosts")
start = "# BEGIN CUA-GYM MANAGED HOSTS"
end = "# END CUA-GYM MANAGED HOSTS"
lines = path.read_text(encoding="utf-8").splitlines()
if (
    lines.count(start) != lines.count(end)
    or lines.count(start) > 1
    or (start in lines and lines.index(start) > lines.index(end))
):
    raise RuntimeError("invalid CUA-Gym managed-host block")
result = []
inside = False
for line in lines:
    if line == start:
        inside = True
        continue
    if line == end:
        inside = False
        continue
    if not inside:
        result.append(line)
entries = [f"{sys.argv[1]} {hostname}" for hostname in sys.argv[2:]]
path.write_text("\\n".join([*result, start, *entries, end]) + "\\n", encoding="utf-8")
"""

_CHROME_LAUNCH_SOURCE = f"""\
import selectors
import socket
import socketserver
import subprocess
import sys
from pathlib import Path


class Forwarder(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ForwardingHandler(socketserver.BaseRequestHandler):
    def handle(self):
        upstream = socket.create_connection(("127.0.0.1", {_CHROME_DEBUGGING_PORT}))
        selector = selectors.DefaultSelector()
        try:
            selector.register(self.request, selectors.EVENT_READ, upstream)
            selector.register(upstream, selectors.EVENT_READ, self.request)
            while True:
                for key, _events in selector.select():
                    chunk = key.fileobj.recv(65536)
                    if not chunk:
                        return
                    key.data.sendall(chunk)
        finally:
            selector.close()
            upstream.close()


def guest_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.0.2.2", 9))
        return probe.getsockname()[0]
    finally:
        probe.close()


if sys.argv[1:] == ["forward"]:
    with Forwarder((guest_ip(), {_CHROME_DEBUGGING_PORT}), ForwardingHandler) as server:
        server.serve_forever()
    raise SystemExit(0)


log_path = Path({_CHROME_LOG_PATH!r})
with log_path.open("ab", buffering=0) as log:
    subprocess.Popen(
        [sys.executable, __file__, "forward"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    subprocess.Popen(
        [
            "google-chrome",
            "--remote-debugging-port={_CHROME_DEBUGGING_PORT}",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
"""


class GuestCommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class GuestClient(Protocol):
    def write_file(self, path: str, content: bytes) -> None: ...

    def execute(
        self,
        argv: list[str],
        *,
        check: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    def execute_detached(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> GuestCommandResult: ...


class BrowserUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CuaGymDesktopConfig:
    browser_debugging_url: str
    guest_gateway_host: str
    guest_hostnames: tuple[str, ...]
    guest_password: str = field(repr=False)
    browser_ready_timeout_s: float

    def __post_init__(self) -> None:
        _validate_debugging_url(self.browser_debugging_url)
        try:
            gateway = ipaddress.ip_address(self.guest_gateway_host)
        except ValueError as error:
            raise ValueError("guest_gateway_host must be an IPv4 address") from error
        if gateway.version != 4 or str(gateway) != self.guest_gateway_host:
            raise ValueError("guest_gateway_host must be a normalized IPv4 address")
        if (
            not self.guest_hostnames
            or tuple(sorted(self.guest_hostnames)) != self.guest_hostnames
            or len(set(self.guest_hostnames)) != len(self.guest_hostnames)
            or any(
                not isinstance(hostname, str)
                or _CUA_HOSTNAME_RE.fullmatch(hostname) is None
                for hostname in self.guest_hostnames
            )
        ):
            raise ValueError(
                "guest_hostnames must be sorted, unique authenticated CUA hosts"
            )
        if (
            not isinstance(self.guest_password, str)
            or not self.guest_password
            or any(character in self.guest_password for character in "\0\r\n")
        ):
            raise ValueError("guest_password must be non-empty and single-line")
        timeout = self.browser_ready_timeout_s
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("browser_ready_timeout_s must be positive")
        object.__setattr__(self, "browser_ready_timeout_s", float(timeout))


class CuaGymDesktopBrowser:
    def __init__(self, *, guest: GuestClient, config: CuaGymDesktopConfig) -> None:
        self.guest = guest
        self.config = config

    def configure_guest_hosts(self) -> None:
        result = self.guest.execute(
            [
                "bash",
                "-c",
                'printf "%s\\n" "$1" | sudo --stdin --prompt= -- "${@:2}"',
                "cua-gym-sudo",
                self.config.guest_password,
                "python3",
                "-c",
                _CONFIGURE_CUA_HOSTS_SOURCE,
                self.config.guest_gateway_host,
                *self.config.guest_hostnames,
            ],
            check=False,
            timeout_s=self.config.browser_ready_timeout_s,
        )
        _require_execute_result(result, allowed_returncodes={0}, operation="host setup")

    def browser_identity(self) -> str:
        url = self.config.browser_debugging_url + "/json/version"
        try:
            with urlopen(url, timeout=self.config.browser_ready_timeout_s) as response:
                if response.headers.get_content_type() != "application/json":
                    raise RuntimeError("Chrome /json/version did not return JSON")
                body = response.read(_MAX_VERSION_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise BrowserUnavailable(
                f"Chrome debugging endpoint is unavailable: {error}"
            ) from error
        if len(body) > _MAX_VERSION_RESPONSE_BYTES:
            raise RuntimeError("Chrome /json/version response is too large")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Chrome /json/version returned invalid JSON") from error
        value = (
            payload.get("webSocketDebuggerUrl") if isinstance(payload, dict) else None
        )
        return _validate_browser_identity(value)

    def ensure_browser(self) -> str:
        try:
            return self.browser_identity()
        except BrowserUnavailable:
            pass

        stopped = self.guest.execute(
            [
                "pkill",
                "--signal",
                "TERM",
                "--full",
                "/opt/google/[c]hrome/[c]hrome|google-[c]hrome|[c]hromium",
            ],
            check=False,
            timeout_s=self.config.browser_ready_timeout_s,
        )
        _require_execute_result(
            stopped, allowed_returncodes={0, 1}, operation="stale Chrome cleanup"
        )
        self.guest.write_file(
            _CHROME_LAUNCH_PATH, _CHROME_LAUNCH_SOURCE.encode("utf-8")
        )
        launched = self.guest.execute_detached(
            ["python3", _CHROME_LAUNCH_PATH],
            timeout_s=self.config.browser_ready_timeout_s,
        )
        if (
            type(launched.returncode) is not int
            or not isinstance(launched.stdout, str)
            or not isinstance(launched.stderr, str)
        ):
            raise RuntimeError("Chrome launcher returned an invalid result")
        if launched.returncode != 0:
            raise RuntimeError(
                f"Chrome launcher failed with {launched.returncode}: "
                f"{launched.stderr.strip()}"
            )

        deadline = time.monotonic() + self.config.browser_ready_timeout_s
        last_error: BrowserUnavailable | None = None
        while time.monotonic() < deadline:
            try:
                return self.browser_identity()
            except BrowserUnavailable as error:
                last_error = error
                time.sleep(0.25)
        readiness_error = TimeoutError(f"Chrome did not become ready: {last_error!r}")
        log = self.guest.execute(
            ["tail", "-c", str(_MAX_CHROME_LOG_BYTES), _CHROME_LOG_PATH],
            check=False,
            timeout_s=self.config.browser_ready_timeout_s,
        )
        if not isinstance(log, dict):
            raise TypeError(
                "Chrome log retrieval returned a non-object"
            ) from readiness_error
        returncode = log.get("returncode")
        output = log.get("output")
        if (
            log.get("status") != "success"
            or type(returncode) is not int
            or returncode != 0
            or not isinstance(output, str)
            or len(output.encode("utf-8")) > _MAX_CHROME_LOG_BYTES
        ):
            raise RuntimeError(
                "Chrome log retrieval returned an invalid result"
            ) from readiness_error
        detail = output.strip() or "Chrome log is empty"
        raise TimeoutError(
            f"{readiness_error}; guest Chrome log:\n{detail}"
        ) from readiness_error

    def verify_after_setup(self, expected_identity: str) -> None:
        expected = _validate_browser_identity(expected_identity)
        if self.browser_identity() != expected:
            raise RuntimeError("CUA-Gym setup replaced the reusable Chrome process")

    def cleanup_browser(
        self,
        *,
        origins: tuple[str, ...],
        expected_identity: str,
    ) -> None:
        normalized_origins = _validate_origins(origins)
        expected = _validate_browser_identity(expected_identity)
        _cleanup_over_cdp(
            self.config.browser_debugging_url,
            normalized_origins,
            self.config.browser_ready_timeout_s,
        )
        if self.browser_identity() != expected:
            raise RuntimeError("Browser cleanup replaced the reusable Chrome process")


def _require_execute_result(
    result: object,
    *,
    allowed_returncodes: set[int],
    operation: str,
) -> None:
    if not isinstance(result, dict):
        raise TypeError(f"{operation} returned a non-object")
    returncode = result.get("returncode")
    if result.get("status") != "success" or type(returncode) is not int:
        raise RuntimeError(f"{operation} returned an invalid result: {result!r}")
    if returncode not in allowed_returncodes:
        detail = result.get("error")
        raise RuntimeError(f"{operation} failed with {returncode}: {detail!r}")


def _validate_debugging_url(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("browser_debugging_url must be a loopback HTTP URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("browser_debugging_url must be a loopback HTTP URL") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.netloc != f"127.0.0.1:{port}"
    ):
        raise ValueError("browser_debugging_url must be http://127.0.0.1:<port>")
    return value


def _validate_browser_identity(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("browser identity must be a Chrome websocket URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ValueError("browser identity must be a Chrome websocket URL") from error
    if (
        parsed.scheme != "ws"
        or not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or _BROWSER_IDENTITY_PATH_RE.fullmatch(parsed.path) is None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("browser identity must be a Chrome browser websocket URL")
    return value


def _validate_origins(origins: object) -> tuple[str, ...]:
    if (
        not isinstance(origins, tuple)
        or not origins
        or tuple(sorted(origins)) != origins
        or len(set(origins)) != len(origins)
    ):
        raise ValueError("origins must be a sorted, unique tuple")
    for origin in origins:
        if not isinstance(origin, str):
            raise TypeError("origins must contain CUA-Gym HTTP origins")
        try:
            parsed = urlsplit(origin)
            port = parsed.port
        except ValueError as error:
            raise ValueError("origins must contain CUA-Gym HTTP origins") from error
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or _CUA_HOSTNAME_RE.fullmatch(parsed.hostname) is None
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.netloc != f"{parsed.hostname}:{port}"
        ):
            raise ValueError("origins must be authenticated CUA-Gym HTTP origins")
    return origins


def _cleanup_over_cdp(
    debugging_url: str,
    origins: tuple[str, ...],
    timeout_s: float,
) -> None:
    from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

    timeout_ms = timeout_s * 1000
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(
            debugging_url,
            timeout=timeout_ms,
            no_defaults=True,
        )
        try:
            if len(browser.contexts) != 1:
                raise RuntimeError("Chrome must have exactly one browser context")
            context = browser.contexts[0]
            keep_page = context.new_page()
            session = context.new_cdp_session(keep_page)
            for origin in origins:
                session.send(
                    "Storage.clearDataForOrigin",
                    {"origin": origin, "storageTypes": "all"},
                )
            session.send("Network.clearBrowserCache")
            context.clear_cookies()
            for page in tuple(context.pages):
                if page is not keep_page:
                    page.close()
            keep_page.goto("about:blank", timeout=timeout_ms)
        finally:
            browser.close()
