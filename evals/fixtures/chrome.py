"""Chrome fixture launch and browser diagnostics.

This module starts Chromium in the guest on a fixture URL, waits for the page to
report ready, and reads the live DOM through CDP. The session forwards a port and
reaps process groups; it does not know Chrome exists.

`launch` retries under one absolute deadline rather than per-attempt timeouts.
Per-attempt timeouts compose into an unbounded total, which is how a "30 second"
readiness wait becomes four minutes. A relaunch is arm-neutral browser setup, not an
action retry: nothing about the model's turn is repeated.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from evals.fixtures.cdp import cdp_evaluate, find_page_target, local_websocket_url
from evals.fixtures.web import WebFixture, WebFixtureServer

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "BROWSER_RESTART_BUDGET_S",
    "MAX_ATTEMPTS",
    "SETUP_DEADLINE_S",
    "ChromeFixtureError",
    "ChromeLaunch",
    "capture_browser_diagnostics",
    "capture_chrome_log",
    "launch",
]

SETUP_DEADLINE_S = 120.0
MAX_ATTEMPTS = 3
BROWSER_RESTART_BUDGET_S = 20.0
"""Refuse a further attempt when less than this remains: a relaunch that cannot
finish inside the deadline only turns a clean timeout into a torn-down browser."""

CHROME_LOG = "/tmp/fixture_chrome.log"


class ChromeFixtureError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChromeLaunch:
    """Launch evidence, as data. Nothing here gates anything."""

    status: str
    attempts: list[dict[str, Any]]
    generation: int | None
    url: str
    ready_state: dict[str, Any] | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "generation": self.generation,
            "url": self.url,
            "ready_state": self.ready_state,
            "error": self.error,
        }


def _launch_script(url: str, *, restart: bool, attempt: int) -> str:
    restart_script = (
        """
pkill -TERM -f '([c]hrome|[c]hromium)' || true
for _wait in $(seq 1 20); do
  if ! pgrep -f '([c]hrome|[c]hromium)' >/dev/null; then break; fi
  sleep 0.1
done
pkill -KILL -f '([c]hrome|[c]hromium)' || true
""".strip()
        if restart
        else ""
    )
    log_setup = (
        f": >{CHROME_LOG}"
        if attempt == 1
        else f"printf '\\n--- readiness attempt {attempt} ---\\n' >>{CHROME_LOG}"
    )
    return f"""
set -euo pipefail
{restart_script}
{log_setup}
browser="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
test -n "$browser"
nohup "$browser" --no-first-run --no-default-browser-check \
  --disable-session-crashed-bubble --disable-features=TranslateUI \
  --start-maximized {url!r} >>{CHROME_LOG} 2>&1 </dev/null &
""".strip()


def launch(
    session: Any,
    server: WebFixtureServer,
    fixture: WebFixture,
    *,
    timeout_s: float = SETUP_DEADLINE_S,
) -> ChromeLaunch:
    """Bring the fixture page to ready inside one absolute deadline.

    Only a timeout or a transport failure may relaunch Chromium. A page that loads
    and reports a wrong state is a deterministic failure and stays terminal.
    """
    if timeout_s <= 0:
        raise ChromeFixtureError("fixture setup deadline must be positive")
    deadline = time.monotonic() + float(timeout_s)
    url = server.guest_url(fixture)
    attempts: list[dict[str, Any]] = []
    generation: int | None = None
    last_error: Exception | None = None

    for attempt_index in range(1, MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (attempt_index > 1 and remaining <= BROWSER_RESTART_BUDGET_S):
            break
        attempt: dict[str, Any] = {
            "attempt_index": attempt_index,
            "restart": attempt_index > 1,
            "remaining_s_at_start": round(remaining, 3),
            "status": "running",
            "error": None,
        }
        attempts.append(attempt)
        try:
            generation = server.store.reset(fixture)
            attempt["generation"] = generation
            session.execute_argv(
                [
                    "bash",
                    "-lc",
                    _launch_script(url, restart=attempt_index > 1, attempt=attempt_index),
                ]
            )
            wait_s = max(1.0, deadline - time.monotonic())
            attempt["wait_timeout_s"] = round(wait_s, 3)
            ready = server.store.wait_ready(fixture.id, timeout_s=wait_s)
            attempt["status"] = "ready"
            return ChromeLaunch(
                status="ready",
                attempts=attempts,
                generation=generation,
                url=url,
                ready_state=ready,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then retried or reported
            last_error = exc
            attempt["status"] = "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            _LOGGER.warning("chrome fixture attempt %d failed: %r", attempt_index, exc)

    return ChromeLaunch(
        status="timeout" if last_error is None else "failed",
        attempts=attempts,
        generation=generation,
        url=url,
        ready_state=None,
        error=None if last_error is None else f"{type(last_error).__name__}: {last_error}",
    )


_PAGE_EXPRESSION = r"""
(() => {
  const elementState = (element) => element ? {
    id: element.id || '',
    tag: element.tagName ? element.tagName.toLowerCase() : '',
    checked: typeof element.checked === 'boolean' ? element.checked : null,
    value: 'value' in element ? String(element.value) : null,
    disabled: Boolean(element.disabled),
    outer_html: element.outerHTML
  } : null;
  return {
    schema_version: 1,
    captured_browser_wall_time_ms: Date.now(),
    performance_time_origin_ms: performance.timeOrigin,
    performance_now_ms: performance.now(),
    url: location.href,
    title: document.title,
    ready_state: document.readyState,
    visibility_state: document.visibilityState,
    has_focus: document.hasFocus(),
    diagnostics: window.__FIXTURE_DIAGNOSTICS__
      ? JSON.parse(JSON.stringify(window.__FIXTURE_DIAGNOSTICS__)) : null,
    dom: {
      active_element: elementState(document.activeElement),
      target: elementState(document.getElementById('target')),
      decoy: elementState(document.getElementById('decoy')),
      scroll_x: Math.round(window.scrollX),
      scroll_y: Math.round(window.scrollY),
      body_text: document.body ? document.body.innerText : null
    }
  };
})()
""".strip()


def capture_browser_diagnostics(
    fixture: WebFixture, chromium_port: int, *, timeout_s: float = 2.0
) -> dict[str, Any]:
    """Read the live fixture page through CDP without modifying page state.

    This is the second, independent read of state the page also posts over HTTP. A
    fixture whose DOM and posted state disagree is broken, and having both is what
    makes that detectable.
    """
    target = find_page_target(chromium_port, fixture.id, timeout_s=timeout_s)
    websocket_url = local_websocket_url(target, chromium_port)
    page = cdp_evaluate(websocket_url, _PAGE_EXPRESSION, timeout_s=timeout_s)
    if not isinstance(page, dict):
        raise ChromeFixtureError("CDP page diagnostic result was not an object")
    return {
        "schema_version": 1,
        "status": "captured",
        "transport": "cdp_runtime_evaluate",
        "host_forwarded_port": chromium_port,
        "target": {
            "id": target.get("id"),
            "type": target.get("type"),
            "url": target.get("url"),
            "title": target.get("title"),
            "advertised_websocket_url": target.get("webSocketDebuggerUrl"),
            "local_websocket_url": websocket_url,
        },
        "page": page,
    }


def capture_chrome_log(session: Any, *, limit_bytes: int = 64 * 1024) -> dict[str, Any]:
    """The tail of the guest Chrome log. Diagnostics only; never raises."""
    try:
        output = session.execute_argv(
            ["bash", "-lc", f"tail -c {int(limit_bytes)} {CHROME_LOG} 2>/dev/null || true"]
        )
        return {"status": "captured", "log": output.get("output", "")}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
