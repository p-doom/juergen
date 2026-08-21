"""Browser task fixtures: the pages, and the host HTTP server that serves them.

A browser task needs a page whose DOM state an oracle can read; the session forwards
a port and knows nothing about checkboxes.

Four templates, each isolating one input primitive so a failure names a capability
rather than "the browser task failed":

  * `click`   — a target checkbox plus a decoy, so a click that lands one row off is
                distinguishable from a click that missed entirely;
  * `focus_type` — a text input pre-filled with known content, so typing without
                focusing first is visible in the state rather than merely absent;
  * `drag`    — a range slider, the one primitive that needs press-move-release to be
                ordered correctly;
  * `scroll`  — twelve tall alternating sections, so scroll distance is measurable and
                the direction is unambiguous.

Page state is posted back to this server, so the oracle reads it over HTTP instead of
screen-scraping. `cdp.py` is the second, independent read of the same state — a
fixture whose HTTP state and DOM disagree is broken, and having both is how you find
that out.

Event ordering is reported, never enforced: the page posts its events and a caller
that wants ordering evidence reads `state["events"]`. Nothing here refuses to run on
an unexpected ordering.
"""

from __future__ import annotations

import html
import json
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = [
    "TEMPLATES",
    "FixtureServerError",
    "FixtureStateStore",
    "WebFixture",
    "WebFixtureServer",
    "render_fixture_html",
]

TEMPLATES = ("click", "focus_type", "drag", "scroll")
EVENT_LIMIT = 512
DESIGN_WIDTH = 1920
DESIGN_HEIGHT = 1080


class FixtureServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebFixture:
    """One page instance. `params` are the template's knobs, in design pixels."""

    id: str
    template: str
    instruction: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if self.template not in TEMPLATES:
            raise FixtureServerError(f"unknown template {self.template!r}")


def _initial_state(fixture: WebFixture) -> dict[str, Any]:
    params = fixture.params
    if fixture.template == "click":
        return {"kind": "click", "checked": False, "decoy_checked": False}
    if fixture.template == "focus_type":
        return {"kind": "text", "text": str(params.get("initial_text", ""))}
    if fixture.template == "drag":
        return {"kind": "drag", "value": int(params.get("initial_value", 0))}
    return {"kind": "scroll", "scroll_y": int(params.get("initial_y", 0))}


@dataclass
class FixtureStateStore:
    """Per-fixture live state plus a bounded event log.

    `generation` increments on every `reset`, and the page carries its generation in
    every post. A post from a previous generation is dropped: without that, a page
    that was still unloading when the next episode reset would write its dying
    scroll position into the new episode's state.
    """

    fixtures: dict[str, WebFixture]
    _state: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self, fixture: WebFixture) -> int:
        with self._lock:
            previous = self._state.get(fixture.id, {})
            generation = int(previous.get("generation", 0)) + 1
            self._state[fixture.id] = {
                "generation": generation,
                "ready": False,
                "current": _initial_state(fixture),
                "events": [],
                "geometry": None,
                "last_pointer_buttons": 0,
                "reset_wall_time": time.time(),
            }
            return generation

    def snapshot(self, fixture_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._state.get(fixture_id)
            if state is None:
                raise FixtureServerError(f"fixture {fixture_id!r} was never reset")
            return json.loads(json.dumps(state))

    def apply(self, fixture_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            state = self._state.get(fixture_id)
            if state is None:
                raise FixtureServerError(f"fixture {fixture_id!r} was never reset")
            if int(payload.get("generation", -1)) != state["generation"]:
                return {"status": "stale_generation", "generation": state["generation"]}
            kind = str(payload.get("kind", ""))
            if kind == "ready":
                state["ready"] = True
                state["geometry"] = payload.get("geometry")
                if payload.get("value") is not None:
                    state["current"] = {**state["current"], "value": payload["value"]}
            else:
                state["current"] = {
                    k: v for k, v in payload.items() if k not in {"generation"}
                }
            if payload.get("pointer_buttons") is not None:
                state["last_pointer_buttons"] = int(payload["pointer_buttons"])
            events = state["events"]
            events.append({"received_wall_time": time.time(), **payload})
            if len(events) > EVENT_LIMIT:
                del events[: len(events) - EVENT_LIMIT]
            return {"status": "ok", "generation": state["generation"]}

    def wait_ready(self, fixture_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            state = self.snapshot(fixture_id)
            if state["ready"]:
                return state
            if time.monotonic() >= deadline:
                raise FixtureServerError(
                    f"fixture {fixture_id!r} did not report ready within {timeout_s}s"
                )
            time.sleep(0.1)


def _card_position(params: dict[str, Any], *, width: int, height: int) -> str:
    """Map sealed 1920x1080 design coordinates into the measured viewport.

    The CSS clamp keeps the whole card visible when Chrome's inner viewport is
    smaller than the qcow's desktop — otherwise a target scrolls off-screen and the
    task becomes unreachable rather than hard.
    """
    left_percent = 100.0 * int(params["left"]) / float(DESIGN_WIDTH)
    top_percent = 100.0 * int(params["top"]) / float(DESIGN_HEIGHT)
    return (
        f"left:clamp(24px,{left_percent:.6f}vw,calc(100vw - {width + 24}px));"
        f"top:clamp(104px,{top_percent:.6f}vh,calc(100vh - {height + 24}px))"
    )


_COMMON_JS = """
<script>
const GENERATION = %(generation)d;
const POST_URL = '/state/%(fixture_id)s';
function screenRect(el) {
  const r = el.getBoundingClientRect();
  return {left: Math.round(r.left), top: Math.round(r.top),
          width: Math.round(r.width), height: Math.round(r.height),
          center_x: Math.round(r.left + r.width / 2),
          center_y: Math.round(r.top + r.height / 2)};
}
function measuredGeometry(parts) {
  return {viewport: {width: window.innerWidth, height: window.innerHeight,
                     device_pixel_ratio: window.devicePixelRatio},
          screen: {width: window.screen.width, height: window.screen.height},
          parts: parts};
}
function post(payload) {
  payload.generation = GENERATION;
  payload.client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
  navigator.sendBeacon
    ? navigator.sendBeacon(POST_URL, new Blob([JSON.stringify(payload)],
        {type: 'application/json'}))
    : fetch(POST_URL, {method: 'POST', body: JSON.stringify(payload), keepalive: true});
}
window.__FIXTURE_DIAGNOSTICS__ = {generation: GENERATION, events: []};
for (const name of ['mousedown','mouseup','click','keydown','wheel']) {
  window.addEventListener(name, (e) => {
    const log = window.__FIXTURE_DIAGNOSTICS__.events;
    log.push({name: name, t: Math.round(performance.now()), buttons: e.buttons ?? null});
    if (log.length > 512) log.shift();
  }, true);
}
window.addEventListener('load', () => {
  %(setup_js)s
  const ready = %(ready_expr)s;
  post({kind: 'ready', geometry: ready.geometry, value: ready.value});
});
</script>
"""


def render_fixture_html(fixture: WebFixture, generation: int) -> str:
    """The page for one fixture at one generation."""
    params = fixture.params
    accent = html.escape(str(params.get("accent", "#1a73e8")), quote=True)
    head = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Capability fixture</title>
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin:0; width:100%; min-height:100%; font-family:Ubuntu,Arial,sans-serif;
  color:#17202a; background:#f5f7fa; }}
.banner {{ position:fixed; z-index:10; left:0; top:0; right:0; height:86px;
  background:#fff; border-bottom:4px solid {accent}; padding:16px 28px; }}
.banner h1 {{ margin:0 0 5px; font-size:24px; }} .banner p {{ margin:0; font-size:17px; }}
.card {{ position:absolute; background:#fff; border:2px solid #d5dbe3; border-radius:12px;
  padding:24px; box-shadow:0 4px 14px rgba(0,0,0,.10); }}
label {{ font-size:21px; font-weight:600; }}
</style></head><body>
<div class="banner"><h1>Browser control</h1><p>{html.escape(fixture.instruction)}</p></div>
"""
    if fixture.template == "click":
        position = _card_position(params, width=360, height=180)
        body = f"""
<div class="card" style="{position};width:360px">
 <label><input id="target" type="checkbox" style="width:28px;height:28px;vertical-align:middle">
 {html.escape(str(params['label']))}</label>
 <hr><label style="font-weight:400"><input id="decoy" type="checkbox"> Preview only</label>
</div>
<script>
const report = () => post({{kind:'click', checked:target.checked, decoy_checked:decoy.checked}});
target.addEventListener('change', report);
decoy.addEventListener('change', report);
</script>
"""
        ready = (
            "({geometry:measuredGeometry({target:screenRect(target),"
            "decoy:screenRect(decoy)}), value:target.checked})"
        )
        setup = ""
    elif fixture.template == "focus_type":
        position = _card_position(params, width=520, height=160)
        body = f"""
<div class="card" style="{position};width:520px">
 <label for="target">{html.escape(str(params['label']))}</label><br>
 <input id="target" value="{html.escape(str(params.get('initial_text', '')), quote=True)}"
  style="margin-top:14px;width:100%;height:52px;font-size:22px;padding:8px">
</div>
<script>target.addEventListener('input', () => post({{kind:'text', text:target.value}}));</script>
"""
        ready = "({geometry:measuredGeometry({target:screenRect(target)}), value:target.value})"
        setup = ""
    elif fixture.template == "drag":
        card_width = int(params["width"]) + 70
        position = _card_position(params, width=card_width, height=180)
        body = f"""
<div class="card" style="{position};width:{card_width}px">
 <label for="target">{html.escape(str(params['label']))}</label><br>
 <input id="target" type="range" min="0" max="100" step="1"
  value="{int(params['initial_value'])}"
  style="margin-top:22px;width:{int(params['width'])}px;height:42px;accent-color:{accent}">
 <output id="readout">{int(params['initial_value'])}</output>
</div>
<script>target.addEventListener('input', () => {{readout.value=target.value;
 post({{kind:'drag', value:Number(target.value)}});}});</script>
"""
        ready = "({geometry:measuredGeometry({target:screenRect(target)}), value:Number(target.value)})"
        setup = ""
    else:  # scroll
        blocks = "".join(
            f'<section style="height:420px;padding:120px 80px;font-size:28px;'
            f'background:{"#fff" if i % 2 else "#edf2f7"}">'
            f'{html.escape(str(params["label"]))} checkpoint {i}</section>'
            for i in range(1, 13)
        )
        body = f"""
<main style="padding-top:86px">{blocks}</main>
<script>
let scrollTimer;
window.addEventListener('scroll', () => {{ clearTimeout(scrollTimer);
 scrollTimer=setTimeout(() => post({{kind:'scroll', scroll_y:Math.round(window.scrollY)}}), 40); }});
</script>
"""
        ready = (
            "({geometry:measuredGeometry({viewport:{width:window.innerWidth,"
            "height:window.innerHeight}}), value:Math.round(window.scrollY)})"
        )
        setup = f"window.scrollTo(0, {int(params.get('initial_y', 0))});"
    common = _COMMON_JS % {
        "generation": generation,
        "fixture_id": urllib.parse.quote(fixture.id),
        "setup_js": setup,
        "ready_expr": ready,
    }
    return head + body + common + "</body></html>"


class WebFixtureServer:
    """Host-side HTTP server the guest browser loads pages from and posts state to.

    Bound on an ephemeral port and reached from the guest at `10.0.2.2` (qemu
    user-mode networking exposes the host there), so no guest-side server or port
    forward is needed in the inbound direction.
    """

    def __init__(self, fixtures: dict[str, WebFixture], *, host: str = "0.0.0.0") -> None:
        self.store = FixtureStateStore(fixtures=dict(fixtures))
        store = self.store
        # The store's dict IS the registry, not a copy: `preparers.web_fixture_server`
        # registers later fixtures by assigning into `server.store.fixtures`.
        registry = store.fixtures

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _send(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                if path == "/health":
                    self._send(200, b'{"status":"ok"}', "application/json")
                    return
                prefix = "/fixture/"
                if not path.startswith(prefix):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                fixture_id = urllib.parse.unquote(path[len(prefix) :])
                try:
                    fixture = registry[fixture_id]
                    generation = store.snapshot(fixture_id)["generation"]
                    body = render_fixture_html(fixture, generation).encode("utf-8")
                except Exception as exc:  # noqa: BLE001 - fail closed, return no state
                    self._send(404, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, body, "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                prefix = "/state/"
                if not path.startswith(prefix):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                fixture_id = urllib.parse.unquote(path[len(prefix) :])
                length = int(self.headers.get("Content-Length") or 0)
                if length > 1024 * 1024:
                    self._send(413, b"too large", "text/plain; charset=utf-8")
                    return
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    result = store.apply(fixture_id, payload)
                except Exception as exc:  # noqa: BLE001
                    self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def guest_url(self, fixture: WebFixture) -> str:
        return f"http://10.0.2.2:{self.port}/fixture/{urllib.parse.quote(fixture.id)}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "WebFixtureServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
