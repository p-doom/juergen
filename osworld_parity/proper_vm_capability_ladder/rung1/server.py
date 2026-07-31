from __future__ import annotations

import copy
import html
import json
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .fixtures import Fixture, FixtureManifest


class FixtureServerError(RuntimeError):
    pass


def _initial_current(fixture: Fixture) -> dict[str, Any]:
    if fixture.template == "click":
        return {"checked": False, "decoy_checked": False}
    if fixture.template == "focus_type":
        return {"text": fixture.params["initial_text"]}
    if fixture.template == "scroll":
        return {"scroll_y": int(fixture.params["initial_y"])}
    if fixture.template == "drag":
        return {"value": int(fixture.params["initial_value"])}
    raise FixtureServerError(f"unknown template {fixture.template!r}")


class FixtureStateStore:
    """Host-only oracle state. No HTTP read route is registered for this store."""

    def __init__(self, manifest: FixtureManifest) -> None:
        self._manifest = manifest
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._states: dict[str, dict[str, Any]] = {}
        for fixture in manifest.fixtures:
            self.reset(fixture)

    def reset(self, fixture: Fixture) -> int:
        with self._condition:
            generation = int(self._states.get(fixture.id, {}).get("generation", 0)) + 1
            self._states[fixture.id] = {
                "fixture_id": fixture.id,
                "fixture_sha256": fixture.fixture_sha256,
                "generation": generation,
                "ready": False,
                "geometry": {},
                "current": _initial_current(fixture),
                "events": [],
                "last_pointer_buttons": 0,
                "last_client_sequence": 0,
            }
            self._condition.notify_all()
            return generation

    def apply_event(self, fixture: Fixture, payload: dict[str, Any]) -> None:
        with self._condition:
            state = self._states[fixture.id]
            if payload.get("generation") != state["generation"]:
                raise FixtureServerError("stale fixture generation")
            kind = payload.get("kind")
            if not isinstance(kind, str):
                raise FixtureServerError("event kind missing")
            client_sequence = int(payload.get("client_sequence", -1))
            if client_sequence >= 0:
                if client_sequence <= state["last_client_sequence"]:
                    raise FixtureServerError(
                        f"non-monotonic client event sequence {client_sequence}"
                    )
                state["last_client_sequence"] = client_sequence
            event: dict[str, Any] = {
                "kind": kind,
                "client_sequence": client_sequence,
                "client_monotonic_ms": float(
                    payload.get("client_monotonic_ms", -1.0)
                ),
                "host_monotonic_ns": time.monotonic_ns(),
            }
            if kind == "ready":
                geometry = payload.get("geometry")
                if not isinstance(geometry, dict):
                    raise FixtureServerError("ready geometry missing")
                state["geometry"] = copy.deepcopy(geometry)
                state["ready"] = True
                browser_value = payload.get("value")
                if fixture.template == "click":
                    state["current"]["checked"] = bool(browser_value)
                elif fixture.template == "focus_type":
                    state["current"]["text"] = str(browser_value)
                elif fixture.template == "scroll":
                    state["current"]["scroll_y"] = int(browser_value)
                elif fixture.template == "drag":
                    state["current"]["value"] = int(browser_value)
            elif kind == "pointer":
                buttons = int(payload.get("buttons", 0))
                state["last_pointer_buttons"] = buttons
                event.update(
                    {
                        "event": str(payload.get("event", "")),
                        "button": int(payload.get("button", -1)),
                        "buttons": buttons,
                        "client_x": int(payload.get("client_x", -1)),
                        "client_y": int(payload.get("client_y", -1)),
                        "screen_x": int(payload.get("screen_x", -1)),
                        "screen_y": int(payload.get("screen_y", -1)),
                        "hit_id": str(payload.get("hit_id", "")),
                        "hit_tag": str(payload.get("hit_tag", "")),
                    }
                )
            elif kind == "click":
                state["current"]["checked"] = bool(payload.get("checked"))
                state["current"]["decoy_checked"] = bool(payload.get("decoy_checked"))
            elif kind == "text":
                state["current"]["text"] = str(payload.get("text", ""))
            elif kind == "scroll":
                state["current"]["scroll_y"] = int(payload.get("scroll_y", 0))
            elif kind == "drag":
                state["current"]["value"] = int(payload.get("value", -1))
            else:
                raise FixtureServerError(f"unsupported browser event {kind!r}")
            state["events"].append(event)
            self._condition.notify_all()

    def snapshot(self, fixture_id: str) -> dict[str, Any]:
        with self._lock:
            if fixture_id not in self._states:
                raise FixtureServerError(f"unknown fixture {fixture_id!r}")
            return copy.deepcopy(self._states[fixture_id])

    def wait_ready(self, fixture_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while time.monotonic() < deadline:
                state = self._states[fixture_id]
                if state["ready"]:
                    return copy.deepcopy(state)
                self._condition.wait(timeout=min(0.2, deadline - time.monotonic()))
        raise TimeoutError(f"fixture {fixture_id} did not report ready in {timeout_s}s")


class FixtureHttpServer:
    def __init__(self, manifest: FixtureManifest, *, host: str = "0.0.0.0") -> None:
        self.manifest = manifest
        self.store = FixtureStateStore(manifest)
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
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
                    fixture = manifest.by_id(fixture_id)
                    generation = store.snapshot(fixture.id)["generation"]
                    body = render_fixture_html(fixture, generation).encode("utf-8")
                except Exception as exc:  # fail closed; no state is returned
                    self._send(404, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, body, "text/html; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlsplit(self.path).path
                prefix = "/event/"
                if not path.startswith(prefix):
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 65536:
                        raise FixtureServerError("invalid event body length")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise FixtureServerError("event body must be an object")
                    fixture = manifest.by_id(urllib.parse.unquote(path[len(prefix) :]))
                    store.apply_event(fixture, payload)
                except Exception as exc:
                    self._send(
                        HTTPStatus.BAD_REQUEST,
                        json.dumps({"error": str(exc)}).encode("utf-8"),
                        "application/json",
                    )
                    return
                self._send(200, b'{"status":"accepted"}', "application/json")

        self._server = ThreadingHTTPServer((host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def guest_url(self, fixture: Fixture) -> str:
        # qemu user-mode networking exposes the host as 10.0.2.2.
        return f"http://10.0.2.2:{self.port}/fixture/{urllib.parse.quote(fixture.id)}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "FixtureHttpServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _common_script(fixture: Fixture, generation: int, ready_js: str) -> str:
    endpoint = f"/event/{urllib.parse.quote(fixture.id)}"
    return f"""
<script>
const generation = {generation};
const endpoint = {json.dumps(endpoint)};
let clientSequence = 0;
let postQueue = Promise.resolve();
function post(payload) {{
  payload.generation = generation;
  payload.client_sequence = ++clientSequence;
  payload.client_monotonic_ms = Math.round(performance.now() * 1000) / 1000;
  const body = JSON.stringify(payload);
  const send = () => fetch(endpoint, {{method: 'POST',
    headers: {{'Content-Type':'application/json'}}, body, cache: 'no-store'}})
    .then(response => {{ if (!response.ok) throw new Error(`event POST ${{response.status}}`);
      return response; }});
  // Preserve browser dispatch order at the host oracle. Independent fetches can
  // otherwise arrive out of order and make pointer traces non-causal.
  postQueue = postQueue.then(send, send);
  return postQueue;
}}
function screenRect(element) {{
  const r = element.getBoundingClientRect();
  const topChrome = Math.max(0, window.outerHeight - window.innerHeight);
  return {{left: Math.round(window.screenX + r.left),
           top: Math.round(window.screenY + topChrome + r.top),
           right: Math.round(window.screenX + r.right),
           bottom: Math.round(window.screenY + topChrome + r.bottom),
           width: Math.round(r.width), height: Math.round(r.height),
           center_x: Math.round(window.screenX + r.left + r.width / 2),
           center_y: Math.round(window.screenY + topChrome + r.top + r.height / 2)}};
}}
function measuredGeometry(parts) {{
  parts.window = {{screen_x: Math.round(window.screenX), screen_y: Math.round(window.screenY),
    screen_width: Math.round(window.screen.width), screen_height: Math.round(window.screen.height),
    outer_width: Math.round(window.outerWidth), outer_height: Math.round(window.outerHeight),
    inner_width: Math.round(window.innerWidth), inner_height: Math.round(window.innerHeight),
    chrome_top: Math.max(0, Math.round(window.outerHeight - window.innerHeight))}};
  return parts;
}}
for (const name of ['pointerdown', 'pointerup', 'pointermove']) {{
  document.addEventListener(name, (e) => {{
    if (name !== 'pointermove' || e.buttons) {{
      const hit = document.elementFromPoint(e.clientX, e.clientY);
      const topChrome = Math.max(0, window.outerHeight - window.innerHeight);
      post({{kind:'pointer', event:name, button:e.button, buttons:e.buttons,
        client_x:Math.round(e.clientX), client_y:Math.round(e.clientY),
        screen_x:Math.round(window.screenX + e.clientX),
        screen_y:Math.round(window.screenY + topChrome + e.clientY),
        hit_id:hit && hit.id ? hit.id : '',
        hit_tag:hit && hit.tagName ? hit.tagName.toLowerCase() : ''}});
    }}
  }}, true);
}}
window.addEventListener('load', () => requestAnimationFrame(() => {{ {ready_js} }}));
</script>
"""


def render_fixture_html(fixture: Fixture, generation: int) -> str:
    p = fixture.params
    accent = html.escape(str(p["accent"]), quote=True)
    base = f"""<!doctype html>
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
        position = _card_position(p, width=360, height=180)
        content = f"""
<div class="card" style="{position};width:360px">
 <label><input id="target" type="checkbox" style="width:28px;height:28px;vertical-align:middle">
 {html.escape(str(p['label']))}</label>
 <hr><label style="font-weight:400"><input id="decoy" type="checkbox"> Preview only</label>
</div>
<script>
target.addEventListener('change', () => post({{kind:'click', checked:target.checked,
 decoy_checked:decoy.checked}}));
decoy.addEventListener('change', () => post({{kind:'click', checked:target.checked,
 decoy_checked:decoy.checked}}));
</script>
"""
        ready = "post({kind:'ready', geometry:measuredGeometry({target:screenRect(target), decoy:screenRect(decoy)}), value:target.checked});"
    elif fixture.template == "focus_type":
        position = _card_position(p, width=520, height=160)
        content = f"""
<div class="card" style="{position};width:520px">
 <label for="target">{html.escape(str(p['label']))}</label><br>
 <input id="target" value="{html.escape(str(p['initial_text']), quote=True)}"
  style="margin-top:14px;width:100%;height:52px;font-size:22px;padding:8px">
</div>
<script>target.addEventListener('input', () => post({{kind:'text', text:target.value}}));</script>
"""
        ready = "post({kind:'ready', geometry:measuredGeometry({target:screenRect(target)}), value:target.value});"
    elif fixture.template == "drag":
        card_width = int(p["width"]) + 70
        position = _card_position(p, width=card_width, height=180)
        content = f"""
<div class="card" style="{position};width:{card_width}px">
 <label for="target">{html.escape(str(p['label']))}</label><br>
 <input id="target" type="range" min="0" max="100" step="1" value="{int(p['initial_value'])}"
  style="margin-top:22px;width:{int(p['width'])}px;height:42px;accent-color:{accent}">
 <output id="readout">{int(p['initial_value'])}</output>
</div>
<script>target.addEventListener('input', () => {{readout.value=target.value;
 post({{kind:'drag', value:Number(target.value)}});}});</script>
"""
        ready = "post({kind:'ready', geometry:measuredGeometry({target:screenRect(target)}), value:Number(target.value)});"
    elif fixture.template == "scroll":
        blocks = "".join(
            f'<section style="height:420px;padding:120px 80px;font-size:28px;background:{"#fff" if i % 2 else "#edf2f7"}">'
            f'{html.escape(str(p["label"]))} checkpoint {i}</section>'
            for i in range(1, 13)
        )
        content = f"""
<main style="padding-top:86px">{blocks}</main>
<script>
let scrollTimer;
window.addEventListener('scroll', () => {{ clearTimeout(scrollTimer); scrollTimer=setTimeout(() =>
 post({{kind:'scroll', scroll_y:Math.round(window.scrollY)}}), 40); }});
</script>
"""
        ready = (
            f"window.scrollTo(0, {int(p['initial_y'])}); requestAnimationFrame(() => "
            "post({kind:'ready', geometry:measuredGeometry({viewport:{width:window.innerWidth,height:window.innerHeight}}), "
            "value:Math.round(window.scrollY)}));"
        )
    else:
        raise FixtureServerError(f"unknown template {fixture.template!r}")
    return base + content + _common_script(fixture, generation, ready) + "</body></html>"


def _card_position(params: dict[str, Any], *, width: int, height: int) -> str:
    """Map sealed 1920x1080 design coordinates into the measured viewport.

    The final CSS clamp keeps the whole card visible even when Chrome's actual
    inner viewport is smaller than the QCOW's 1920x1080 desktop.
    """
    left_percent = 100.0 * int(params["left"]) / 1920.0
    top_percent = 100.0 * int(params["top"]) / 1080.0
    return (
        f"left:clamp(24px,{left_percent:.6f}vw,calc(100vw - {width + 24}px));"
        f"top:clamp(104px,{top_percent:.6f}vh,calc(100vh - {height + 24}px))"
    )
