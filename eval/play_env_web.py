"""Web UI backend for play_env — a stdlib http.server that drives a Session.

Started by ``play_env.py --ui-port N`` (in place of the REPL). It serves a single
vanilla-JS page (``play_env_ui.html``) plus JSON ``/api/*`` endpoints and the live
screenshot, and calls the same ``Session`` methods the REPL exposes. View-only for
the VM screen (no click/type from the browser, per design); you drive the model
(ask/step/run), edit goal/prompt/decoding, start/browse conversations.

Pattern mirrors the repo's other stdlib dashboards
(data_pipeline/annotation_pipeline/goal_timeline_viewer/annotator.py and
visualize_run.py): ThreadingHTTPServer + do_GET/do_POST + JSON helpers, HTML on
disk, media root-gated.

Concurrency: a single lock serializes VM/model operations. run(n) executes in a
background thread taking the lock per step, so /api/state (lockless) and /api/stop
stay responsive while a rollout is in flight.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_HTML = Path(__file__).resolve().parent / "play_env_ui.html"
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class _App:
    """Shared server state: the Session, a serialization lock, and run status."""

    def __init__(self, sess: Any):
        self.sess = sess
        self.lock = threading.Lock()
        self.run = {"active": False, "target": 0, "done": 0, "stop": False, "error": None}

    # --- background rollout ------------------------------------------------
    def start_run(self, n: int) -> None:
        if self.run["active"]:
            return
        self.run.update(active=True, target=n, done=0, stop=False, error=None)
        threading.Thread(target=self._run_worker, args=(n,), daemon=True).start()

    def _run_worker(self, n: int) -> None:
        try:
            for _ in range(n):
                if self.run["stop"]:
                    break
                with self.lock:
                    self.sess.step()
                self.run["done"] += 1
                if getattr(self.sess, "terminated", False):
                    break
        except Exception as e:  # surface worker crashes to the UI
            self.run["error"] = str(e)
        finally:
            self.run["active"] = False

    def busy(self) -> bool:
        return self.run["active"] or self.lock.locked()


def _handler(app: _App) -> type[BaseHTTPRequestHandler]:
    out_dir: Path = app.sess.out
    conv_root: Path = app.sess.conv_root

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a: Any) -> None:  # silence access log
            pass

        # --- response helpers ---------------------------------------------
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        def _png(self, path: Path) -> None:
            if not path.is_file():
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send(HTTPStatus.OK, path.read_bytes(), "image/png")

        # --- GET ----------------------------------------------------------
        def do_GET(self) -> None:
            u = urllib.parse.urlparse(self.path)
            path, q = u.path, urllib.parse.parse_qs(u.query)
            try:
                if path == "/":
                    self._send(HTTPStatus.OK, _HTML.read_bytes(), "text/html; charset=utf-8")
                elif path == "/latest.png":
                    self._png(out_dir / "latest.png")
                elif path == "/api/state":
                    st = app.sess.state()
                    st["busy"] = app.busy()
                    st["run"] = dict(app.run)
                    self._json(st)
                elif path == "/api/prompts":
                    self._json(app.sess.list_prompts())
                elif path == "/api/prompt_text":
                    self._json({"text": self._prompt_text(q)})
                elif path == "/api/episodes":
                    self._json(app.sess.list_episodes())
                elif path == "/api/episode":
                    self._json(app.sess.read_episode(q.get("name", [""])[0]))
                elif path == "/api/frame":
                    self._frame(q.get("ep", [""])[0], q.get("name", [""])[0])
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._json({"error": repr(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _prompt_text(self, q: dict[str, list[str]]) -> str:
            """Load a prompt's text for the editor (no session mutation)."""
            pid = q.get("id", [""])[0]
            if pid:
                from osworld_system_prompts import SYSTEM_PROMPTS
                if pid not in SYSTEM_PROMPTS:
                    raise ValueError("unknown prompt id")
                return SYSTEM_PROMPTS[pid]
            path = q.get("path", [""])[0]
            allowed = set(app.sess.list_prompts()["files"])  # only advertised files
            if path not in allowed:
                raise ValueError("prompt file not allowed")
            return Path(path).read_text()

        def _frame(self, ep: str, name: str) -> None:
            if not (_NAME_RE.match(ep) and _NAME_RE.match(name) and name.endswith(".png")):
                raise ValueError("bad frame ref")
            p = (conv_root / ep / name).resolve()
            if p.parent.parent != conv_root.resolve():
                raise ValueError("frame outside conversation root")
            self._png(p)

        # --- POST ---------------------------------------------------------
        def do_POST(self) -> None:
            u = urllib.parse.urlparse(self.path)
            path = u.path
            body = self._body()
            try:
                if path == "/api/stop":  # lockless: just flip the flag
                    app.run["stop"] = True
                    self._json({"ok": True})
                    return
                if path == "/api/run":
                    if body.get("message") is not None:
                        app.sess.set_message(body["message"])
                    app.start_run(int(body.get("n", 10)))
                    self._json({"started": True, "run": dict(app.run)})
                    return

                # Everything else mutates the Session → serialize on the lock.
                with app.lock:
                    # ask/step may carry the current "message to model" (the UI
                    # sends the message box's content with each action).
                    if path in ("/api/ask", "/api/step") and body.get("message") is not None:
                        app.sess.set_message(body["message"])
                    if path == "/api/goal":
                        app.sess.goal(body.get("goal"))
                    elif path == "/api/prompt":
                        app.sess.setprompt(body.get("value", ""))
                    elif path == "/api/config":
                        self._apply_config(body)
                    elif path == "/api/save_prompt":
                        app.sess.save_prompt(body.get("name", ""), body.get("text", ""))
                    elif path == "/api/rename":
                        app.sess.rename_episode(body.get("name", ""), body.get("label", ""))
                    elif path == "/api/message":
                        app.sess.set_message(body.get("text"))
                    elif path == "/api/note":
                        app.sess.add_note(body.get("text", ""))
                    elif path == "/api/ask":
                        app.sess.ask()
                    elif path == "/api/step":
                        app.sess.step()
                    elif path == "/api/new_conversation":
                        prompt = body.get("prompt_text") or body.get("prompt_id")
                        app.sess.new_conversation(
                            goal=body.get("goal"), prompt=prompt,
                            reboot=bool(body.get("reboot")),
                            terminal=bool(body.get("terminal")),
                        )
                    elif path == "/api/replay":
                        # Seed a fresh conversation from a past one + arm its actions.
                        t = body.get("terminal")
                        app.sess.replay_conversation(
                            body.get("name", ""),
                            reboot=bool(body.get("reboot")),
                            terminal=None if t is None else bool(t),
                        )
                    elif path == "/api/replay_step":
                        # op: 'next' (copy+dispatch), 'skip' (advance), 'stop' (disarm).
                        app.sess.replay_step(body.get("op", "next"))
                    else:
                        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                        return
                st = app.sess.state()
                st["busy"] = app.busy()
                st["run"] = dict(app.run)
                self._json(st)
            except ValueError as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._json({"error": repr(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _apply_config(self, body: dict[str, Any]) -> None:
            s = app.sess
            if "max_tokens" in body:
                s.max_tokens = int(body["max_tokens"])
            if "temperature" in body:
                s.temperature = float(body["temperature"])
            if "n_history_frames" in body:
                s.n_history_frames = int(body["n_history_frames"])
            if "history_action_only" in body:
                s.history_action_only = bool(body["history_action_only"])
            s._log_config()

    return Handler


def serve(sess: Any, *, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Blocking: serve the play_env web UI until Ctrl-C."""
    app = _App(sess)
    httpd = ThreadingHTTPServer((host, port), _handler(app))
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
