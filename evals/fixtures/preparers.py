"""The `Preparer`s that attach fixtures to tasks.

This is the seam the coordinator asked for: a task that needs Chrome brings its own
fixture, and the session never learns Chrome exists. Two kinds:

  * `web_fixture` — browser fixtures served from the host. The `WebFixtureServer` is
    process-global and lazily started, because a per-episode server would burn a port
    and a thread per rollout and the pages are pure functions of `(fixture, generation)`.
  * `app_fixture` — Writer / Calc / Files / in-guest Chrome.

Both publish `postcondition_success` in the probe only when the family can decide it
in-loop; otherwise the oracle reward decides and the episode runs its full horizon.
"""

from __future__ import annotations

import threading
from typing import Any

from evals.fixtures.apps import AppFixture, probe_app_state, setup_app_fixture
from evals.fixtures.chrome import capture_browser_diagnostics, launch as launch_chrome
from evals.fixtures.web import WebFixture, WebFixtureServer
from evals.tasks import DesktopTaskData, register_preparer

__all__ = ["AppFixturePreparer", "WebFixturePreparer", "web_fixture_server"]

_SERVER: WebFixtureServer | None = None
_SERVER_LOCK = threading.Lock()


def _web_fixture(task: DesktopTaskData) -> WebFixture:
    spec = dict(task.setup.get("fixture") or {})
    return WebFixture(
        id=str(spec.get("id") or task.name or f"fixture_{task.idx}"),
        template=str(spec["template"]),
        instruction=task.instruction,
        params=dict(spec.get("params") or {}),
    )


def web_fixture_server(fixture: WebFixture) -> WebFixtureServer:
    """The process's fixture server, started on first use.

    Registered fixtures accumulate: a worker serves every fixture any of its rollouts
    asked for, and a rollout's page is identified by fixture id, so two concurrent
    rollouts of different cells cannot collide.

    `server.store.fixtures` is the one registry. There used to be a module-level dict
    as well, handed to the constructor for the first fixture and then written to in
    parallel forever after — and `FixtureStateStore` copies what it is given, so the
    two diverged by construction.
    """
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = WebFixtureServer({})
            _SERVER.start()
        _SERVER.store.fixtures[fixture.id] = fixture
        return _SERVER


class WebFixturePreparer:
    """Browser fixture: serve the page, launch Chrome, read state back."""

    kind = "web_fixture"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        fixture = _web_fixture(task)
        server = web_fixture_server(fixture)
        result = launch_chrome(session, server, fixture)
        # Launch evidence is data, not a gate: a page that came up on the third
        # attempt is a slow VM, not a different experiment.
        return {"fixture": fixture.id, "template": fixture.template, "chrome": result.as_dict()}

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        del session
        fixture = _web_fixture(task)
        server = web_fixture_server(fixture)
        state = server.store.snapshot(fixture.id)
        probe: dict[str, Any] = {
            "schema_version": 1,
            "task_id": task.name,
            "fixture_id": fixture.id,
            "generation": state["generation"],
            "ready": state["ready"],
            "current": state["current"],
            "geometry": state["geometry"],
            "events": state["events"][-32:],
            "postcondition_status": "ok",
        }
        port = task.setup.get("chromium_port")
        if port:
            try:
                probe["browser"] = capture_browser_diagnostics(fixture, int(port))
            except Exception as exc:  # noqa: BLE001 - the HTTP state is the primary read
                probe["browser"] = {"status": "unavailable", "error": repr(exc)}
        return probe


def _app_fixture(task: DesktopTaskData) -> AppFixture:
    spec = dict(task.setup.get("fixture") or {})
    return AppFixture(
        id=str(spec.get("id") or task.name or f"fixture_{task.idx}"),
        app=str(spec["app"]),
        instruction=task.instruction,
        params=dict(spec.get("params") or {}),
    )


class AppFixturePreparer:
    """Writer / Calc / Files / in-guest Chrome fixture."""

    kind = "app_fixture"

    def prepare(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        return setup_app_fixture(session, _app_fixture(task))

    def probe(self, session: Any, task: DesktopTaskData) -> dict[str, Any]:
        state = probe_app_state(session, _app_fixture(task))
        return {**state, "task_id": task.name, "postcondition_status": "ok"}


register_preparer(WebFixturePreparer())
register_preparer(AppFixturePreparer())
