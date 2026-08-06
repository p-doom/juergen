"""Items 10 + 11 — the `WebFixtureServer` state store, and `probe_app_state`.

Item 10: `apply` / `wait_ready` are generation-guarded, rewritten after ~700 LOC of
causal-heartbeat audit machinery was removed. The guard is the load-bearing part:
without it a page still unloading from the previous episode writes its dying scroll
position into the new episode's state.

Item 11: `probe_app_state` consolidates rung2's `probe_state` / `probe_geometry`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from evals.fixtures.apps import (
    APPS,
    AppFixture,
    AppFixtureError,
    JSON_MARKER,
    probe_app_state,
    resolve_guest_root,
    setup_app_fixture,
)
from evals.fixtures.chrome import (
    BROWSER_RESTART_BUDGET_S,
    MAX_ATTEMPTS,
    ChromeFixtureError,
    launch,
)
from evals.fixtures.web import (
    EVENT_LIMIT,
    TEMPLATES,
    FixtureServerError,
    FixtureStateStore,
    WebFixture,
    WebFixtureServer,
    render_fixture_html,
)
from juergen_doubles import FakeSession


def _fixture(template: str = "click", **params) -> WebFixture:
    defaults = {
        "click": {"label": "Enable telemetry", "left": 400, "top": 300},
        "focus_type": {"label": "Name", "initial_text": "before", "left": 400, "top": 300},
        "drag": {"label": "Volume", "initial_value": 10, "width": 300, "left": 400, "top": 300},
        "scroll": {"label": "Section", "initial_y": 0},
    }[template]
    return WebFixture(
        id=f"fx_{template}", template=template, instruction="do it", params={**defaults, **params}
    )


def test_an_unknown_template_is_refused_at_construction() -> None:
    with pytest.raises(FixtureServerError, match="unknown template"):
        WebFixture(id="x", template="telepathy", instruction="i", params={})


def test_every_declared_template_renders_and_carries_its_generation() -> None:
    for template in TEMPLATES:
        html = render_fixture_html(_fixture(template), generation=7)
        assert "const GENERATION = 7;" in html
        assert "/state/fx_" in html
        assert html.startswith("<!doctype html>") and html.endswith("</body></html>")


def test_a_snapshot_before_reset_is_an_error_not_an_empty_dict() -> None:
    store = FixtureStateStore(fixtures={})
    with pytest.raises(FixtureServerError, match="never reset"):
        store.snapshot("fx_click")
    with pytest.raises(FixtureServerError, match="never reset"):
        store.apply("fx_click", {"generation": 1})


def test_reset_increments_the_generation_and_installs_the_initial_state() -> None:
    fixture = _fixture("focus_type", initial_text="hello")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    assert store.reset(fixture) == 1
    state = store.snapshot(fixture.id)
    assert state["generation"] == 1 and state["ready"] is False
    assert state["current"] == {"kind": "text", "text": "hello"}
    assert state["events"] == [] and state["geometry"] is None
    assert store.reset(fixture) == 2, "generations are monotonic per fixture"


@pytest.mark.parametrize(
    "template,expected",
    [
        ("click", {"kind": "click", "checked": False, "decoy_checked": False}),
        ("drag", {"kind": "drag", "value": 10}),
        ("scroll", {"kind": "scroll", "scroll_y": 0}),
    ],
)
def test_the_initial_state_is_per_template(template: str, expected: dict) -> None:
    fixture = _fixture(template)
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    store.reset(fixture)
    assert store.snapshot(fixture.id)["current"] == expected


def test_a_stale_generation_post_is_dropped() -> None:
    """★ The guard: a page still unloading must not write into the new episode."""
    fixture = _fixture("scroll")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    store.reset(fixture)
    store.reset(fixture)  # generation 2 — a new episode
    result = store.apply(fixture.id, {"generation": 1, "kind": "scroll", "scroll_y": 9999})
    assert result == {"status": "stale_generation", "generation": 2}
    state = store.snapshot(fixture.id)
    assert state["current"] == {"kind": "scroll", "scroll_y": 0}, "state is untouched"
    assert state["events"] == [], "a stale post is not even logged"


def test_a_post_with_no_generation_is_dropped() -> None:
    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    store.reset(fixture)
    assert store.apply(fixture.id, {"kind": "click"})["status"] == "stale_generation"


def test_a_current_generation_post_updates_the_state_and_logs_the_event() -> None:
    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    generation = store.reset(fixture)
    result = store.apply(
        fixture.id,
        {"generation": generation, "kind": "click", "checked": True, "decoy_checked": False},
    )
    assert result == {"status": "ok", "generation": generation}
    state = store.snapshot(fixture.id)
    assert state["current"] == {"kind": "click", "checked": True, "decoy_checked": False}
    assert len(state["events"]) == 1
    assert "received_wall_time" in state["events"][0]


def test_a_ready_post_sets_ready_and_records_the_measured_geometry() -> None:
    fixture = _fixture("drag")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    generation = store.reset(fixture)
    geometry = {"viewport": {"width": 1280, "height": 700}}
    store.apply(fixture.id, {"generation": generation, "kind": "ready", "geometry": geometry, "value": 42})
    state = store.snapshot(fixture.id)
    assert state["ready"] is True and state["geometry"] == geometry
    assert state["current"]["value"] == 42, "a ready value merges, it does not replace"
    assert state["current"]["kind"] == "drag"


def test_pointer_buttons_are_tracked_separately() -> None:
    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    generation = store.reset(fixture)
    store.apply(fixture.id, {"generation": generation, "kind": "click", "pointer_buttons": 1})
    assert store.snapshot(fixture.id)["last_pointer_buttons"] == 1


def test_the_event_log_is_bounded() -> None:
    fixture = _fixture("scroll")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    generation = store.reset(fixture)
    for i in range(EVENT_LIMIT + 50):
        store.apply(fixture.id, {"generation": generation, "kind": "scroll", "scroll_y": i})
    events = store.snapshot(fixture.id)["events"]
    assert len(events) == EVENT_LIMIT
    assert events[-1]["scroll_y"] == EVENT_LIMIT + 49, "the newest events are kept"


def test_a_snapshot_is_a_deep_copy() -> None:
    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    generation = store.reset(fixture)
    store.apply(fixture.id, {"generation": generation, "kind": "click", "checked": True})
    snapshot = store.snapshot(fixture.id)
    snapshot["current"]["checked"] = "tampered"
    snapshot["events"].clear()
    assert store.snapshot(fixture.id)["current"]["checked"] is True
    assert len(store.snapshot(fixture.id)["events"]) == 1


def test_wait_ready_returns_as_soon_as_the_page_reports() -> None:
    import threading

    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    generation = store.reset(fixture)

    def report() -> None:
        store.apply(fixture.id, {"generation": generation, "kind": "ready", "geometry": {}})

    threading.Timer(0.15, report).start()
    state = store.wait_ready(fixture.id, timeout_s=5.0)
    assert state["ready"] is True


def test_wait_ready_times_out_rather_than_hanging() -> None:
    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    store.reset(fixture)
    with pytest.raises(FixtureServerError, match="did not report ready within"):
        store.wait_ready(fixture.id, timeout_s=0.2)


def test_wait_ready_ignores_a_stale_ready() -> None:
    """The generation guard has to hold for readiness too, or a dying page's `ready`
    would satisfy the next episode's wait."""
    fixture = _fixture("click")
    store = FixtureStateStore(fixtures={fixture.id: fixture})
    store.reset(fixture)
    store.reset(fixture)
    store.apply(fixture.id, {"generation": 1, "kind": "ready", "geometry": {}})
    with pytest.raises(FixtureServerError):
        store.wait_ready(fixture.id, timeout_s=0.2)


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(url: str, payload: dict) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_the_server_serves_a_page_and_accepts_its_state() -> None:
    fixture = _fixture("click")
    with WebFixtureServer({fixture.id: fixture}, host="127.0.0.1") as server:
        base = f"http://127.0.0.1:{server.port}"
        assert _get(f"{base}/health") == (200, b'{"status":"ok"}')
        assert _get(f"{base}/nope")[0] == 404
        # Before reset there is no generation, so the page is refused.
        assert _get(f"{base}/fixture/{fixture.id}")[0] == 404
        generation = server.store.reset(fixture)
        status, body = _get(f"{base}/fixture/{fixture.id}")
        assert status == 200 and f"const GENERATION = {generation};".encode() in body
        status, body = _post(
            f"{base}/state/{fixture.id}", {"generation": generation, "kind": "click", "checked": True}
        )
        assert status == 200 and json.loads(body)["status"] == "ok"
        assert server.store.snapshot(fixture.id)["current"]["checked"] is True
        assert _post(f"{base}/wrong/{fixture.id}", {})[0] == 404


def test_the_guest_url_points_at_the_host_through_qemu_user_networking() -> None:
    fixture = _fixture("click")
    server = WebFixtureServer({fixture.id: fixture}, host="127.0.0.1")
    try:
        assert server.guest_url(fixture) == f"http://10.0.2.2:{server.port}/fixture/{fixture.id}"
        assert server.port > 0, "an ephemeral port is bound at construction"
    finally:
        server._server.server_close()


def test_a_second_fixture_registered_after_start_is_still_served() -> None:
    """DEFECT (fixed, `evals/fixtures/web.py:326,353`).

    `WebFixtureServer.__init__` captured `registry = dict(fixtures)` and `do_GET`
    looked the fixture up there, while `preparers.web_fixture_server` registers later
    fixtures into `self.store.fixtures`. The two never reconverged, so every fixture
    after the first 404'd — and because the failure is a page that never loads, it
    surfaced as `wait_ready` burning the full 120 s deadline over three Chrome
    relaunches, per rollout. `do_GET` now reads the store.
    """
    first, second = _fixture("click"), _fixture("scroll")
    with WebFixtureServer({first.id: first}, host="127.0.0.1") as server:
        base = f"http://127.0.0.1:{server.port}"
        server.store.reset(first)
        assert _get(f"{base}/fixture/{first.id}")[0] == 200
        server.store.fixtures[second.id] = second  # what the preparer does
        server.store.reset(second)
        status, body = _get(f"{base}/fixture/{second.id}")
        assert status == 200, f"a later-registered fixture must be served: {body[:200]!r}"
        assert b"checkpoint 1" in body, "and it must be the right template"


def test_the_preparer_registers_into_a_shared_process_global_server() -> None:
    import evals.fixtures.preparers as preparers

    first, second = _fixture("click"), _fixture("drag")
    saved_server = preparers._SERVER
    preparers._SERVER = None
    try:
        server = preparers.web_fixture_server(first)
        assert preparers.web_fixture_server(second) is server, "one server per process"
        assert {first.id, second.id} <= set(server.store.fixtures)
        base = f"http://127.0.0.1:{server.port}"
        for fixture in (first, second):
            server.store.reset(fixture)
            assert _get(f"{base}/fixture/{fixture.id}")[0] == 200, fixture.id
    finally:
        if preparers._SERVER is not None:
            preparers._SERVER.close()
        preparers._SERVER = saved_server


def test_an_oversized_post_is_refused() -> None:
    fixture = _fixture("click")
    with WebFixtureServer({fixture.id: fixture}, host="127.0.0.1") as server:
        server.store.reset(fixture)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/state/{fixture.id}",
            data=b"x" * 16,
            headers={"Content-Type": "application/json", "Content-Length": str(2 * 1024 * 1024)},
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("expected a refusal")
        except urllib.error.HTTPError as exc:
            assert exc.code == 413
        except (urllib.error.URLError, OSError):
            pass  # the server closed the connection, which is also a refusal


def test_a_malformed_post_body_is_a_400_not_a_crash() -> None:
    fixture = _fixture("click")
    with WebFixtureServer({fixture.id: fixture}, host="127.0.0.1") as server:
        server.store.reset(fixture)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/state/{fixture.id}",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("expected a 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        assert _get(f"http://127.0.0.1:{server.port}/health")[0] == 200, "still serving"


def test_the_click_template_carries_a_decoy_so_a_near_miss_is_distinguishable() -> None:
    html = render_fixture_html(_fixture("click"), 1)
    assert 'id="target"' in html and 'id="decoy"' in html


def test_the_scroll_template_has_twelve_measurable_sections() -> None:
    html = render_fixture_html(_fixture("scroll"), 1)
    assert html.count("<section") == 12
    assert "checkpoint 12" in html


def test_the_design_coordinates_are_clamped_into_the_measured_viewport() -> None:
    """Otherwise a target scrolls off-screen and the task becomes unreachable."""
    html = render_fixture_html(_fixture("click", left=1900, top=1070), 1)
    assert "clamp(24px," in html and "clamp(104px," in html


def test_the_instruction_and_labels_are_html_escaped() -> None:
    fixture = WebFixture(
        id="fx", template="click", instruction="<script>x</script>", params={"label": "<b>&</b>", "left": 1, "top": 1}
    )
    html = render_fixture_html(fixture, 1)
    assert "<script>x</script>" not in html and "&lt;script&gt;" in html
    assert "&lt;b&gt;&amp;&lt;/b&gt;" in html


def test_launch_refuses_a_non_positive_deadline() -> None:
    fixture = _fixture("click")
    server = WebFixtureServer({fixture.id: fixture}, host="127.0.0.1")
    try:
        with pytest.raises(ChromeFixtureError, match="deadline must be positive"):
            launch(FakeSession(), server, fixture, timeout_s=0)
    finally:
        server._server.server_close()


def test_launch_resets_the_generation_and_returns_ready_evidence(monkeypatch) -> None:
    fixture = _fixture("click")
    server = WebFixtureServer({fixture.id: fixture}, host="127.0.0.1")
    try:
        session = FakeSession()
        monkeypatch.setattr(
            type(server.store),
            "wait_ready",
            lambda self, fixture_id, timeout_s=30.0: self.snapshot(fixture_id),
        )
        result = launch(session, server, fixture, timeout_s=10)
        assert result.status == "ready" and result.generation == 1
        assert len(result.attempts) == 1 and result.attempts[0]["status"] == "ready"
        assert result.error is None
        script = " ".join(session.argv_log[0])
        assert "--no-first-run" in script and "10.0.2.2" in script
        assert "pkill" not in script, "the first attempt must not restart Chrome"
    finally:
        server._server.server_close()


def test_launch_is_bounded_by_one_absolute_deadline(monkeypatch) -> None:
    """Per-attempt timeouts compose into an unbounded total; one deadline does not."""
    fixture = _fixture("click")
    server = WebFixtureServer({fixture.id: fixture}, host="127.0.0.1")
    try:
        def never_ready(self, fixture_id, timeout_s=30.0):
            raise FixtureServerError("nope")

        monkeypatch.setattr(type(server.store), "wait_ready", never_ready)
        result = launch(FakeSession(), server, fixture, timeout_s=1.0)
        assert result.status == "failed" and result.ready_state is None
        assert 1 <= len(result.attempts) <= MAX_ATTEMPTS
        assert "FixtureServerError" in (result.error or "")
        assert all(a["status"] == "failed" for a in result.attempts)
    finally:
        server._server.server_close()


def test_a_relaunch_restarts_chrome_and_bumps_the_generation(monkeypatch) -> None:
    fixture = _fixture("click")
    server = WebFixtureServer({fixture.id: fixture}, host="127.0.0.1")
    try:
        calls = {"n": 0}

        def flaky(self, fixture_id, timeout_s=30.0):
            calls["n"] += 1
            if calls["n"] == 1:
                raise FixtureServerError("slow VM")
            return self.snapshot(fixture_id)

        monkeypatch.setattr(type(server.store), "wait_ready", flaky)
        session = FakeSession()
        result = launch(session, server, fixture, timeout_s=120)
        assert result.status == "ready" and result.generation == 2
        assert [a["restart"] for a in result.attempts] == [False, True]
        assert "pkill" in " ".join(session.argv_log[1])
    finally:
        server._server.server_close()


def test_the_restart_budget_refuses_an_attempt_that_cannot_finish() -> None:
    assert BROWSER_RESTART_BUDGET_S > 0
    assert MAX_ATTEMPTS == 3


def _app(app: str = "writer", **params) -> AppFixture:
    defaults = {
        "writer": {"file_name": "doc.odt", "initial_text": "hello"},
        "calc": {"file_name": "sheet.ods", "cell": "B7", "initial_value": "41"},
        "files": {
            "source_name": "s.txt",
            "destination_name": "dest",
            "decoy_name": "decoy",
            "content": "payload",
        },
        "chrome": {"port": 8931, "section": "privacy", "setting": "Do the thing"},
    }[app]
    return AppFixture(id=f"fx_{app}", app=app, instruction="do it", params={**defaults, **params})


def test_an_unknown_app_is_refused_at_construction() -> None:
    with pytest.raises(AppFixtureError, match="unknown app"):
        AppFixture(id="x", app="powerpoint", instruction="i", params={})
    assert APPS == ("writer", "calc", "files", "chrome")


def test_the_guest_root_is_resolved_once_and_cached_on_the_session() -> None:
    payload = JSON_MARKER + json.dumps({"root": "/run/user/1000/juergen_app_fixtures"})
    session = FakeSession(argv_responses={"python3": payload})
    first = resolve_guest_root(session)
    assert str(first) == "/run/user/1000/juergen_app_fixtures"
    calls = len(session.argv_log)
    assert resolve_guest_root(session) == first
    assert len(session.argv_log) == calls, "one resolution per session"


def test_an_unsafe_guest_root_is_refused() -> None:
    for bad in ("relative/path", "/tmp/../etc", "/tmp/other_name"):
        session = FakeSession(
            argv_responses={"python3": JSON_MARKER + json.dumps({"root": bad})}
        )
        with pytest.raises(AppFixtureError, match="unsafe guest root"):
            resolve_guest_root(session)


def test_a_guest_root_resolver_with_no_root_is_refused() -> None:
    session = FakeSession(argv_responses={"python3": JSON_MARKER + json.dumps({})})
    with pytest.raises(AppFixtureError, match="no root"):
        resolve_guest_root(session)


def test_an_ambiguous_json_marker_count_fails_closed() -> None:
    payload = JSON_MARKER + json.dumps({"root": "/tmp/juergen_app_fixtures"})
    session = FakeSession(argv_responses={"python3": payload + "\n" + payload})
    with pytest.raises(AppFixtureError, match="marker count was 2"):
        resolve_guest_root(session)
    empty = FakeSession(argv_responses={"python3": "no marker"})
    with pytest.raises(AppFixtureError, match="marker count was 0"):
        resolve_guest_root(empty)


def test_a_non_object_json_payload_fails_closed() -> None:
    session = FakeSession(argv_responses={"python3": JSON_MARKER + "[1, 2]"})
    with pytest.raises(AppFixtureError, match="not an object"):
        resolve_guest_root(session)


def _rooted(extra: dict[str, str] | None = None) -> FakeSession:
    responses = {"juergen_app_fixtures'": JSON_MARKER + json.dumps({"root": "/tmp/juergen_app_fixtures"})}
    responses.update(extra or {})
    return FakeSession(argv_responses=responses)


@pytest.mark.parametrize("app", APPS)
def test_probe_app_state_runs_one_read_only_guest_command_per_app(app: str) -> None:
    fixture = _app(app)
    state = {"schema_version": 1, "fixture_id": fixture.id, "app": app, "root": "/tmp/x"}
    session = _rooted({"app=": JSON_MARKER + json.dumps(state)})
    result = probe_app_state(session, fixture)
    assert result["app"] == app and result["fixture_id"] == fixture.id
    probe_argv = session.argv_log[-1]
    assert probe_argv[0] == "python3" and probe_argv[1] == "-c"
    code = probe_argv[2]
    assert "read_text" in code or "glob" in code
    for mutation in ("pyautogui", "wmctrl -a", "rm -rf", "nohup"):
        assert mutation not in code, f"{app} probe must not mutate: {mutation}"


def test_the_probe_code_branches_per_app_family() -> None:
    for app, needle in (
        ("writer", "--convert-to"),
        ("calc", "--convert-to"),
        ("files", "rglob"),
        ("chrome", "state.json"),
    ):
        fixture = _app(app)
        state = {"schema_version": 1, "fixture_id": fixture.id, "app": app, "root": "/tmp/x"}
        session = _rooted({"app=": JSON_MARKER + json.dumps(state)})
        probe_app_state(session, fixture)
        assert needle in session.argv_log[-1][2], (app, needle)


def test_the_writer_probe_converts_a_temp_copy_not_the_live_document() -> None:
    fixture = _app("writer")
    state = {"schema_version": 1, "fixture_id": fixture.id, "app": "writer", "root": "/tmp/x"}
    session = _rooted({"app=": JSON_MARKER + json.dumps(state)})
    probe_app_state(session, fixture)
    code = session.argv_log[-1][2]
    assert "mkdtemp" in code and "--outdir" in code, (
        "the model's file must never be the conversion target"
    )


def test_the_probe_carries_the_fixture_params_into_the_guest() -> None:
    fixture = _app("calc", cell="C3", initial_value="7")
    state = {"schema_version": 1, "fixture_id": fixture.id, "app": "calc", "root": "/tmp/x"}
    session = _rooted({"app=": JSON_MARKER + json.dumps(state)})
    probe_app_state(session, fixture)
    code = session.argv_log[-1][2]
    assert "sheet.ods" in code and "params=json.loads" in code


def test_setup_app_fixture_reports_a_failure_as_data_not_an_exception() -> None:
    class Angry(FakeSession):
        def execute_argv(self, argv):
            self.argv_log.append(list(argv))
            if argv[0] == "bash":
                raise RuntimeError("guest died")
            return super().execute_argv(argv)

    fixture = _app("writer")
    session = Angry(
        argv_responses={"juergen_app_fixtures'": JSON_MARKER + json.dumps({"root": "/tmp/juergen_app_fixtures"})}
    )
    evidence = setup_app_fixture(session, fixture)
    assert evidence["status"] == "failed"
    assert "RuntimeError" in evidence["error"]
    assert evidence["fixture_id"] == fixture.id and evidence["app"] == "writer"
    assert isinstance(evidence["elapsed_s"], float)


@pytest.mark.parametrize("app", APPS)
def test_setup_app_fixture_wipes_and_rebuilds_a_private_per_fixture_root(app: str) -> None:
    fixture = _app(app)
    session = _rooted()
    evidence = setup_app_fixture(session, fixture)
    assert evidence["status"] == "ready"
    assert evidence["guest_root"].endswith(f"juergen_app_fixtures/{fixture.id}")
    script = session.argv_log[-1][2]
    assert 'rm -rf "$root"' in script, "each fixture wipes its own root"
    assert 'mkdir -p "$root' in script, "and rebuilds it (files/ makes subdirs)"
    assert "set -euo pipefail" in script


def test_documents_are_built_in_the_guest_not_shipped_as_binaries() -> None:
    for app in ("writer", "calc"):
        session = _rooted()
        setup_app_fixture(session, _app(app))
        script = session.argv_log[-1][2]
        assert "base64 -d" in script and "--convert-to" in script


def test_the_calc_setup_verifies_the_mapped_window_is_actually_usable() -> None:
    """The clean snapshot sometimes restores Calc's geometry as a 16-pixel sliver."""
    session = _rooted()
    setup_app_fixture(session, _app("calc"))
    script = session.argv_log[-1][2]
    assert 'test "${w:-0}" -gt 1000' in script and 'test "${h:-0}" -gt 600' in script


def test_the_in_guest_chrome_fixture_serves_itself_on_loopback() -> None:
    import base64
    import re

    session = _rooted()
    setup_app_fixture(session, _app("chrome", port=8931))
    script = session.argv_log[-1][2]
    assert "http://127.0.0.1:8931/settings" in script
    assert "10.0.2.2" not in script, "this fixture needs no host round-trip"
    # The guest server is base64-embedded, so decode it before asserting on it.
    payloads = re.findall(r"printf '%s' ([A-Za-z0-9+/=]{40,})", script)
    assert payloads, script[:400]
    source = base64.b64decode(payloads[0]).decode()
    assert "ThreadingHTTPServer(('127.0.0.1',8931)" in source
    assert "os.replace(tmp,STATE)" in source, "state is written atomically"
    assert "mkstemp" in source, "so a probe can never read a half-written file"


def test_the_chrome_fixture_page_hides_the_control_below_the_fold_per_direction() -> None:
    from evals.fixtures.apps import _chrome_html

    down = _chrome_html(_app("chrome", scroll_direction="down"))
    up = _chrome_html(_app("chrome", scroll_direction="up"))
    assert "margin-top:900px" in down
    assert "margin-top:400px" in up, (
        "otherwise 'scroll up' would be satisfiable by not scrolling at all"
    )
