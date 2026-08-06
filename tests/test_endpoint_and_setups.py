"""The default transport, the remaining in-guest setups, and the RL preparers.

`EndpointTransport` is the **default** — it posts to verifiers' interception endpoint,
which is the path that commits the turn to the trace graph, so losing it loses tokens,
logprobs and branch structure. It is driven here against a fake OpenAI-shaped HTTP
server rather than mocked, because the thing worth testing is the wire body.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import verifiers.v1 as vf

from agent.agent import EndpointTransport, ModelCallError, load_codec
from agent.history import History, ImageBudget, StatelessSingleTurn
from juergen_doubles import FakeSession, make_ctx, make_task_data, png


# --------------------------------------------------------------------------- #
# a fake OpenAI-shaped endpoint
# --------------------------------------------------------------------------- #


class FakeOpenAI:
    """Serves `/chat/completions` and records every request body."""

    def __init__(self, *, reply: str = "0 0 0 ;", status: int = 200) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                outer.headers.append(dict(self.headers))
                try:
                    outer.requests.append(json.loads(raw))
                except json.JSONDecodeError:
                    outer.requests.append({"__raw__": raw.decode("utf-8", "replace")})
                if status != 200:
                    body = json.dumps({"error": {"message": "upstream exploded"}}).encode()
                    self.send_response(status)
                else:
                    body = json.dumps(
                        {
                            "id": "cmpl-1",
                            "object": "chat.completion",
                            "created": 0,
                            "model": "test-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": reply},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def endpoint():
    servers: list[FakeOpenAI] = []

    def build(**kwargs) -> FakeOpenAI:
        server = FakeOpenAI(**kwargs)
        servers.append(server)
        return server

    yield build
    for server in servers:
        server.close()


def _history() -> History:
    history = History(n_history_frames=4)
    history.start(png())
    return history


# --------------------------------------------------------------------------- #
# EndpointTransport
# --------------------------------------------------------------------------- #


def test_the_endpoint_transport_posts_and_returns_the_content(endpoint) -> None:
    server = endpoint(reply="10 20 0 ; +LMB -LMB")

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="sk-test")
        try:
            return await transport.complete(
                make_ctx(model="my-model"),
                {"messages": [vf.UserMessage(content="hi")], "max_tokens": 32},
                session_id="t1",
            )
        finally:
            await transport.close()

    assert asyncio.run(body()) == "10 20 0 ; +LMB -LMB"
    request = server.requests[0]
    assert request["model"] == "my-model", "ctx.model is what goes on the wire"
    assert request["max_tokens"] == 32
    assert request["messages"][0]["role"] == "user"


def test_the_secret_is_sent_as_a_bearer_token(endpoint) -> None:
    server = endpoint()

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="sk-abc123")
        try:
            await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    asyncio.run(body())
    assert server.headers[0]["Authorization"] == "Bearer sk-abc123"


def test_only_the_unset_knobs_reach_the_wire_through_the_endpoint(endpoint) -> None:
    """The proxy applies ctx.sampling on top, so sending ours would be a lie."""
    from agent.agent import Agent

    server = endpoint()
    agent = Agent(
        codec=load_codec("deltatype_v2"),
        policy=StatelessSingleTurn(),
        budget=ImageBudget(max_images=1),
        transport=EndpointTransport(endpoint=server.base_url, secret="s"),
        max_tokens=64,
        temperature=0.0,
    )
    async def body():
        try:
            return await agent.step(
                make_ctx(model="m", temperature=0.9),
                history=_history(),
                instruction="do it",
                step=1,
                geometry=_geometry(),
                cursor=(0, 0),
            )
        finally:
            await agent.close()

    decision = asyncio.run(body())
    request = server.requests[0]
    assert "temperature" not in request, "the eval set it, so the harness must not send it"
    assert request["max_tokens"] == 64, "the eval left this unset, so the harness fills in"
    assert decision.sampling.temperature == 0.9
    assert decision.sampling.temperature_source == "ctx.sampling"


def test_an_image_rides_the_endpoint_body_as_a_data_url(endpoint) -> None:
    from agent.agent import Agent

    server = endpoint()
    agent = Agent(
        codec=load_codec("deltatype_v2"),
        policy=StatelessSingleTurn(),
        budget=ImageBudget(max_images=1),
        transport=EndpointTransport(endpoint=server.base_url, secret="s"),
    )
    async def body():
        try:
            await agent.step(
                make_ctx(),
                history=_history(),
                instruction="do it",
                step=1,
                geometry=_geometry(),
                cursor=(0, 0),
            )
        finally:
            await agent.close()

    asyncio.run(body())
    parts = server.requests[0]["messages"][1]["content"]
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_an_http_failure_becomes_a_model_call_error(endpoint) -> None:
    server = endpoint(status=500)

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        try:
            await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    with pytest.raises(ModelCallError):
        asyncio.run(body())


def test_an_unreachable_endpoint_becomes_a_model_call_error() -> None:
    async def body():
        transport = EndpointTransport(
            endpoint="http://127.0.0.1:1/v1", secret="s", timeout_s=2.0
        )
        try:
            await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    with pytest.raises(ModelCallError):
        asyncio.run(body())


def test_the_client_is_reused_across_turns_and_closed_once(endpoint) -> None:
    server = endpoint()
    seen = {}

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        ctx = make_ctx()
        await transport.complete(ctx, {"messages": []}, session_id=None)
        seen["first"] = transport._client
        await transport.complete(ctx, {"messages": []}, session_id=None)
        seen["second"] = transport._client
        await transport.close()
        seen["after_close"] = transport._client
        await transport.close()  # idempotent

    asyncio.run(body())
    assert seen["second"] is seen["first"], "one AsyncOpenAI per transport, not per turn"
    assert seen["after_close"] is None


def test_an_empty_completion_is_an_empty_string_not_none(endpoint) -> None:
    server = endpoint(reply="")

    async def body():
        transport = EndpointTransport(endpoint=server.base_url, secret="s")
        try:
            return await transport.complete(make_ctx(), {"messages": []}, session_id=None)
        finally:
            await transport.close()

    assert asyncio.run(body()) == ""


def _geometry():
    from pixeldesk.geometry import DisplayGeometry

    return DisplayGeometry(desktop_width=1920, desktop_height=1080)


# --------------------------------------------------------------------------- #
# the remaining three sign-of-life setups
# --------------------------------------------------------------------------- #


_GEOMETRY = "SOLV2_GEOMETRY=" + json.dumps(
    {"window_id": "0x1", "x": 80, "y": 120, "width": 1120, "height": 720, "window_line": "x"}
)
_STATE = json.dumps(
    {
        "schema_version": 1,
        "task_id": "cell",
        "active_window": "xterm",
        "windows": "",
        "chrome_process": False,
        "history": None,
        "transcript": None,
        "prompt_count": 0,
        "capture_file_exists": False,
        "captured_text": None,
        "proof_file_exists": False,
        "proof_file_content": None,
    }
)


def _guest_session(active: str = "xmessage") -> FakeSession:
    return FakeSession(
        argv_responses={
            "SOLV2_GEOMETRY": _GEOMETRY,
            "wmctrl -lGx": _GEOMETRY,
            "python3": _GEOMETRY,
            "xprop -root": active,
        }
    )


def test_the_terminal_command_setup_seeds_its_own_listing_anchor() -> None:
    import evals.signoflife.guest as guest

    session = _guest_session()
    task = make_task_data(
        kind="terminal_command",
        name="terminal_ls",
        expected={"command": "ls", "listing_marker": "anchor_7c39.txt"},
    )
    evidence = guest._setup_terminal_command(session, task)
    script = session.argv_log[0][2]
    assert "anchor_7c39.txt" in script, "the marker file is created by setup, not by luck"
    assert "HISTFILE=" in script and "PS1='SOLV2-LS$ '" in script
    assert "tee -a" in script, "the transcript is captured for the oracle"
    assert evidence["title"] == "SOLV2 terminal_ls"
    assert evidence["window"]["width"] == 1120


def test_the_open_chrome_setup_kills_any_running_chrome_first() -> None:
    """Otherwise the cell starts solved."""
    import evals.signoflife.guest as guest

    session = FakeSession(
        argv_responses={"python3": f"SOLV2_STATE={_STATE}", "pkill": "", "bash": ""}
    )
    task = make_task_data(
        kind="open_chrome",
        name="desktop_open_chrome",
        expected={"active_window_class_any": ["chrome"]},
    )
    evidence = guest._setup_open_chrome(session, task)
    script = session.argv_log[0][2]
    for name in ("chrome", "google-chrome", "chromium", "chromium-browser"):
        assert f"pkill -x {name}" in script, name
    assert "moveTo(960,540)" in script, "the cursor starts at screen centre"
    assert evidence["chrome_absent_before"] is True
    assert evidence["dock_chrome_coordinate"] == list(guest.DOCK_CHROME_COORDINATE)


def test_the_compound_setup_refuses_to_proceed_with_the_terminal_focused() -> None:
    """The whole point of the cell is that the model must click to focus first."""
    import evals.signoflife.guest as guest

    session = _guest_session(active="gnome-terminal-server")
    task = make_task_data(
        kind="focus_terminal_and_type",
        name="focus_terminal_and_type",
        expected={"command": "printf x > /tmp/p", "file": "/tmp/p", "content": "x"},
    )
    with pytest.raises(RuntimeError, match="did not remove terminal focus"):
        guest._setup_focus_terminal_and_type(session, task)


def test_the_compound_setup_reports_the_terminal_click_coordinate() -> None:
    import evals.signoflife.guest as guest

    session = _guest_session(active="xmessage SOLV2 desktop note")
    task = make_task_data(
        kind="focus_terminal_and_type",
        name="focus_terminal_and_type",
        expected={"command": "printf x > /tmp/p", "file": "/tmp/p", "content": "x"},
    )
    evidence = guest._setup_focus_terminal_and_type(session, task)
    assert evidence["terminal_click_coordinate"] == [80 + 1120 // 2, 120 + 100]
    script = session.argv_log[0][2]
    assert "xmessage" in script and "nautilus" in script, "a fallback if xmessage is absent"


def test_a_missing_window_geometry_marker_fails_closed() -> None:
    import evals.signoflife.guest as guest

    session = FakeSession(argv_responses={"python3": "no marker at all"})
    with pytest.raises(RuntimeError, match="window geometry evidence missing"):
        guest._window_geometry(session, "SOLV2 nothing")


def test_the_active_window_probe_refuses_a_zero_window_id() -> None:
    import evals.signoflife.guest as guest

    script = guest._active_window_script()
    assert 'test "$wid" != "0x0"' in script, "an unmapped desktop must not read as focused"


# --------------------------------------------------------------------------- #
# the RL preparers
# --------------------------------------------------------------------------- #


def test_the_movebox_preparer_installs_the_scene_on_a_virtual_desktop(tmp_path) -> None:
    from rl.desktop import VirtualDesktop
    from rl.movebox.harness import MoveBoxPreparer

    background = tmp_path / "bg.png"
    background.write_bytes(png(400, 300))
    desktop = VirtualDesktop()
    task = make_task_data(
        kind="movebox",
        bbox=(100, 100, 200, 200),
        cursor_start=(10, 10),
        setup={
            "background_path": str(background),
            "band": "near",
            "start_distance": 120.0,
            "screen": [400, 300],
        },
    )
    evidence = MoveBoxPreparer().prepare(desktop, task)
    assert evidence == {
        "band": "near",
        "start_distance": 120.0,
        "cursor_start": [10, 10],
        "box": [100, 100, 200, 200],
    }
    assert desktop.screen_size() == (400, 300)
    assert desktop.cursor_position() == (10, 10)
    probe = MoveBoxPreparer().probe(desktop, task)
    assert probe["in_bbox"] is False and probe["postcondition_success"] is False
    desktop.execute_atomic([{"kind": "move_to", "args": (150, 150)}])
    solved = MoveBoxPreparer().probe(desktop, task)
    assert solved["in_bbox"] is True and solved["postcondition_success"] is True


def test_the_movebox_preparer_refuses_a_real_session() -> None:
    from rl.movebox.harness import MoveBoxPreparer

    with pytest.raises(TypeError, match="requires a virtual desktop"):
        MoveBoxPreparer().prepare(FakeSession(), make_task_data(kind="movebox"))


def test_the_grounding_canvas_preparer_loads_the_labelled_screenshot(tmp_path) -> None:
    from rl.desktop import VirtualDesktop
    from rl.grounding.harness import GroundingCanvasPreparer

    image = tmp_path / "step_001.png"
    image.write_bytes(png(320, 240))
    desktop = VirtualDesktop()
    task = make_task_data(
        kind="grounding_canvas",
        bbox=(50, 50, 100, 100),
        regime="near",
        cursor_start=(5, 5),
        setup={"image_path": str(image), "screen": [320, 240]},
    )
    evidence = GroundingCanvasPreparer().prepare(desktop, task)
    assert evidence == {"regime": "near", "cursor_start": [5, 5], "bbox": [50, 50, 100, 100]}
    assert desktop.screen_size() == (320, 240) and desktop.cursor_position() == (5, 5)
    probe = GroundingCanvasPreparer().probe(desktop, task)
    assert "postcondition_success" not in probe, (
        "this env deliberately does not stop on a hit — the frame is a final state"
    )
    assert probe["in_bbox"] is False and probe["distance"] > 0


def test_the_grounding_canvas_preparer_refuses_a_real_session() -> None:
    from rl.grounding.harness import GroundingCanvasPreparer

    with pytest.raises(TypeError, match="requires a virtual desktop"):
        GroundingCanvasPreparer().prepare(FakeSession(), make_task_data(kind="grounding_canvas"))


def test_the_target_box_preparer_places_the_real_cursor_and_annotates_every_frame() -> None:
    from rl.target_box.harness import TargetBoxPreparer

    session = FakeSession(screen=(1920, 1080))
    task = make_task_data(
        kind="target_box",
        name="target_box/t0",
        setup={
            "screen": [1920, 1080],
            "instance_key": "t0:/p/t0.json",
            "box": {},
            "config": [{"type": "launch"}],
        },
    )
    preparer = TargetBoxPreparer()
    evidence = preparer.prepare(session, task)
    assert evidence["screen"] == [1920, 1080]
    assert session.cursor == tuple(evidence["cursor_start"])
    assert any("moveTo" in code for code in session.pyautogui_log)
    annotated = preparer.observe(png(1920, 1080), task)
    assert annotated != png(1920, 1080), "the box is drawn on every observation"
    probe = preparer.probe(session, task)
    assert probe["postcondition_success"] is False, (
        "success needs the model to declare it, so entering the box does not end it"
    )
    assert probe["box"] == evidence["box"]


def test_the_target_box_preparer_refuses_a_screen_size_mismatch() -> None:
    """A misconfigured screen is silently a different task."""
    from rl.target_box.harness import TargetBoxPreparer

    session = FakeSession(screen=(1280, 720))
    task = make_task_data(
        kind="target_box",
        setup={"screen": [1920, 1080], "instance_key": "k", "box": {}},
    )
    with pytest.raises(ValueError, match="does not match the configured"):
        TargetBoxPreparer().prepare(session, task)


def test_the_target_box_scene_is_stable_across_prepare_observe_and_probe() -> None:
    """All three derive the box from the row; a drift would annotate one box and
    score another."""
    from rl.target_box.harness import TargetBoxPreparer, _scene

    task = make_task_data(
        kind="target_box", setup={"screen": [1920, 1080], "instance_key": "k:p", "box": {}}
    )
    boxes = {_scene(task)[0] for _ in range(3)}
    assert len(boxes) == 1
    session = FakeSession(screen=(1920, 1080))
    preparer = TargetBoxPreparer()
    prepared = preparer.prepare(session, task)["box"]
    assert preparer.probe(session, task)["box"] == prepared


# --------------------------------------------------------------------------- #
# movebox background loading
# --------------------------------------------------------------------------- #


def test_list_backgrounds_is_sorted_and_refuses_an_empty_directory(tmp_path) -> None:
    from rl.movebox.dataset import list_backgrounds

    with pytest.raises(FileNotFoundError, match="no \\*.png backgrounds"):
        list_backgrounds(str(tmp_path))
    for name in ("b.png", "a.png", "c.txt"):
        (tmp_path / name).write_bytes(png() if name.endswith(".png") else b"x")
    found = list_backgrounds(str(tmp_path))
    assert [Path(p).name for p in found] == ["a.png", "b.png"], "sorted, and PNGs only"


def test_load_canvas_resizes_the_background_and_draws_the_box(tmp_path) -> None:
    import io

    from PIL import Image

    from rl.movebox.dataset import MoveBoxScene, load_canvas

    background = tmp_path / "bg.png"
    background.write_bytes(png(80, 60, colour=(0, 0, 0)))
    scene = MoveBoxScene(
        idx=0,
        background_path=str(background),
        box=(10, 10, 40, 40),
        cursor_start=(1, 1),
        screen_w=200,
        screen_h=150,
        band="near",
        start_distance=30.0,
    )
    canvas = load_canvas(scene)
    assert canvas.size == (200, 150), "the background is resized to the scene's screen"
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    with Image.open(io.BytesIO(buffer.getvalue())) as handle:
        assert handle.convert("RGB").load()[10, 10] == (0, 255, 0)


# --------------------------------------------------------------------------- #
# the fixture preparers
# --------------------------------------------------------------------------- #


def test_the_web_fixture_preparer_serves_launches_and_reads_back(monkeypatch) -> None:
    import evals.fixtures.preparers as preparers
    from evals.fixtures.chrome import ChromeLaunch

    saved_server = preparers._SERVER
    preparers._SERVER = None
    try:
        launched: list = []

        def fake_launch(session, server, fixture):
            launched.append(fixture.id)
            server.store.reset(fixture)
            server.store.apply(
                fixture.id,
                {
                    "generation": server.store.snapshot(fixture.id)["generation"],
                    "kind": "ready",
                    "geometry": {"viewport": {"width": 1280, "height": 700}},
                },
            )
            return ChromeLaunch(
                status="ready", attempts=[], generation=1, url="u", ready_state={}
            )

        monkeypatch.setattr(preparers, "launch_chrome", fake_launch)
        task = make_task_data(
            kind="web_fixture",
            name="cell_click",
            setup={
                "fixture": {
                    "id": "fx1",
                    "template": "click",
                    "params": {"label": "Enable", "left": 400, "top": 300},
                }
            },
        )
        preparer = preparers.WebFixturePreparer()
        evidence = preparer.prepare(FakeSession(), task)
        assert evidence == {"fixture": "fx1", "template": "click", "chrome": ChromeLaunch(
            status="ready", attempts=[], generation=1, url="u", ready_state={}
        ).as_dict()}
        assert launched == ["fx1"]
        probe = preparer.probe(FakeSession(), task)
        assert probe["ready"] is True and probe["generation"] == 1
        assert probe["postcondition_status"] == "ok"
        assert probe["current"] == {"kind": "click", "checked": False, "decoy_checked": False}
        assert probe["geometry"]["viewport"]["width"] == 1280
        assert "browser" not in probe, "CDP is only read when a port is configured"
    finally:
        if preparers._SERVER is not None:
            preparers._SERVER.close()
        preparers._SERVER = saved_server


def test_a_cdp_failure_does_not_lose_the_primary_http_state(monkeypatch) -> None:
    """The HTTP state is the primary read; CDP is the second, independent one."""
    import evals.fixtures.preparers as preparers

    saved_server = preparers._SERVER
    preparers._SERVER = None
    try:
        monkeypatch.setattr(
            preparers,
            "capture_browser_diagnostics",
            lambda fixture, port: (_ for _ in ()).throw(RuntimeError("no chrome")),
        )
        task = make_task_data(
            kind="web_fixture",
            name="cell_click",
            setup={
                "fixture": {"id": "fx2", "template": "click", "params": {"label": "x", "left": 1, "top": 1}},
                "chromium_port": 9222,
            },
        )
        preparer = preparers.WebFixturePreparer()
        server = preparers.web_fixture_server(preparers._web_fixture(task))
        server.store.reset(preparers._web_fixture(task))
        probe = preparer.probe(FakeSession(), task)
        assert probe["browser"]["status"] == "unavailable"
        assert "no chrome" in probe["browser"]["error"]
        assert probe["ready"] is False, "the HTTP state still came through"
    finally:
        if preparers._SERVER is not None:
            preparers._SERVER.close()
        preparers._SERVER = saved_server


def test_the_app_fixture_preparer_sets_up_and_probes() -> None:
    import evals.fixtures.preparers as preparers
    from evals.fixtures.apps import JSON_MARKER

    root = JSON_MARKER + json.dumps({"root": "/tmp/juergen_app_fixtures"})
    state = JSON_MARKER + json.dumps(
        {"schema_version": 1, "fixture_id": "fx_writer", "app": "writer", "root": "/tmp/x"}
    )
    session = FakeSession(argv_responses={"juergen_app_fixtures'": root, "app=": state})
    task = make_task_data(
        kind="app_fixture",
        name="cell_writer",
        setup={
            "fixture": {
                "id": "fx_writer",
                "app": "writer",
                "params": {"file_name": "doc.odt", "initial_text": "hello"},
            }
        },
    )
    preparer = preparers.AppFixturePreparer()
    evidence = preparer.prepare(session, task)
    assert evidence["status"] == "ready" and evidence["app"] == "writer"
    probe = preparer.probe(session, task)
    assert probe["task_id"] == "cell_writer" and probe["postcondition_status"] == "ok"
    assert probe["app"] == "writer"


def test_a_fixture_spec_falls_back_to_the_task_name_for_its_id() -> None:
    import evals.fixtures.preparers as preparers

    task = make_task_data(
        kind="web_fixture", name="named_cell", setup={"fixture": {"template": "scroll", "params": {"label": "S"}}}
    )
    assert preparers._web_fixture(task).id == "named_cell"
    unnamed = make_task_data(
        kind="web_fixture", name=None, idx=7, setup={"fixture": {"template": "scroll", "params": {"label": "S"}}}
    )
    assert preparers._web_fixture(unnamed).id == "fixture_7"


def test_a_fixture_spec_with_no_template_is_refused() -> None:
    import evals.fixtures.preparers as preparers

    task = make_task_data(kind="web_fixture", setup={"fixture": {}})
    with pytest.raises(KeyError):
        preparers._web_fixture(task)


def test_capture_chrome_log_never_raises() -> None:
    from evals.fixtures.chrome import capture_chrome_log

    class Angry(FakeSession):
        def execute_argv(self, argv):
            raise OSError("guest gone")

    assert capture_chrome_log(Angry())["status"] == "unavailable"
    ok = capture_chrome_log(FakeSession(argv_responses={"tail": "log tail here"}))
    assert ok == {"status": "captured", "log": "log tail here"}
