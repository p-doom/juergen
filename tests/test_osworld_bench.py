"""The OSWorld benchmark family, end to end without OSWorld or a VM.

`DesktopFacade` is the only session a real VM ever hands the harness, and
`osworld_modules()` is the one seam that imports OSWorld, so substituting it here
exercises every line of the binding and of the ported `DesktopEnv.evaluate`
arithmetic against fakes. A difference from upstream is a difference from the
published benchmark, so the conjunction, the mean-vs-max and the FAIL inversion are
pinned by value.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.osworld import OSWorldBridge, OSWorldNotAvailable, osworld_root
from evals.vm import DesktopFacade
from juergen_doubles import FakeGuestReceipt


class _Getters:
    """OSWorld's `evaluators.getters`. Each `get_<type>` takes `(env, config)`."""

    @staticmethod
    def get_literal(env: Any, config: dict[str, Any]) -> Any:
        return config["value"]

    @staticmethod
    def get_from_guest(env: Any, config: dict[str, Any]) -> Any:
        # Reads the env, which is why getters take one.
        return env.controller.run(config["command"])

    @staticmethod
    def get_missing_file(env: Any, config: dict[str, Any]) -> Any:
        raise FileNotFoundError(config.get("path", "?"))


class _Metrics:
    @staticmethod
    def equals(result: Any, expected: Any = None, **options: Any) -> float:
        if options.get("casefold"):
            return 1.0 if str(result).casefold() == str(expected).casefold() else 0.0
        return 1.0 if result == expected else 0.0

    @staticmethod
    def truthy(result: Any, **options: Any) -> float:
        return 1.0 if result else 0.0

    @staticmethod
    def half(result: Any, **options: Any) -> float:
        return 0.5

    @staticmethod
    def file_bytes(result: str, **options: Any) -> float:
        path = Path(result)
        return float(
            path.name == options["name"]
            and path.read_bytes() == options["content"].encode()
        )

    @staticmethod
    def infeasible() -> float:
        """OSWorld really does bind a metric named `infeasible`
        (`evaluators/metrics/__init__.py:169`) even though `evaluate()` never
        calls it — the binding has to survive, so the fake has to have it."""
        return 0.0


class _SetupController:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[list[dict[str, Any]]] = []
        self.cache_dirs: list[str] = []

    def reset_cache_dir(self, cache_dir: str) -> None:
        self.cache_dirs.append(cache_dir)

    def setup(self, config: list[dict[str, Any]], use_proxy: bool = False) -> bool:
        self.calls.append(list(config))
        return True


class _PythonController:
    def __init__(self, *, vm_ip: str, server_port: int) -> None:
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.commands: list[str] = []

    def run(self, command: str) -> str:
        self.commands.append(command)
        return f"ran:{command}"


def _loader() -> tuple[Any, Any, Any, Any]:
    return _SetupController, _PythonController, _Getters, _Metrics


def _bridge(tmp_path: Path, **kwargs: Any) -> OSWorldBridge:
    return OSWorldBridge(
        base_url="http://127.0.0.1:5111",
        cache_dir=tmp_path / "cache",
        loader=_loader,
        **kwargs,
    )


def test_an_unset_osworld_root_names_the_variable_not_a_submodule(monkeypatch) -> None:
    monkeypatch.delenv("OSWORLD_ROOT", raising=False)
    with pytest.raises(OSWorldNotAvailable) as excinfo:
        osworld_root()
    assert "OSWORLD_ROOT" in str(excinfo.value)


def test_a_root_that_is_not_an_osworld_checkout_is_rejected(tmp_path, monkeypatch) -> None:
    """The Jul-23 re-clone left a tree that looked like a path and behaved like a
    missing submodule three layers down."""
    monkeypatch.setenv("OSWORLD_ROOT", str(tmp_path))
    with pytest.raises(OSWorldNotAvailable) as excinfo:
        osworld_root()
    assert "evaluators" in str(excinfo.value)


def test_a_real_looking_root_resolves(tmp_path, monkeypatch) -> None:
    # OSWorld's own layout, not ours: `desktop_env/` here is xlang-ai's.
    (tmp_path / "desktop_env" / "evaluators").mkdir(parents=True)
    monkeypatch.setenv("OSWORLD_ROOT", str(tmp_path))
    assert osworld_root() == tmp_path


def test_setup_runs_the_config_steps_and_binds_the_evaluator(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    steps = bridge.setup(
        {
            "id": "chrome/task-1",
            "instruction": "do it",
            "config": [{"type": "launch"}, {"type": "open"}],
            "evaluator": {"func": "truthy", "result": {"type": "literal", "value": 1}},
        }
    )
    assert steps == 2
    assert bridge.setup_controller.calls == [[{"type": "launch"}, {"type": "open"}]]
    assert bridge.task_id == "chrome/task-1"
    assert Path(bridge.cache_dir).is_dir()
    assert bridge.evaluate() == 1.0


def test_unstaged_network_assets_are_rejected_before_setup(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    with pytest.raises(ValueError, match="unstaged download"):
        bridge.setup(
            {
                "id": "task-1",
                "config": [
                    {
                        "type": "download",
                        "parameters": {
                            "files": [
                                {
                                    "url": "https://example.test/input.txt",
                                    "path": "/tmp/input.txt",
                                }
                            ]
                        },
                    }
                ],
            }
        )
    assert bridge._setup_controller is None


def test_unstaged_cloud_evaluators_are_rejected_before_setup(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    with pytest.raises(ValueError, match="unstaged cloud_file"):
        bridge.setup(
            {
                "id": "task-1",
                "evaluator": {
                    "func": "file_bytes",
                    "result": {
                        "type": "cloud_file",
                        "path": "https://example.test/gold.txt",
                        "dest": "gold.txt",
                    },
                },
            }
        )
    assert bridge._setup_controller is None


def test_a_task_with_no_evaluator_binds_but_cannot_be_scored(tmp_path) -> None:
    """The grounding family: OSWorld *setup*, our own bbox oracle. Forcing it to
    carry a benchmark evaluator would invent a dependency it does not have."""
    bridge = _bridge(tmp_path)
    assert bridge.setup({"id": "t", "config": [{"type": "launch"}]}) == 1
    with pytest.raises(OSWorldNotAvailable) as excinfo:
        bridge.evaluate()
    assert "evaluator" in str(excinfo.value)


def test_the_ports_and_ip_come_off_the_base_url(tmp_path) -> None:
    bridge = _bridge(tmp_path, chromium_port=9333, vlc_port=8111)
    assert (bridge.vm_ip, bridge.server_port) == ("127.0.0.1", 5111)
    assert bridge.controller.server_port == 5111
    assert bridge.setup_controller.kwargs["chromium_port"] == 9333
    assert bridge.setup_controller.kwargs["vlc_port"] == 8111


def test_a_length_mismatch_between_metrics_and_getters_is_refused(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        bridge.bind(
            {
                "id": "t",
                "evaluator": {
                    "func": ["truthy", "truthy"],
                    "result": [{"type": "literal", "value": 1}],
                },
            }
        )
    assert "result getters" in str(excinfo.value)


def test_a_scalar_evaluator_with_an_expected_getter_and_options(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {
            "id": "t",
            "evaluator": {
                "func": "equals",
                "result": {"type": "literal", "value": "OK"},
                "expected": {"type": "literal", "value": "ok"},
                "options": {"casefold": True},
            },
        }
    )
    assert bridge.evaluate() == 1.0


def test_the_getter_is_handed_the_bridge_as_its_env(tmp_path) -> None:
    """106 of OSWorld's getter env-reads are `env.controller`; the bridge is the
    env, which is the only reason the getters work outside `DesktopEnv`."""
    bridge = _bridge(tmp_path)
    bridge.bind(
        {
            "id": "t",
            "evaluator": {
                "func": "equals",
                "result": {"type": "from_guest", "command": "pgrep chrome"},
                "expected": {"type": "literal", "value": "ran:pgrep chrome"},
            },
        }
    )
    assert bridge.evaluate() == 1.0
    assert bridge.controller.commands == ["pgrep chrome"]


def test_an_and_conjunction_short_circuits_to_zero(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["truthy", "truthy"],
                "conj": "and",
                "result": [
                    {"type": "literal", "value": 0},
                    {"type": "literal", "value": 1},
                ],
            },
        }
    )
    assert bridge.evaluate() == 0.0


def test_an_and_conjunction_of_partial_scores_is_the_mean(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["truthy", "half"],
                "conj": "and",
                "result": [
                    {"type": "literal", "value": 1},
                    {"type": "literal", "value": 1},
                ],
            },
        }
    )
    assert bridge.evaluate() == pytest.approx(0.75)


def test_an_or_conjunction_short_circuits_to_one(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["truthy", "truthy"],
                "conj": "or",
                "result": [
                    {"type": "literal", "value": 0},
                    {"type": "literal", "value": 1},
                ],
            },
        }
    )
    assert bridge.evaluate() == 1.0


def test_an_or_conjunction_with_no_winner_is_the_max(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["half", "truthy"],
                "conj": "or",
                "result": [
                    {"type": "literal", "value": 1},
                    {"type": "literal", "value": 0},
                ],
            },
        }
    )
    assert bridge.evaluate() == 0.5


def test_a_missing_file_under_and_is_zero_and_under_or_is_skipped(tmp_path) -> None:
    """Upstream's `and` branch verbatim. Its `or` branch cannot produce a number
    at all (it reuses an unbound `result_state`), so skipping is the deviation
    recorded in `evals/osworld.py`."""
    conjoined = _bridge(tmp_path)
    conjoined.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["truthy", "truthy"],
                "conj": "and",
                "result": [{"type": "missing_file"}, {"type": "literal", "value": 1}],
            },
        }
    )
    assert conjoined.evaluate() == 0.0

    disjoined = _bridge(tmp_path)
    disjoined.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["truthy", "half"],
                "conj": "or",
                "result": [{"type": "missing_file"}, {"type": "literal", "value": 1}],
            },
        }
    )
    assert disjoined.evaluate() == 0.5

    everything_missing = _bridge(tmp_path)
    everything_missing.bind(
        {
            "id": "t",
            "evaluator": {
                "func": ["truthy"],
                "conj": "or",
                "result": [{"type": "missing_file"}],
            },
        }
    )
    assert everything_missing.evaluate() == 0.0, "no metric read is not a pass"


def test_a_missing_file_in_a_scalar_evaluator_is_zero(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {"id": "t", "evaluator": {"func": "truthy", "result": {"type": "missing_file"}}}
    )
    assert bridge.evaluate() == 0.0


def test_a_postconfig_runs_before_the_getters(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.setup(
        {
            "id": "t",
            "config": [{"type": "launch"}],
            "evaluator": {
                "func": "truthy",
                "result": {"type": "literal", "value": 1},
                "postconfig": [{"type": "save"}],
            },
        }
    )
    assert bridge.evaluate() == 1.0
    assert bridge.setup_controller.calls == [[{"type": "launch"}], [{"type": "save"}]]


def test_declaring_fail_on_an_infeasible_task_is_the_success_condition(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind({"id": "t", "evaluator": {"func": "infeasible"}})
    assert bridge.evaluate() == 0.0, "silence on an infeasible task is not a pass"
    bridge.declare_terminal("fail")
    assert bridge.evaluate() == 1.0


def test_declaring_fail_on_a_feasible_task_forfeits(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind(
        {"id": "t", "evaluator": {"func": "truthy", "result": {"type": "literal", "value": 1}}}
    )
    assert bridge.evaluate() == 1.0
    bridge.declare_terminal("fail")
    assert bridge.evaluate() == 0.0


def test_a_plain_terminate_is_not_a_fail(tmp_path) -> None:
    bridge = _bridge(tmp_path)
    bridge.bind({"id": "t", "evaluator": {"func": "infeasible"}})
    bridge.declare_terminal("terminate")
    bridge.declare_terminal(None)
    assert bridge.evaluate() == 0.0


class _Transport:
    base_url = "http://127.0.0.1:5222"

    def __init__(self) -> None:
        self.argv: list[list[str]] = []

    def screen_size(self) -> tuple[int, int]:
        return (1280, 800)

    def cursor_position(self) -> tuple[int, int]:
        return (5, 6)

    def execute_atomic(self, operations: Any) -> FakeGuestReceipt:
        return FakeGuestReceipt(ok=True, cursor_before=(5, 6), cursor_after=(5, 6))

    def execute_argv(self, argv: list[str], *, check: bool = True) -> dict[str, Any]:
        self.argv.append(list(argv))
        return {"output": "", "check": check}

    def execute_pyautogui(self, code: str) -> None:
        self.argv.append(["<pyautogui>", code])


class _Client:
    def __init__(self) -> None:
        self.settled: list[dict[str, float]] = []

    def screenshot(self) -> bytes:
        return b"png"

    def screenshot_settled(self, **kwargs: float) -> bytes:
        self.settled.append(dict(kwargs))
        return b"settled"


class _Ports:
    chromium = 9444
    vlc = 8222


class _Runtime:
    def state(self) -> Any:
        return type("State", (), {"ports": _Ports()})()


class _Session:
    def __init__(self, *, runtime: Any | None = None) -> None:
        self.transport = _Transport()
        self.client = _Client()
        self.runtime = runtime


class _Checkout:
    session_id = "sess-1"

    def __init__(self) -> None:
        self.touches = 0
        self.released: list[tuple[bool, str | None]] = []

    def touch(self) -> None:
        self.touches += 1

    def release(self, *, failed: bool = False, error: str | None = None) -> None:
        self.released.append((failed, error))


def _facade(tmp_path: Path, *, runtime: Any | None = None) -> tuple[DesktopFacade, _Checkout]:
    checkout = _Checkout()
    # A QEMU-backed runtime by default: it is the only backing the production
    # selector can reach (`evals/signoflife/__main__.py` -> `kvm_desktop_pool` ->
    # `qemu_session_factory`), and the forwarded guest ports come off it.
    facade = DesktopFacade(
        checkout, _Session(runtime=runtime or _Runtime()), osworld_cache_dir=tmp_path / "cache"
    )
    facade._osworld_bridge = OSWorldBridge(
        base_url="http://127.0.0.1:5222",
        cache_dir=tmp_path / "cache",
        loader=_loader,
        **facade._guest_ports(),
    )
    return facade, checkout


def test_the_facade_still_refuses_to_be_a_catch_all_proxy() -> None:
    """The harness probes optional capabilities with `getattr`, so anything the
    facade does not name must read as absent."""
    facade = DesktopFacade(_Checkout(), _Session())
    assert getattr(facade, "reset_to_checkpoint", None) is None
    assert getattr(facade, "evaluate", None) is not None
    assert getattr(facade, "setup", None) is not None
    with pytest.raises(AttributeError):
        facade.anything_at_all  # noqa: B018


def test_the_facade_sets_up_and_scores_an_osworld_task(tmp_path) -> None:
    facade, checkout = _facade(tmp_path)
    steps = facade.setup(
        {
            "id": "chrome/t",
            "config": [{"type": "launch"}],
            "evaluator": {"func": "truthy", "result": {"type": "literal", "value": 1}},
        }
    )
    assert steps == 1
    assert facade.evaluate() == 1.0
    assert checkout.touches >= 4, "both calls feed the lease watchdog on the way in and out"


def test_the_facades_bridge_survives_between_setup_and_evaluate(tmp_path) -> None:
    """A fresh bridge per call would forget the task and make the no-argument
    `evaluate()` unanswerable."""
    facade, _ = _facade(tmp_path)
    facade.setup(
        {"id": "t", "evaluator": {"func": "truthy", "result": {"type": "literal", "value": 1}}}
    )
    assert facade._osworld is facade._osworld
    assert facade.evaluate() == 1.0


def test_the_facade_forwards_the_declared_fail(tmp_path) -> None:
    facade, _ = _facade(tmp_path)
    facade.setup({"id": "t", "evaluator": {"func": "infeasible"}})
    facade.declare_terminal("fail")
    assert facade.evaluate() == 1.0


def test_the_facade_reads_the_guest_ports_off_the_runtime(tmp_path) -> None:
    facade, _ = _facade(tmp_path, runtime=_Runtime())
    assert facade._guest_ports() == {"chromium_port": 9444, "vlc_port": 8222}


def test_the_facade_delegates_the_rest_of_the_surface(tmp_path) -> None:
    """The two real halves `FakeSession` unions: `HttpGuiTransport` for
    input/geometry, `OSWorldClient` for pixels."""
    facade, checkout = _facade(tmp_path)
    assert facade.session_id == "sess-1"
    assert facade.screen_size() == (1280, 800)
    assert facade.cursor_position() == (5, 6)
    assert facade.screenshot() == b"png"
    assert facade.screenshot_settled(min_delay_s=0.5) == b"settled"
    # The guest's own account, forwarded verbatim: `evals/harness.py` publishes it.
    assert facade.execute_atomic([1, 2, 3]) == FakeGuestReceipt(
        ok=True, cursor_before=(5, 6), cursor_after=(5, 6)
    )
    # check=False on purpose: `pgrep chrome` returning 1 is an answer, not an error.
    assert facade.execute_argv(["pgrep", "chrome"])["check"] is False
    facade.execute_pyautogui("pyautogui.moveTo(1, 2)")
    facade.release(failed=True, error="boom")
    assert checkout.released == [(True, "boom")]


def _osworld_tree(tmp_path: Path, *, evaluator: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "osworld"
    examples = root / "evaluation_examples" / "examples" / "chrome"
    examples.mkdir(parents=True)
    payload: dict[str, Any] = {
        "id": "t1",
        "instruction": "open a tab",
        "config": [{"type": "launch"}],
    }
    if evaluator is not None:
        payload["evaluator"] = evaluator
    (examples / "t1.json").write_text(json.dumps(payload))
    (root / "split.json").write_text(json.dumps({"chrome": ["t1"]}))
    return root


@pytest.mark.parametrize(
    "asset_bundle",
    [
        pytest.param("", id="empty"),
        pytest.param("relative-assets", id="relative"),
        pytest.param(None, id="missing"),
    ],
)
def test_an_asset_free_task_refuses_an_invalid_bundle_before_yield(
    tmp_path: Path,
    asset_bundle: str | None,
) -> None:
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root = _osworld_tree(tmp_path)
    bundle = str(tmp_path / "missing-assets") if asset_bundle is None else asset_bundle
    tasks = OSWorldTaskset(
        OSWorldTasksetConfig(
            osworld_root=str(root),
            split_path=str(root / "split.json"),
            asset_bundle=bundle,
        )
    ).load()
    with pytest.raises(ValueError, match="absolute existing asset_bundle directory"):
        next(iter(tasks))


_ASSET_PREFIX = (
    "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/"
)


def _offline_osworld_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "osworld"
    examples = root / "evaluation_examples" / "examples" / "writer"
    examples.mkdir(parents=True)
    task_id = "task-1"
    input_relative = f"writer/{task_id}/input.txt"
    ignored_relative = f"writer/{task_id}/ignored.txt"
    expected_relative = f"writer/{task_id}/expected.bin"
    post_relative = f"writer/{task_id}/post.txt"
    payload = {
        "id": task_id,
        "instruction": "edit the file",
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": _ASSET_PREFIX + input_relative,
                            "path": "Downloads/input.txt",
                        }
                    ]
                },
            },
            {"type": "launch", "parameters": {"command": ["writer"]}},
        ],
        "evaluator": {
            "func": "file_bytes",
            "result": {
                "type": "cloud_file",
                "path": [
                    _ASSET_PREFIX + ignored_relative,
                    _ASSET_PREFIX + expected_relative,
                ],
                "dest": ["ignored.txt", "expected.txt"],
                "multi": True,
                "gives": [1],
            },
            "options": {"content": "expected", "name": "expected.txt"},
            "postconfig": [
                {
                    "type": "download",
                    "parameters": {
                        "files": [
                            {
                                "url": _ASSET_PREFIX + post_relative,
                                "path": "/tmp/post.txt",
                            }
                        ]
                    },
                }
            ],
        },
    }
    (examples / f"{task_id}.json").write_text(json.dumps(payload))
    (root / "split.json").write_text(json.dumps({"writer": [task_id]}))
    bundle = tmp_path / "assets"
    (bundle / input_relative).parent.mkdir(parents=True)
    (bundle / input_relative).write_bytes(b"input")
    (bundle / ignored_relative).write_bytes(b"ignored")
    (bundle / expected_relative).write_bytes(b"expected")
    (bundle / post_relative).write_bytes(b"post")
    return root, bundle


def _offline_file_config(source: Path, dest: str) -> dict[str, Any]:
    return {
        "type": "offline_file",
        "path": [str(source)],
        "dest": [dest],
        "gives": [0],
    }


@pytest.mark.parametrize(
    "dest",
    [
        "",
        ".",
        "..",
        "../file.txt",
        "/tmp/file.txt",
        "nested/file.txt",
        "nested\\file.txt",
    ],
)
def test_cloud_file_destinations_are_safe_at_both_boundaries(
    tmp_path: Path,
    dest: str,
) -> None:
    from evals.osworld_assets import get_offline_file
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root, bundle = _offline_osworld_tree(tmp_path)
    task_path = (
        root / "evaluation_examples" / "examples" / "writer" / "task-1.json"
    )
    payload = json.loads(task_path.read_text())
    payload["evaluator"]["result"]["dest"][1] = dest
    task_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid cloud_file entries"):
        list(
            OSWorldTaskset(
                OSWorldTasksetConfig(
                    osworld_root=str(root),
                    split_path=str(root / "split.json"),
                    asset_bundle=str(bundle),
                )
            ).load()
        )
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="invalid staged OSWorld asset destination"):
        get_offline_file(
            SimpleNamespace(cache_dir=str(tmp_path / "cache")),
            _offline_file_config(source, dest),
        )


def test_cloud_file_multi_requires_a_boolean(tmp_path: Path) -> None:
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root, bundle = _offline_osworld_tree(tmp_path)
    task_path = (
        root / "evaluation_examples" / "examples" / "writer" / "task-1.json"
    )
    payload = json.loads(task_path.read_text())
    payload["evaluator"]["result"]["multi"] = "true"
    task_path.write_text(json.dumps(payload))

    with pytest.raises(TypeError, match="invalid cloud_file multi"):
        list(
            OSWorldTaskset(
                OSWorldTasksetConfig(
                    osworld_root=str(root),
                    split_path=str(root / "split.json"),
                    asset_bundle=str(bundle),
                )
            ).load()
        )


def test_the_offline_getter_leaves_an_existing_final_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.osworld_assets as assets

    source = tmp_path / "source.bin"
    source.write_bytes(b"new")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    final = cache_dir / "renamed.txt"
    final.write_bytes(b"existing")

    def fail_mkstemp(**_kwargs: Any) -> tuple[int, str]:
        raise AssertionError("an existing final must not create a temporary file")

    monkeypatch.setattr(assets.tempfile, "mkstemp", fail_mkstemp)
    result = assets.get_offline_file(
        SimpleNamespace(cache_dir=str(cache_dir)),
        _offline_file_config(source, final.name),
    )

    assert Path(result) == final
    assert final.read_bytes() == b"existing"


def test_the_offline_getter_atomically_replaces_a_cache_local_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.osworld_assets as assets

    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    cache_dir = tmp_path / "cache"
    final = cache_dir / "renamed.txt"
    real_replace = os.replace
    replaced: list[Path] = []

    def record_replace(source_path: str | Path, target_path: str | Path) -> None:
        temporary = Path(source_path)
        assert temporary.parent == cache_dir
        assert temporary != final
        assert temporary.read_bytes() == b"payload"
        assert Path(target_path) == final
        replaced.append(temporary)
        real_replace(source_path, target_path)

    monkeypatch.setattr(assets.os, "replace", record_replace)
    result = assets.get_offline_file(
        SimpleNamespace(cache_dir=str(cache_dir)),
        _offline_file_config(source, final.name),
    )

    assert Path(result) == final
    assert final.read_bytes() == b"payload"
    assert len(replaced) == 1
    assert not replaced[0].exists()


def test_the_offline_getter_cleans_a_failed_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.osworld_assets as assets

    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    cache_dir = tmp_path / "cache"

    def fail_copy(_source: str | Path, temporary: str | Path) -> None:
        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("copy failed")

    monkeypatch.setattr(assets.shutil, "copyfile", fail_copy)
    with pytest.raises(RuntimeError, match="copy failed"):
        assets.get_offline_file(
            SimpleNamespace(cache_dir=str(cache_dir)),
            _offline_file_config(source, "renamed.txt"),
        )

    assert list(cache_dir.iterdir()) == []


def test_the_taskset_stages_one_local_bundle_through_the_real_consumer(
    tmp_path,
) -> None:
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig, preparer_for

    root, bundle = _offline_osworld_tree(tmp_path)
    task = next(
        iter(
            OSWorldTaskset(
                OSWorldTasksetConfig(
                    osworld_root=str(root),
                    split_path=str(root / "split.json"),
                    asset_bundle=str(bundle),
                )
            ).load()
        )
    ).data
    task_config = task.setup["task_config"]
    upload = task_config["config"][0]
    assert upload["type"] == "upload_file"
    assert upload["parameters"]["files"][0]["path"] == "Downloads/input.txt"
    assert Path(upload["parameters"]["files"][0]["local_path"]).read_bytes() == b"input"
    assert task_config["config"][1] == {
        "type": "launch",
        "parameters": {"command": ["writer"]},
    }
    assert task_config["evaluator"]["result"] == {
        "type": "offline_file",
        "path": [
            str(bundle / "writer/task-1/ignored.txt"),
            str(bundle / "writer/task-1/expected.bin"),
        ],
        "dest": ["ignored.txt", "expected.txt"],
        "gives": [1],
    }
    post_upload = task_config["evaluator"]["postconfig"][0]
    assert post_upload["type"] == "upload_file"
    assert Path(post_upload["parameters"]["files"][0]["local_path"]).read_bytes() == b"post"

    facade, _ = _facade(tmp_path)
    evidence = preparer_for("osworld").prepare(facade, task)
    assert evidence == {"prepared": "osworld", "steps": 2, "scorable": True}
    assert facade.evaluate() == 1.0
    assert facade._osworld.setup_controller.calls[0][0] == upload
    assert facade._osworld.setup_controller.calls[1][0] == post_upload


def test_every_selected_bundle_is_validated_before_the_first_task_is_yielded(
    tmp_path,
) -> None:
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root, bundle = _offline_osworld_tree(tmp_path)
    examples = root / "evaluation_examples" / "examples" / "writer"
    missing_id = "task-2"
    missing_relative = f"writer/{missing_id}/missing.txt"
    (examples / f"{missing_id}.json").write_text(
        json.dumps(
            {
                "id": missing_id,
                "instruction": "edit another file",
                "config": [
                    {
                        "type": "download",
                        "parameters": {
                            "files": [
                                {
                                    "url": _ASSET_PREFIX + missing_relative,
                                    "path": "/tmp/missing.txt",
                                }
                            ]
                        },
                    }
                ],
            }
        )
    )
    (root / "split.json").write_text(json.dumps({"writer": ["task-1", missing_id]}))
    tasks = OSWorldTaskset(
        OSWorldTasksetConfig(
            osworld_root=str(root),
            split_path=str(root / "split.json"),
            asset_bundle=str(bundle),
        )
    ).load()
    with pytest.raises(FileNotFoundError, match="task-2.*absent from the bundle"):
        next(iter(tasks))


def test_an_asset_url_outside_the_canonical_bundle_is_rejected(tmp_path) -> None:
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    root, bundle = _offline_osworld_tree(tmp_path)
    path = root / "evaluation_examples" / "examples" / "writer" / "task-1.json"
    payload = json.loads(path.read_text())
    payload["config"][0]["parameters"]["files"][0]["url"] = (
        "https://example.test/input.txt"
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported asset URL"):
        list(
            OSWorldTaskset(
                OSWorldTasksetConfig(
                    osworld_root=str(root),
                    split_path=str(root / "split.json"),
                    asset_bundle=str(bundle),
                )
            ).load()
        )


def test_the_benchmark_arm_pairs_the_flag_with_the_reward() -> None:
    """`OSWorldEvaluateOracle.task_success` raises on a missing reward and only
    `evaluate_on_finish` publishes one, so the two are one decision."""
    import osworld_bench

    assert osworld_bench.OSWORLD_BENCH_ARM.evaluate_on_finish is True
    assert osworld_bench.OSWORLD_BENCH_ARM.id == osworld_bench.PLUGIN_ID
    assert osworld_bench.OSWORLD_BENCH_ARM.require_unsolved_start is False


def test_the_benchmark_plugin_satisfies_the_verifiers_contract() -> None:
    """`loaders._plugin_class` scans `__all__` and needs exactly one Taskset and
    at most one Harness; a second of either breaks resolution at dispatch."""
    import verifiers.v1 as vf

    import osworld_bench

    exported = [getattr(osworld_bench, name) for name in osworld_bench.__all__]
    tasksets = [obj for obj in exported if isinstance(obj, type) and issubclass(obj, vf.Taskset)]
    harnesses = [obj for obj in exported if isinstance(obj, type) and issubclass(obj, vf.Harness)]
    assert len(tasksets) == 1 and len(harnesses) == 1


def test_the_benchmark_taskset_yields_rows_that_carry_the_reward(tmp_path) -> None:
    from evals.tasks import OSWorldTasksetConfig

    from osworld_bench import OSWorldBenchTask, OSWorldBenchTaskset

    root = _osworld_tree(tmp_path, evaluator={"func": "truthy"})
    tasks = list(
        OSWorldBenchTaskset(
            OSWorldTasksetConfig(
                osworld_root=str(root),
                split_path=str(root / "split.json"),
                asset_bundle=str(tmp_path),
            )
        ).load()
    )
    assert [type(task) for task in tasks] == [OSWorldBenchTask]
    assert callable(getattr(OSWorldBenchTask, "task_success", None)), (
        "the row must carry the OSWorld reward; a bare DesktopTask carries none, "
        "which is why the family produced episodes and no score"
    )


def test_the_taskset_carries_the_whole_task_json_not_just_its_config(tmp_path) -> None:
    """The row and the score must not disagree because the checkout moved under a
    running 369-task array."""
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig

    evaluator = {"func": "truthy", "result": {"type": "literal", "value": 1}}
    root = _osworld_tree(tmp_path, evaluator=evaluator)
    task = next(
        iter(
            OSWorldTaskset(
                OSWorldTasksetConfig(
                    osworld_root=str(root),
                    split_path=str(root / "split.json"),
                    asset_bundle=str(tmp_path),
                )
            ).load()
        )
    )
    assert task.data.setup["task_config"]["evaluator"] == evaluator


def test_the_preparer_and_the_facade_meet(tmp_path) -> None:
    """A taskset row -> the OSWorld preparer -> the production facade -> a
    benchmark score. Nothing here is a `FakeSession`."""
    from evals.tasks import OSWorldTaskset, OSWorldTasksetConfig, preparer_for

    root = _osworld_tree(
        tmp_path, evaluator={"func": "truthy", "result": {"type": "literal", "value": 1}}
    )
    task = next(
        iter(
            OSWorldTaskset(
                OSWorldTasksetConfig(
                    osworld_root=str(root),
                    split_path=str(root / "split.json"),
                    asset_bundle=str(tmp_path),
                )
            ).load()
        )
    ).data
    facade, _ = _facade(tmp_path)
    evidence = preparer_for("osworld").prepare(facade, task)
    assert evidence == {"prepared": "osworld", "steps": 1, "scorable": True}
    assert preparer_for("osworld").probe(facade, task) == {"cursor": [5, 6]}
    assert facade.evaluate() == 1.0
