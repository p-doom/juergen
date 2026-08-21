"""The cuagym family without a VM: suite pins, setup steps, reward parsing."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from evals.cuagym.bundles import load_suite, verify_bundle
from evals.cuagym.guest import CuaGymPreparer, run_reward, run_steps, write_guest_file
from evals.cuagym.oracle import parse_reward_stdout
from evals.cuagym.taskset import CuaGymTaskset, CuaGymTasksetConfig
from evals.tasks import preparer_for


class FakeSession:
    """Records guest commands and replays canned stdout, like the real facade."""

    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.files: dict[str, bytes] = {}
        self.outputs = outputs or {}

    def execute_argv(self, argv: list[str]) -> dict[str, str]:
        self.commands.append(list(argv))
        joined = " ".join(argv)
        if len(argv) == 3 and argv[:2] == ["bash", "-lc"] and "| base64 -d >>" in argv[2]:
            encoded = argv[2].split("printf '%s' ", 1)[1].split(" | base64", 1)[0]
            target = argv[2].rsplit(">> ", 1)[1]
            self.files[target] = self.files.get(target, b"") + base64.b64decode(
                encoded.strip("'")
            )
        for needle, output in self.outputs.items():
            if needle in joined:
                return {"output": output}
        return {"output": ""}

    def cursor_position(self) -> tuple[int, int]:
        return (12, 34)

    def screen_size(self) -> tuple[int, int]:
        return (1920, 1080)


# --- reward parsing -------------------------------------------------------


def test_reward_line_parses() -> None:
    assert parse_reward_stdout("Component 1 ok\nREWARD: 0.65\n") == 0.65


def test_repeated_identical_reward_lines_parse() -> None:
    assert parse_reward_stdout("REWARD: 1.0\nREWARD: 1.0\n") == 1.0


@pytest.mark.parametrize(
    "stdout",
    [
        "no reward here",
        "REWARD: 0.5\nREWARD: 0.7",  # conflicting
        "REWARD: 1.5",  # out of range
        "REWARD: nan",  # non-finite
        "REWARD:",  # malformed prefix
        "0.5",  # bare numbers are not this suite's format
    ],
)
def test_deviant_reward_stdout_raises(stdout: str) -> None:
    with pytest.raises(ValueError):
        parse_reward_stdout(stdout)


# --- setup steps ----------------------------------------------------------


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "initial_setup.py").write_bytes(b"print('setup')\n" * 100)
    (bundle / "reward.py").write_text("print('REWARD: 1.0')\n")
    return bundle


def test_run_steps_covers_all_four_kinds(tmp_path: Path, monkeypatch) -> None:
    session = FakeSession()
    slept: list[float] = []
    monkeypatch.setattr("evals.cuagym.guest.time.sleep", slept.append)
    bundle = _bundle(tmp_path)
    ran = run_steps(
        session,
        [
            {
                "type": "download",
                "parameters": {
                    "files": [{"url": "./initial_setup.py", "path": "/home/user/s.py"}]
                },
            },
            {"type": "execute", "parameters": {"command": "python3 /home/user/s.py"}},
            {"type": "execute", "parameters": {"command": ["bash", "-c", "true"]}},
            {"type": "sleep", "parameters": {"seconds": 2}},
            {"type": "open", "parameters": {"path": "/home/user/a b.docx"}},
        ],
        bundle,
    )
    assert ran == 5
    assert session.files["/home/user/s.py"] == (bundle / "initial_setup.py").read_bytes()
    assert ["bash", "-lc", "python3 /home/user/s.py"] in session.commands
    assert ["bash", "-c", "true"] in session.commands
    assert slept == [2.0]
    assert any("xdg-open '/home/user/a b.docx'" in c[-1] for c in session.commands)


def test_run_steps_refuses_http_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative bundle member"):
        run_steps(
            FakeSession(),
            [
                {
                    "type": "download",
                    "parameters": {
                        "files": [{"url": "http://x/y", "path": "/home/user/y"}]
                    },
                }
            ],
            _bundle(tmp_path),
        )


def test_write_guest_file_chunks_large_payload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evals.cuagym.guest._WRITE_CHUNK_BYTES", 8)
    session = FakeSession()
    write_guest_file(session, "/tmp/t.bin", b"0123456789abcdef!")
    assert session.files["/tmp/t.bin"] == b"0123456789abcdef!"


def test_run_reward_runs_postconfig_then_verifier(tmp_path: Path, monkeypatch) -> None:
    session = FakeSession(outputs={"/tmp/cuagym_reward.py": "REWARD: 0.3\n"})
    monkeypatch.setattr("evals.cuagym.guest.time.sleep", lambda _s: None)
    bundle = _bundle(tmp_path)
    stdout = run_reward(
        session,
        (bundle / "reward.py").read_text(),
        [{"type": "execute", "parameters": {"command": ["python", "-c", "pass"]}}],
        bundle,
    )
    assert parse_reward_stdout(stdout) == 0.3
    reward_run = session.commands.index(["python3", "/tmp/cuagym_reward.py"])
    postconfig_run = session.commands.index(["python", "-c", "pass"])
    assert postconfig_run < reward_run


def test_preparer_is_registered_and_probe_is_read_only() -> None:
    preparer = preparer_for("cuagym")
    assert isinstance(preparer, CuaGymPreparer)
    session = FakeSession()
    probe = preparer.probe(session, None)  # type: ignore[arg-type]
    assert probe == {"cursor": [12, 34], "screen": [1920, 1080]}
    assert session.commands == []  # read-only, no guest commands


# --- suite + taskset ------------------------------------------------------


def test_shipped_suite_loads_and_is_pinned() -> None:
    suite = load_suite()
    assert suite["suite"] == "cuagym-mini-v1"
    assert len(suite["tasks"]) == 28
    for task in suite["tasks"]:
        pins = set(task["sha256"])
        # every bundle pins its manifest, its verifier, and one setup script
        # (a few ship initial_setup.sh rather than .py)
        assert {"task.json", "reward.py"} <= pins
        assert any(name.startswith("initial_setup.") for name in pins)


def _fake_suite_tree(tmp_path: Path) -> tuple[Path, Path]:
    bundle_root = tmp_path / "bundles"
    task_dir = bundle_root / "t-0001"
    task_dir.mkdir(parents=True)
    payload = {
        "id": "t-0001",
        "instruction": "Do the thing",
        "app_type": "vscode",
        "config": [{"type": "execute", "parameters": {"command": "true"}}],
        "evaluator": {"type": "python", "url": "./reward.py"},
    }
    (task_dir / "task.json").write_text(json.dumps(payload))
    (task_dir / "initial_setup.py").write_text("pass\n")
    (task_dir / "reward.py").write_text("print('REWARD: 0.0')\n")
    suite = {
        "suite": "fake",
        "dataset_revision": "deadbeef",
        "archive": "artifacts/x.tar.zst",
        "defaults": {"max_steps": 7},
        "tasks": [
            {
                "id": "t-0001",
                "app": "vscode",
                "instruction": "Do the thing",
                "sha256": {
                    name: hashlib.sha256((task_dir / name).read_bytes()).hexdigest()
                    for name in ("task.json", "initial_setup.py", "reward.py")
                },
            }
        ],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    return suite_path, bundle_root


def test_taskset_yields_verified_rows(tmp_path: Path) -> None:
    suite_path, bundle_root = _fake_suite_tree(tmp_path)
    taskset = CuaGymTaskset(
        CuaGymTasksetConfig(bundles_root=str(bundle_root), suite_path=str(suite_path))
    )
    rows = list(taskset.load())
    assert len(rows) == 1
    data = rows[0].data
    assert data.kind == "cuagym"
    assert data.max_steps == 7
    assert data.setup["bundle_dir"] == str(bundle_root / "t-0001")
    assert Path(data.setup["reward_path"]).name == "reward.py"


def test_taskset_refuses_a_drifted_bundle(tmp_path: Path) -> None:
    suite_path, bundle_root = _fake_suite_tree(tmp_path)
    (bundle_root / "t-0001" / "reward.py").write_text("print('REWARD: 1.0')\n")
    taskset = CuaGymTaskset(
        CuaGymTasksetConfig(bundles_root=str(bundle_root), suite_path=str(suite_path))
    )
    with pytest.raises(ValueError, match="does not match the suite's pin"):
        list(taskset.load())


def test_verify_bundle_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reward.py"):
        verify_bundle(tmp_path, {"reward.py": "0" * 64}, "t-0002")
