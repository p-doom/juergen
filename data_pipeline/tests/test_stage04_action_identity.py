"""Action-mode byte-identity gate for the stage-04 merge.

The single ``stage_04_conversations.py --mode action`` must be byte-identical
to the OLD ``stage_04_build_conversations.py`` (recovered from git HEAD, since
the merge git-rm's it) on the conversation output — conversations.jsonl and
chat.jsonl, the actual training records. This proves the 1b merge did not
change action-mode semantics. Runs both builders as subprocesses on a synthetic
stage-03 filter artifact (and, when present, one real legacy day slice).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import msgpack

from realigned_pipeline.lib.manifest import make_artifact_id

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_PIPELINE_DIR.parent
NEW_SCRIPT = DATA_PIPELINE_DIR / "realigned_pipeline" / "stage_04_conversations.py"
OLD_REL = "data_pipeline/realigned_pipeline/stage_04_build_conversations.py"
REAL_FILTER = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/"
    "yll.kryeziu/realigned_ccast0618d_v3_filter")


def _git_show(rel: str) -> str | None:
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT,
                             capture_output=True, check=True)
        return out.stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _run(script: Path, args: list[str]) -> None:
    res = subprocess.run([sys.executable, str(script), *args], cwd=DATA_PIPELINE_DIR,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"{script.name} failed ({res.returncode}):\nSTDOUT\n{res.stdout}\n"
            f"STDERR\n{res.stderr}")


def _assert_identical(old_dir: Path, new_dir: Path) -> None:
    for name in ("conversations.jsonl", "chat.jsonl"):
        old_b = (old_dir / name).read_bytes()
        new_b = (new_dir / name).read_bytes()
        assert old_b, f"old {name} is empty"
        if old_b != new_b:  # surface the first differing record for debugging
            o = old_b.decode().splitlines()
            n = new_b.decode().splitlines()
            for i, (a, b) in enumerate(zip(o, n)):
                if a != b:
                    raise AssertionError(f"{name} differs at record {i}:\nOLD {a}\nNEW {b}")
            raise AssertionError(f"{name} differs in length: {len(o)} vs {len(n)}")


def _keylog(path: Path) -> None:
    entries = [
        [100_000, ["MouseMove", [3.0, -2.0]]],
        [1_100_000, ["KeyPress", "KeyH"]],
        [1_200_000, ["KeyRelease", "KeyH"]],
        [2_050_000, ["MouseScroll", [0.0, 5.0]]],
        [3_500_000, ["MousePress", "Left"]],
        [3_600_000, ["MouseRelease", "Left"]],
    ]
    path.write_bytes(msgpack.packb(entries, use_bin_type=True))


def _synthetic_filter(root: Path) -> Path:
    master = root / "master"
    master.mkdir()
    (master / "manifest.json").write_text(json.dumps({"artifact_type": "m", "v": 1}))
    master_id = make_artifact_id(master)

    fdir = root / "filter"
    (fdir / "filter").mkdir(parents=True)
    (fdir / "manifest.json").write_text(json.dumps({
        "artifact_type": "realigned_filter_mask",
        "master_fps": 15.0,
        "master_store_id": master_id,
    }))

    keylog = root / "s0.keylog.msgpack"
    _keylog(keylog)
    segs = [
        {"segment_id": "s0", "recording_id": "rA", "segment_idx": 0,
         "master_fps": 15.0, "n_master_records": 150,
         "shard_path": "/frames/s0/images.array_record",
         "keylog_path": str(keylog), "alignment_status": "aligned",
         "kept_ranges": [[0, 150]], "dropped": []},
        {"segment_id": "s1", "recording_id": "rB", "segment_idx": 0,
         "master_fps": 15.0, "n_master_records": 120,
         "shard_path": "/frames/s1/images.array_record",
         "keylog_path": None, "alignment_status": "aligned",
         "kept_ranges": [[0, 60], [75, 120]],
         "dropped": [{"start": 60, "end": 75, "reason": "black"}]},
    ]
    with (fdir / "filter_index.jsonl").open("w") as f:
        for s in segs:
            (fdir / "filter" / f"{s['segment_id']}.json").write_text(json.dumps(s))
            f.write(json.dumps({"segment_id": s["segment_id"], "status": "ok",
                                "recording_id": s["recording_id"],
                                "segment_idx": s["segment_idx"],
                                "n_kept": s["n_master_records"]}) + "\n")
    return fdir


class ActionByteIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        old_src = _git_show(OLD_REL)
        if old_src is None:
            self.skipTest("old stage_04_build_conversations.py not retrievable from git HEAD")
        # Materialize the old builder inside the package so its parents[1]
        # sys.path hack + realigned_pipeline imports resolve exactly as before.
        self.old_script = DATA_PIPELINE_DIR / "realigned_pipeline" / "_golden_stage_04_build_conversations.py"
        self.old_script.write_text(old_src)

    def tearDown(self) -> None:
        script = getattr(self, "old_script", None)
        if script is not None and script.exists():
            script.unlink()

    def _compare(self, extra_old: list[str], extra_new: list[str]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fdir = _synthetic_filter(root)
            old_out, new_out = root / "old", root / "new"
            common = ["--filter-dir", str(fdir), "--fps", "1", "--num-workers", "1"]
            _run(self.old_script, [*common, "--output-dir", str(old_out), *extra_old])
            _run(NEW_SCRIPT, ["--mode", "action", *common,
                              "--clips-manifest", str(root / "nonexistent_manifest.jsonl"),
                              "--day-index-cache", str(root / "cache.json"),
                              "--output-dir", str(new_out), *extra_new])
            _assert_identical(old_out, new_out)

    def test_goal_free_canonical_is_byte_identical(self) -> None:
        # exercises frame selection, canonical action formatting from a keylog,
        # black-zone dead-zone clamping (s1), instruction on the first turn,
        # terminal token on the last, system prompt, sort + serialization.
        shared = ["--instruction", "Do the task.", "--terminal-token", "<terminate>"]
        self._compare(shared, shared)

    def test_ordered_v3_default_prompt_is_byte_identical(self) -> None:
        shared = ["--action-format", "ordered_events_v3"]
        self._compare(shared, shared)


@unittest.skipUnless(REAL_FILTER.is_dir(), "real legacy filter artifact not present")
class ActionByteIdentityRealDayTest(unittest.TestCase):
    """Same gate on a real legacy slice (first --limit segments) — proves the
    merge is byte-identical on production data, not just synthetic fixtures."""

    def setUp(self) -> None:
        old_src = _git_show(OLD_REL)
        if old_src is None:
            self.skipTest("old builder not retrievable from git HEAD")
        self.old_script = DATA_PIPELINE_DIR / "realigned_pipeline" / "_golden_stage_04_build_conversations.py"
        self.old_script.write_text(old_src)

    def tearDown(self) -> None:
        script = getattr(self, "old_script", None)
        if script is not None and script.exists():
            script.unlink()

    def test_real_slice_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_out, new_out = root / "old", root / "new"
            common = ["--filter-dir", str(REAL_FILTER), "--fps", "0.5",
                      "--limit", "20", "--num-workers", "4"]
            _run(self.old_script, [*common, "--output-dir", str(old_out)])
            _run(NEW_SCRIPT, ["--mode", "action", *common,
                              "--clips-manifest", str(root / "nope.jsonl"),
                              "--day-index-cache", str(root / "cache.json"),
                              "--output-dir", str(new_out)])
            _assert_identical(old_out, new_out)


if __name__ == "__main__":
    unittest.main()
