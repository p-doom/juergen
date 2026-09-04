"""Stage 05/06 enforce one tokenizer and one measured-cache contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import grammars
import pytest
import synthetic_clip as clip

from pipeline.lib.manifest import make_artifact_id
from pipeline.stage_04_build_conversations import build_messages

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PIPELINE_DIR = REPO_ROOT / "data_pipeline"
STAGES = REPO_ROOT / "pipeline"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _chat_row(index: int) -> dict[str, Any]:
    shard = f"/nonexistent/{clip.SEGMENT_ID}/images.array_record"
    turns = [(f"ar://{shard}#{turn * clip.STRIDE}", "NO_OP") for turn in range(3)]
    return {
        "conversation_id": f"{clip.SEGMENT_ID}:{index}",
        "recording_id": clip.RECORDING_ID,
        "segment_id": clip.SEGMENT_ID,
        "n_frames": 3,
        "n_turns": 3,
        "messages": build_messages(
            turns,
            instruction="do the synthetic thing",
            system_prompt=grammars.describe("deltatype_v2"),
        ),
    }


def make_source(root: Path, *, n_conversations: int = 2) -> Path:
    root.mkdir(parents=True)
    rows = [_chat_row(index) for index in range(n_conversations)]
    chat = root / "chat.jsonl"
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(chat.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "juergen_annotation_conversations",
                "schema_version": 2,
                "chat": "chat.jsonl",
                "chat_sha256": digest,
                "n_conversations": len(rows),
            }
        )
        + "\n"
    )
    return root


_FAKE_UV = """#!{python}
import json, os, sys
from pathlib import Path

Path(os.environ["FAKE_UV_LOG"]).open("a").write(json.dumps(sys.argv[1:]) + "\\n")
flags = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
script = next((a for a in sys.argv[1:] if a.endswith(".py")), "")
out = Path(flags["out_dir"])
out.mkdir(parents=True, exist_ok=True)
if "measure_message_lengths" in script:
    mode = os.environ.get("FAKE_UV_CACHE", "valid")
    if mode != "absent":
        with (out / "message_lengths.jsonl").open("w") as target:
            if mode == "valid":
                for index in range(6):
                    target.write(json.dumps({{"conv_idx": 0, "msg_offset": index}}) + "\\n")
elif os.environ.get("FAKE_UV_SHARDS", "1") != "0":
    (out / "part-00000.array_record").write_bytes(b"")
"""


@pytest.fixture
def omegalax(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "omegalax"
    (repo / "scripts").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(_FAKE_UV.format(python=sys.executable))
    uv.chmod(0o755)
    log = tmp_path / "uv_argv.jsonl"
    roots = [REPO_ROOT, DATA_PIPELINE_DIR, REPO_ROOT.parent / "desktop"]
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        PYTHONPATH=os.pathsep.join(str(root) for root in roots),
        FAKE_UV_LOG=str(log),
    )
    return {"repo": repo, "env": env, "log": log}


def _run(stage: str, env: dict[str, str], **flags: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(STAGES / stage),
            *[f"--{name}={value}" for name, value in flags.items()],
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _invocations(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _flags_of(argv: list[str]) -> dict[str, str]:
    return dict(
        argument[2:].split("=", 1)
        for argument in argv
        if argument.startswith("--") and "=" in argument
    )


def _measure(
    tmp_path: Path,
    omegalax: dict[str, Any],
    source: Path,
    *,
    processor: str = MODEL_ID,
) -> Path:
    output = tmp_path / f"lengths-{processor.rsplit('/', 1)[-1]}"
    result = _run(
        "stage_05_measure_lengths.py",
        omegalax["env"],
        output_dir=output,
        source_path=source,
        omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID,
        processor=processor,
        num_workers=2,
    )
    assert result.returncode == 0, result.stderr
    return output


def _record_flags(
    tmp_path: Path,
    omegalax: dict[str, Any],
    source: Path,
    lengths: Path,
    **overrides: object,
) -> dict[str, object]:
    values: dict[str, object] = {
        "output_dir": tmp_path / "records",
        "source_path": source,
        "omegalax_repo": omegalax["repo"],
        "model_id": MODEL_ID,
        "processor": MODEL_ID,
        "max_length": 4096,
        "records_per_shard": 8,
        "num_workers": 2,
        "message_lengths_path": lengths,
    }
    values.update(overrides)
    return values


def test_stage_05_refuses_missing_chat_before_launch(tmp_path: Path, omegalax) -> None:
    source = make_source(tmp_path / "source")
    (source / "chat.jsonl").unlink()
    result = _run(
        "stage_05_measure_lengths.py",
        omegalax["env"],
        output_dir=tmp_path / "lengths",
        source_path=source,
        omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID,
        processor=MODEL_ID,
        num_workers=2,
    )
    assert result.returncode != 0
    assert "chat artifact is missing" in result.stderr
    assert _invocations(omegalax["log"]) == []


def test_stage_05_records_cache_digest_and_processor(tmp_path: Path, omegalax) -> None:
    source = make_source(tmp_path / "source")
    output = _measure(tmp_path, omegalax, source)
    (argv,) = _invocations(omegalax["log"])
    assert _flags_of(argv) == {
        "data_path": str(source / "chat.jsonl"),
        "out_dir": str(output),
        "model_id": MODEL_ID,
        "processor": MODEL_ID,
        "num_workers": "2",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    cache = output / "message_lengths.jsonl"
    assert manifest["inputs"]["source_id"] == make_artifact_id(source)
    assert manifest["params"]["processor"] == MODEL_ID
    assert manifest["stats"]["cache"] == {
        "file": "message_lengths.jsonl",
        "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "n_messages": 6,
        "elapsed_s": 0,
    }


@pytest.mark.parametrize("mode, message", [("absent", "no cache"), ("empty", "empty cache")])
def test_stage_05_rejects_success_without_cache(
    tmp_path: Path, omegalax, mode: str, message: str
) -> None:
    source = make_source(tmp_path / "source")
    env = {**omegalax["env"], "FAKE_UV_CACHE": mode}
    result = _run(
        "stage_05_measure_lengths.py",
        env,
        output_dir=tmp_path / "lengths",
        source_path=source,
        omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID,
        processor=MODEL_ID,
        num_workers=2,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_stage_06_requires_matching_source_model_processor_and_digest(
    tmp_path: Path, omegalax
) -> None:
    source = make_source(tmp_path / "source")
    other = make_source(tmp_path / "other", n_conversations=3)
    other_lengths = _measure(tmp_path, omegalax, other)
    result = _run(
        "stage_06_training_records.py",
        omegalax["env"],
        **_record_flags(tmp_path, omegalax, source, other_lengths),
    )
    assert result.returncode != 0
    assert "cache source mismatch" in result.stderr

    lengths = _measure(tmp_path, omegalax, source)
    for field, value in (("model_id", "Other/model"), ("processor", "Other/processor")):
        result = _run(
            "stage_06_training_records.py",
            omegalax["env"],
            **_record_flags(tmp_path, omegalax, source, lengths, **{field: value}),
        )
        assert result.returncode != 0
        assert f"cache {field} mismatch" in result.stderr

    with (lengths / "message_lengths.jsonl").open("a") as target:
        target.write("{}\n")
    result = _run(
        "stage_06_training_records.py",
        omegalax["env"],
        **_record_flags(tmp_path, omegalax, source, lengths),
    )
    assert result.returncode != 0
    assert "cache digest mismatch" in result.stderr


def test_stage_06_reuses_cache_and_builds_each_split(tmp_path: Path, omegalax) -> None:
    source = make_source(tmp_path / "source")
    lengths = _measure(tmp_path, omegalax, source)
    output = tmp_path / "records"
    result = _run(
        "stage_06_training_records.py",
        omegalax["env"],
        **_record_flags(
            tmp_path,
            omegalax,
            source,
            lengths,
            output_dir=output,
            val_fraction=0.25,
        ),
    )
    assert result.returncode == 0, result.stderr
    builds = [
        argv
        for argv in _invocations(omegalax["log"])
        if any("build_sft_records" in argument for argument in argv)
    ]
    assert [_flags_of(argv)["split"] for argv in builds] == ["train", "val"]
    for argv in builds:
        flags = _flags_of(argv)
        assert flags["message_lengths_path"] == str(lengths / "message_lengths.jsonl")
        assert flags["overflow_mode"] == "split"
        assert "--overwrite" in argv
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["inputs"]["source_id"] == make_artifact_id(source)
    assert manifest["inputs"]["message_lengths_id"] == make_artifact_id(lengths)
    assert [row["split"] for row in manifest["stats"]["per_split"]] == [
        "train",
        "val",
    ]


def test_stage_06_rejects_zero_output_shards(tmp_path: Path, omegalax) -> None:
    source = make_source(tmp_path / "source")
    lengths = _measure(tmp_path, omegalax, source)
    env = {**omegalax["env"], "FAKE_UV_SHARDS": "0"}
    result = _run(
        "stage_06_training_records.py",
        env,
        **_record_flags(tmp_path, omegalax, source, lengths),
    )
    assert result.returncode != 0
    assert "produced no shards" in result.stderr
