"""Stage 05/06 seal the compiler, processor, cache, and record artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pipeline.cua_gym.stage_01_image_store import build_store
from pipeline.cua_gym.stage_04_build_conversations import build_dataset
from pipeline.lib.manifest import make_artifact_id
from pipeline.lib.omegalax import attest_processor_snapshot, require_conversations_fit

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = REPO_ROOT / "pipeline"
COMMIT = "a" * 40
TREE = "b" * 40
SNAPSHOT_REVISION = "c" * 40
SNAPSHOT_FILES = {
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def make_source(root: Path, image_store: Path, *, n_conversations: int = 2) -> Path:
    root.mkdir(parents=True)
    curated = root / "curated"
    curated.mkdir()
    curated_rows = curated / "trajectories.jsonl"
    curated_rows.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": f"source-{index}",
                    "instruction": "instruction",
                    "app": "writer",
                    "screen": [1920, 1080],
                    "steps": [
                        {
                            "step": 0,
                            "shard": "screenshots-0000.tar",
                            "member": "task/step_000.png",
                            "reasoning": "reason",
                            "action": {
                                "primitives": [],
                                "no_op": True,
                                "terminate": None,
                            },
                        }
                    ],
                }
            )
            + "\n"
            for index in range(n_conversations)
        )
    )
    empty_digest = hashlib.sha256(b"[]").hexdigest()
    (curated / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "cuagym_stage_03_curated_trajectories",
                "schema_version": 1,
                "trajectories": "trajectories.jsonl",
                "trajectories_sha256": hashlib.sha256(curated_rows.read_bytes()).hexdigest(),
                "inputs": {"source": "/source", "source_sha256": "0" * 64},
                "exclusions": [],
                "exclusions_sha256": empty_digest,
                "dispositions": [],
                "dispositions_sha256": empty_digest,
                "stats": {
                    "excluded_rollouts": 0,
                    "executable_targets": n_conversations,
                    "executed_calls": n_conversations,
                    "logical_targets": n_conversations,
                    "multicall_extra_calls": 0,
                    "multicall_turns": 0,
                    "nonexecutable_calls": 0,
                    "nonexecuted_events": 0,
                    "reasoning_closed": n_conversations,
                    "reasoning_double_open_tool_tag": 0,
                    "reasoning_missing_closer": 0,
                    "reasoning_prose_after_closer": 0,
                    "reasoning_thinking_closer_typo": 0,
                    "retained_rollouts": n_conversations,
                    "source_events": n_conversations,
                    "source_rollouts": n_conversations,
                },
            }
        )
        + "\n"
    )
    build_dataset(curated, image_store, root)
    return root


_FAKE_GIT = """#!{python}
import os, sys
args = sys.argv[1:]
if args == ["rev-parse", "HEAD"]:
    print(os.environ.get("FAKE_GIT_HEAD", "{commit}"))
elif args == ["rev-parse", "HEAD^{{tree}}"]:
    print(os.environ.get("FAKE_GIT_TREE", "{tree}"))
elif args[:2] == ["status", "--porcelain=v1"]:
    print(os.environ.get("FAKE_GIT_STATUS", ""))
elif args[:2] == ["ls-files", "-z"]:
    names = [
        "scripts/measure_message_lengths_from_chat.py",
        "scripts/build_sft_records_from_chat.py",
        "omegalax/data/qwen3_encoding.py",
        "pyproject.toml",
        "uv.lock",
    ]
    sys.stdout.buffer.write(("\\0".join(names) + "\\0").encode())
else:
    raise SystemExit(f"unexpected git invocation: {{args}}")
"""


_FAKE_UV = """#!{python}
import json, os, sys
from pathlib import Path

Path(os.environ["FAKE_UV_LOG"]).open("a").write(json.dumps(sys.argv[1:]) + "\\n")
for name in ("PYTHONHOME", "PYTHONPATH", "UV_NO_SYNC", "UV_PROJECT_ENVIRONMENT"):
    if name in os.environ:
        raise SystemExit(f"unscrubbed environment: {{name}}")
if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise SystemExit("PYTHONNOUSERSITE is not set")
flags = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
script = next(a for a in sys.argv[1:] if a.endswith(".py"))
out = Path(flags["out_dir"])
out.mkdir(parents=True, exist_ok=True)
if "measure_message_lengths" in script:
    mode = os.environ.get("FAKE_UV_CACHE", "valid")
    if mode != "absent":
        with Path(flags["data_path"]).open() as source, (out / "message_lengths.jsonl").open("w") as target:
            if mode != "empty":
                for conv_idx, line in enumerate(row for row in source if row.strip()):
                    messages = json.loads(line)["messages"]
                    for msg_offset, message in enumerate(messages):
                        num_images = sum(
                            part.get("type") == "image"
                            for part in message["content"]
                            if isinstance(part, dict)
                        )
                        measurement = {{
                            "length": 10,
                            "vision_tokens": 4 * num_images,
                            "vision_patches": 16 * num_images,
                            "num_images": num_images,
                            "image_grid_thw": [[1, 4, 4]] * num_images,
                        }}
                        if mode == "malformed":
                            measurement.pop("length")
                        if mode == "bad_vision_tokens" and num_images:
                            measurement["vision_tokens"] = 999
                        target.write(json.dumps({{
                            "conv_idx": conv_idx,
                            "msg_offset": msg_offset,
                            "measurement": measurement,
                        }}) + "\\n")
        if os.environ.get("FAKE_MUTATE_SNAPSHOT"):
            (Path(flags["processor"]) / "tokenizer.json").write_text("mutated")
elif os.environ.get("FAKE_UV_RECORDS", "valid") != "absent":
    import hashlib
    from array_record.python.array_record_module import ArrayRecordWriter
    mode = os.environ.get("FAKE_UV_RECORDS", "valid")
    shard = out / "part-00000.array_record"
    writer = ArrayRecordWriter(str(shard), "group_size:1")
    rows = []
    if mode != "empty":
        with Path(flags["data_path"]).open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                bucket = int(
                    hashlib.sha1(str(row["recording_id"]).encode()).hexdigest(), 16
                ) % 1000
                expected = (
                    "val"
                    if bucket < round(float(flags["val_fraction"]) * 1000)
                    else "train"
                )
                if expected == flags["split"]:
                    rows.append((line_number, row))
        if mode == "omit":
            rows = rows[:1]
        if mode == "reverse":
            rows.reverse()
        for line_number, row in rows:
            record = {{
                key: value
                for key, value in row.items()
                if key not in {{"messages", "session_id"}}
            }}
            record["messages"] = row["messages"]
            record["_omegalax_session_id"] = (
                f"{{Path(flags['data_path']).stem}}-{{line_number:09d}}"
            )
            record["_omegalax_measured_length"] = (
                int(flags["max_length"]) + 1
                if mode == "length_overflow"
                else (1 if mode == "wrong_length" else 10 * len(record["messages"]))
            )
            if mode == "unsupervised":
                for message in record["messages"]:
                    if message.get("role") == "assistant":
                        message["loss"] = False
            payload = json.dumps(record, sort_keys=True).encode()
            if mode == "noncanonical":
                payload = json.dumps(record, sort_keys=False).encode()
            writer.write(payload)
    writer.close()
    if mode == "corrupt":
        shard.write_bytes(b"corrupt")
    if mode == "crc":
        payload = bytearray(shard.read_bytes())
        payload[112] ^= 1
        shard.write_bytes(payload)
    metadata = {{
        "inline_records": True,
        "source_chat_path": str(Path(flags["data_path"]).resolve()),
        "max_length": int(flags["max_length"]),
        "overflow_mode": "split",
        "split": flags["split"],
        "val_fraction": float(flags["val_fraction"]),
        "profile_metadata": {{
            "model_id": flags["model_id"],
            "tokenizer": flags["model_id"],
            "processor": flags["processor"],
            "preprocessor_config": None,
        }},
        "version": 1,
        "num_records": (
            len(rows) + 1
            if mode == "count_mismatch"
            else (0 if mode == "empty" else len(rows))
        ),
        "num_shards": 1,
        "shard_paths": ["part-00000.array_record"],
    }}
    if mode == "bad_metadata":
        metadata["overflow_mode"] = "drop"
    (out / "metadata.json").write_text(json.dumps(metadata))
    for name in (
        "sequence_lengths.jsonl",
        "token_stats.json",
        "truncation_stats.json",
    ):
        if not (mode == "missing_sidecar" and name == "token_stats.json"):
            (out / name).write_text("{{}}\\n")
    if mode == "extra_file":
        (out / "extra.json").write_text("{{}}\\n")
    if os.environ.get("FAKE_MUTATE_TRAIN") and flags["split"] == "val":
        train = out.parent / "train" / "part-00000.array_record"
        payload = bytearray(train.read_bytes())
        payload[112] ^= 1
        train.write_bytes(payload)
    if os.environ.get("FAKE_MUTATE_REPO"):
        (Path.cwd() / "omegalax/data/qwen3_encoding.py").write_text("mutated")
"""


@pytest.fixture
def production_inputs(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "omegalax"
    for relative in (
        "scripts/measure_message_lengths_from_chat.py",
        "scripts/build_sft_records_from_chat.py",
        "omegalax/data/qwen3_encoding.py",
        "pyproject.toml",
        "uv.lock",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    snapshot = tmp_path / "models--test" / "snapshots" / SNAPSHOT_REVISION
    snapshot.mkdir(parents=True)
    for name in SNAPSHOT_FILES:
        value = {"merge_size": 2} if name == "preprocessor_config.json" else {}
        (snapshot / name).write_text(json.dumps(value))
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    image = io.BytesIO()
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(image, format="PNG")
    with tarfile.open(screenshots / "screenshots-0000.tar", "w") as archive:
        info = tarfile.TarInfo("task/step_000.png")
        info.size = len(image.getvalue())
        archive.addfile(info, io.BytesIO(image.getvalue()))
    image_store = tmp_path / "image-store"
    build_store(screenshots, image_store, workers=1)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(_FAKE_GIT.format(python=sys.executable, commit=COMMIT, tree=TREE))
    git.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text(_FAKE_UV.format(python=sys.executable))
    uv.chmod(0o755)
    log = tmp_path / "uv_argv.jsonl"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        PYTHONPATH=str(REPO_ROOT),
        FAKE_UV_LOG=str(log),
    )
    return {
        "repo": repo,
        "snapshot": snapshot,
        "image_store": image_store,
        "env": env,
        "log": log,
    }


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


def _script_invocations(log: Path) -> list[list[str]]:
    return [argv for argv in _invocations(log) if any(arg.endswith(".py") for arg in argv)]


def _flags_of(argv: list[str]) -> dict[str, str]:
    return dict(
        argument[2:].split("=", 1)
        for argument in argv
        if argument.startswith("--") and "=" in argument
    )


def _measure(tmp_path: Path, inputs: dict[str, Any], source: Path) -> Path:
    output = tmp_path / f"lengths-{source.name}"
    result = _run(
        "stage_05_measure_lengths.py",
        inputs["env"],
        output_dir=output,
        source_path=source,
        omegalax_repo=inputs["repo"],
        processor_snapshot=inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode == 0, result.stderr
    return output


def _record_flags(
    tmp_path: Path,
    inputs: dict[str, Any],
    source: Path,
    lengths: Path,
    **overrides: object,
) -> dict[str, object]:
    values: dict[str, object] = {
        "output_dir": tmp_path / "records",
        "source_path": source,
        "omegalax_repo": inputs["repo"],
        "processor_snapshot": inputs["snapshot"],
        "max_length": 4096,
        "records_per_shard": 8,
        "num_workers": 2,
        "message_lengths_path": lengths,
    }
    values.update(overrides)
    return values


def test_processor_snapshot_attests_the_consumed_qwen_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "model" / "snapshots" / SNAPSHOT_REVISION
    snapshot.mkdir(parents=True)
    for name in SNAPSHOT_FILES:
        value = {"merge_size": 2} if name == "preprocessor_config.json" else {}
        (snapshot / name).write_text(json.dumps(value))
    identity = attest_processor_snapshot(snapshot)
    assert set(identity["files"]) == SNAPSHOT_FILES
    assert identity["merge_size"] == 2
    (snapshot / "model.safetensors").write_bytes(b"unused weights")
    assert attest_processor_snapshot(snapshot) == identity
    (snapshot / "tokenizer.json").unlink()
    (snapshot / "tokenizer.json").symlink_to(snapshot / "missing")
    with pytest.raises(ValueError, match="files do not match"):
        attest_processor_snapshot(snapshot)


def test_whole_conversation_fit_preserves_conditioning_context(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "history"}],
            "loss": False,
        },
        {"role": "user", "content": [{"type": "text", "text": "target input"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "target"}]},
    ]
    chat = tmp_path / "chat.jsonl"
    chat.write_text(json.dumps({"messages": messages}) + "\n")
    cache = tmp_path / "message_lengths.jsonl"
    cache.write_text(
        "".join(
            json.dumps(
                {
                    "conv_idx": 0,
                    "msg_offset": offset,
                    "measurement": {"length": 10},
                }
            )
            + "\n"
            for offset in range(len(messages))
        )
    )

    require_conversations_fit(cache, chat, max_length=50)
    with pytest.raises(ValueError, match=r"conversations exceed max_length=40.*50"):
        require_conversations_fit(cache, chat, max_length=40)


def test_stage_05_refuses_missing_chat_before_launch(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    (source / "chat.jsonl").unlink()
    result = _run(
        "stage_05_measure_lengths.py",
        production_inputs["env"],
        output_dir=tmp_path / "lengths",
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode != 0
    assert "chat artifact is missing" in result.stderr
    assert _invocations(production_inputs["log"]) == []


def test_stage_05_refuses_chat_images_outside_the_attested_store(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    chat = source / "chat.jsonl"
    rows = [json.loads(line) for line in chat.read_text().splitlines()]
    rows[0]["messages"][1]["content"][0]["image"] = "ar:///outside.array_record#0"
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chat_sha256"] = hashlib.sha256(chat.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = _run(
        "stage_05_measure_lengths.py",
        production_inputs["env"],
        output_dir=tmp_path / "lengths",
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode != 0
    assert "outside its attested store" in result.stderr
    assert _invocations(production_inputs["log"]) == []


def test_stage_05_records_sealed_inputs_and_exact_cache(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    output = _measure(tmp_path, production_inputs, source)
    (argv,) = _script_invocations(production_inputs["log"])
    assert "--offline" in argv
    assert "--locked" in argv
    assert "-I" in argv
    flags = _flags_of(argv)
    assert flags["model_id"] == flags["processor"] == str(production_inputs["snapshot"].resolve())
    manifest = json.loads((output / "manifest.json").read_text())
    cache = output / "message_lengths.jsonl"
    assert manifest["inputs"]["source_id"] == make_artifact_id(source)
    assert manifest["params"]["omegalax"]["commit"] == COMMIT
    assert manifest["params"]["omegalax"]["tree"] == TREE
    assert manifest["params"]["processor_snapshot"]["revision"] == SNAPSHOT_REVISION
    assert "tokenizer.json" in manifest["params"]["processor_snapshot"]["files"]
    assert manifest["stats"]["cache"] == {
        "file": "message_lengths.jsonl",
        "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "n_messages": 6,
        "elapsed_s": 0,
    }


def test_stage_05_shards_equal_single_worker_output(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"], n_conversations=3)
    single = _measure(tmp_path, production_inputs, source)
    expected = (single / "message_lengths.jsonl").read_text()
    sharded = tmp_path / "sharded"
    common = {
        "output_dir": sharded,
        "source_path": source,
        "omegalax_repo": production_inputs["repo"],
        "processor_snapshot": production_inputs["snapshot"],
        "num_workers": 2,
        "num_shards": 2,
    }
    for index in range(2):
        result = _run(
            "stage_05_measure_lengths.py", production_inputs["env"], **common, shard_index=index
        )
        assert result.returncode == 0, result.stderr
        assert not (sharded / "manifest.json").exists()
    result = _run("stage_05_measure_lengths.py", production_inputs["env"], **common, merge=True)
    assert result.returncode == 0, result.stderr
    assert (sharded / "message_lengths.jsonl").read_text() == expected


def test_stage_05_resumes_only_matching_shard_receipt(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    output = _measure(tmp_path, production_inputs, source)
    before = len(_script_invocations(production_inputs["log"]))
    assert _measure(tmp_path, production_inputs, source) == output
    assert len(_script_invocations(production_inputs["log"])) == before
    receipt = output / "measure_receipt.shard0000_of_0001.json"
    value = json.loads(receipt.read_text())
    value["identity"]["source_sha256"] = "0" * 64
    receipt.write_text(json.dumps(value))
    assert _measure(tmp_path, production_inputs, source) == output
    assert len(_script_invocations(production_inputs["log"])) == before + 1


def test_stage_05_failed_merge_does_not_replace_canonical_cache(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"], n_conversations=3)
    output = tmp_path / "sharded"
    common = {
        "output_dir": output,
        "source_path": source,
        "omegalax_repo": production_inputs["repo"],
        "processor_snapshot": production_inputs["snapshot"],
        "num_workers": 2,
        "num_shards": 2,
    }
    for index in range(2):
        assert (
            _run(
                "stage_05_measure_lengths.py", production_inputs["env"], **common, shard_index=index
            ).returncode
            == 0
        )
    canonical = output / "message_lengths.jsonl"
    canonical.write_text("previous\n")
    shard = output / "message_lengths.shard0001_of_0002.jsonl"
    shard.write_text(shard.read_text().splitlines()[0] + "\n")
    result = _run("stage_05_measure_lengths.py", production_inputs["env"], **common, merge=True)
    assert result.returncode != 0
    assert "does not match receipt" in result.stderr
    assert canonical.read_text() == "previous\n"


def test_stage_05_manifest_failure_restores_sealed_artifact(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    output = _measure(tmp_path, production_inputs, source)
    manifest = (output / "manifest.json").read_bytes()
    cache = (output / "message_lengths.jsonl").read_bytes()
    harness = """
from pipeline import stage_05_measure_lengths as stage
def fail(*args, **kwargs):
    raise RuntimeError("injected manifest failure")
stage.write_manifest = fail
stage.app.run(stage.main)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            f"--output_dir={output}",
            f"--source_path={source}",
            f"--omegalax_repo={production_inputs['repo']}",
            f"--processor_snapshot={production_inputs['snapshot']}",
            "--num_workers=2",
            "--merge=true",
        ],
        env=production_inputs["env"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "injected manifest failure" in result.stderr
    assert (output / "manifest.json").read_bytes() == manifest
    assert (output / "message_lengths.jsonl").read_bytes() == cache


@pytest.mark.parametrize("mode", ["wrong_partition", "bad_counters", "duplicate"])
def test_stage_05_merge_rejects_resealed_bad_shards(
    tmp_path: Path, production_inputs, mode: str
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"], n_conversations=3)
    output = tmp_path / "sharded"
    common = {
        "output_dir": output,
        "source_path": source,
        "omegalax_repo": production_inputs["repo"],
        "processor_snapshot": production_inputs["snapshot"],
        "num_workers": 2,
        "num_shards": 2,
    }
    for index in range(2):
        assert (
            _run(
                "stage_05_measure_lengths.py", production_inputs["env"], **common, shard_index=index
            ).returncode
            == 0
        )
    shard = output / "message_lengths.shard0001_of_0002.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    if mode == "wrong_partition":
        rows[0]["conv_idx"] = 0
    elif mode == "bad_counters":
        rows[0]["measurement"]["vision_tokens"] += 1
    else:
        rows.insert(1, rows[0])
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = output / "measure_receipt.shard0001_of_0002.json"
    value = json.loads(receipt.read_text())
    value["cache"]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    value["cache"]["n_messages"] = len(rows)
    receipt.write_text(json.dumps(value))
    result = _run("stage_05_measure_lengths.py", production_inputs["env"], **common, merge=True)
    assert result.returncode != 0


def test_stage_05_merge_requires_every_shard_receipt(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    result = _run(
        "stage_05_measure_lengths.py",
        production_inputs["env"],
        output_dir=tmp_path / "output",
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
        num_shards=2,
        merge=True,
    )
    assert result.returncode != 0
    assert "missing shard receipt" in result.stderr


@pytest.mark.parametrize("worker_flag", [{"shard_index": 0}, {"work_dir": "/tmp/work"}])
def test_stage_05_merge_rejects_worker_flags(
    tmp_path: Path, production_inputs, worker_flag: dict[str, object]
) -> None:
    result = _run(
        "stage_05_measure_lengths.py",
        production_inputs["env"],
        output_dir=tmp_path / "output",
        source_path=tmp_path / "missing-source",
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
        merge=True,
        **worker_flag,
    )
    assert result.returncode != 0
    assert "merge does not accept worker flags" in result.stderr
    assert not (tmp_path / "output").exists()


def test_stage_05_failed_worker_preserves_sealed_artifact(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    output = _measure(tmp_path, production_inputs, source)
    manifest = (output / "manifest.json").read_bytes()
    cache = (output / "message_lengths.jsonl").read_bytes()
    receipt = output / "measure_receipt.shard0000_of_0001.json"
    receipt.unlink()
    result = _run(
        "stage_05_measure_lengths.py",
        {**production_inputs["env"], "FAKE_UV_CACHE": "malformed"},
        output_dir=output,
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode != 0
    assert (output / "manifest.json").read_bytes() == manifest
    assert (output / "message_lengths.jsonl").read_bytes() == cache


def test_stage_05_supports_empty_partitions(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    result = _run(
        "stage_05_measure_lengths.py",
        production_inputs["env"],
        output_dir=tmp_path / "output",
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
        num_shards=3,
        shard_index=2,
    )
    assert result.returncode == 0, result.stderr
    assert _script_invocations(production_inputs["log"]) == []
    receipt = json.loads(
        (tmp_path / "output" / "measure_receipt.shard0002_of_0003.json").read_text()
    )
    assert receipt["cache"]["n_conversations"] == 0
    assert receipt["cache"]["n_messages"] == 0


def test_stages_remove_interrupted_manifest_temps(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = tmp_path / "lengths-source"
    lengths.mkdir()
    (lengths / "manifest.json.tmp").write_text("interrupted")
    assert _measure(tmp_path, production_inputs, source) == lengths

    output = tmp_path / "records"
    output.mkdir()
    (output / "manifest.json.tmp").write_text("interrupted")
    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(
            tmp_path,
            production_inputs,
            source,
            lengths,
            output_dir=output,
        ),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("absent", "no cache"),
        ("empty", "cache keys"),
        ("malformed", "measurement"),
        ("bad_vision_tokens", "vision token count"),
    ],
)
def test_stage_05_rejects_invalid_cache(
    tmp_path: Path, production_inputs, mode: str, message: str
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    output = tmp_path / "lengths"
    output.mkdir()
    (output / "manifest.json").write_text("stale")
    env = {**production_inputs["env"], "FAKE_UV_CACHE": mode}
    result = _run(
        "stage_05_measure_lengths.py",
        env,
        output_dir=output,
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert (output / "manifest.json").read_text() == "stale"


def test_stage_05_rejects_compiler_changes_during_execution(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    env = {**production_inputs["env"], "FAKE_MUTATE_SNAPSHOT": "1"}
    output = tmp_path / "lengths"
    result = _run(
        "stage_05_measure_lengths.py",
        env,
        output_dir=output,
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode != 0
    assert "compiler identity changed" in result.stderr
    assert not (output / "manifest.json").exists()


def test_omegalax_attestation_rejects_consumed_untracked_files(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    env = {
        **production_inputs["env"],
        "FAKE_GIT_STATUS": "?? omegalax/injected.py",
    }
    result = _run(
        "stage_05_measure_lengths.py",
        env,
        output_dir=tmp_path / "lengths",
        source_path=source,
        omegalax_repo=production_inputs["repo"],
        processor_snapshot=production_inputs["snapshot"],
        num_workers=2,
    )
    assert result.returncode != 0
    assert "checkout has consumed changes" in result.stderr


def test_stage_06_requires_matching_source_compiler_processor_and_digest(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    other = make_source(tmp_path / "other", production_inputs["image_store"], n_conversations=3)
    other_lengths = _measure(tmp_path, production_inputs, other)
    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(tmp_path, production_inputs, source, other_lengths),
    )
    assert result.returncode != 0
    assert "cache source mismatch" in result.stderr

    lengths = _measure(tmp_path, production_inputs, source)
    manifest_path = lengths / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["params"]["processor_snapshot"]["revision"] = "d" * 40
    manifest_path.write_text(json.dumps(manifest))
    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(tmp_path, production_inputs, source, lengths),
    )
    assert result.returncode != 0
    assert "processor snapshot mismatch" in result.stderr

    lengths = _measure(tmp_path, production_inputs, source)
    with (lengths / "message_lengths.jsonl").open("a") as target:
        target.write("{}\n")
    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(tmp_path, production_inputs, source, lengths),
    )
    assert result.returncode != 0
    assert "cache digest mismatch" in result.stderr


def test_stage_06_builds_verified_nonempty_arrayrecords(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    output = tmp_path / "records"
    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(
            tmp_path,
            production_inputs,
            source,
            lengths,
            output_dir=output,
            val_fraction=0.3,
            max_length=30,
        ),
    )
    assert result.returncode == 0, result.stderr
    builds = [
        argv
        for argv in _script_invocations(production_inputs["log"])
        if any("build_sft_records" in argument for argument in argv)
    ]
    assert [_flags_of(argv)["split"] for argv in builds] == ["train", "val"]
    assert all("--offline" in argv and "--locked" in argv and "-I" in argv for argv in builds)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["inputs"]["source_id"] == make_artifact_id(source)
    assert manifest["inputs"]["message_lengths_id"] == make_artifact_id(lengths)
    for row in manifest["stats"]["per_split"]:
        shard = output / row["split"] / "part-00000.array_record"
        assert row["num_records"] == 1
        assert row["shards"] == {
            "part-00000.array_record": {
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "num_records": 1,
            }
        }
        assert {path.name for path in (output / row["split"]).iterdir()} == {
            "metadata.json",
            "part-00000.array_record",
        }


@pytest.mark.parametrize("mode", ["omit", "reverse", "wrong_length"])
def test_stage_06_requires_exact_ordered_source_chunks(
    tmp_path: Path, production_inputs, mode: str
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    result = _run(
        "stage_06_training_records.py",
        {**production_inputs["env"], "FAKE_UV_RECORDS": mode},
        **_record_flags(tmp_path, production_inputs, source, lengths),
    )
    assert result.returncode != 0
    assert (
        "expected source chunk" in result.stderr or "omit expected source chunks" in result.stderr
    )


def test_stage_06_validates_each_split_once(tmp_path: Path, production_inputs) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    flags = _record_flags(
        tmp_path,
        production_inputs,
        source,
        lengths,
        val_fraction=0.3,
    )
    result = _run(
        "stage_06_training_records.py",
        {**production_inputs["env"], "FAKE_MUTATE_TRAIN": "1"},
        **flags,
    )
    assert result.returncode != 0
    assert not (Path(flags["output_dir"]) / "manifest.json").exists()


def test_stage_06_rejects_oversized_conversations_before_replacing_outputs(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    cache = lengths / "message_lengths.jsonl"
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    for row in rows[:3]:
        row["measurement"]["length"] = 2000
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = lengths / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stats"]["cache"]["sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    output = tmp_path / "records"
    (output / "train").mkdir(parents=True)
    (output / "train" / "existing").write_text("keep")
    (output / "manifest.json").write_text("existing")
    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(tmp_path, production_inputs, source, lengths, output_dir=output),
    )
    assert result.returncode != 0
    assert "source conversations exceed" in result.stderr
    assert (output / "manifest.json").read_text() == "existing"
    assert (output / "train" / "existing").read_text() == "keep"
    builds = [
        argv
        for argv in _script_invocations(production_inputs["log"])
        if any("build_sft_records" in argument for argument in argv)
    ]
    assert builds == []


def test_stage_06_rejects_a_masked_target_before_replacing_outputs(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    chat = source / "chat.jsonl"
    rows = [json.loads(line) for line in chat.read_text().splitlines()]
    rows[0]["messages"][-1]["loss"] = False
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_manifest = source / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    manifest["chat_sha256"] = hashlib.sha256(chat.read_bytes()).hexdigest()
    source_manifest.write_text(json.dumps(manifest))
    output = tmp_path / "records"
    (output / "train").mkdir(parents=True)
    (output / "train" / "existing").write_text("keep")
    (output / "manifest.json").write_text("existing")

    result = _run(
        "stage_06_training_records.py",
        production_inputs["env"],
        **_record_flags(tmp_path, production_inputs, source, lengths, output_dir=output),
    )

    assert result.returncode != 0
    assert "supervision context" in result.stderr
    assert (output / "manifest.json").read_text() == "existing"
    assert (output / "train" / "existing").read_text() == "keep"
    builds = [
        argv
        for argv in _script_invocations(production_inputs["log"])
        if any("build_sft_records" in argument for argument in argv)
    ]
    assert builds == []


def test_stage_06_rejects_compiler_changes_during_execution(
    tmp_path: Path, production_inputs
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    output = tmp_path / "records"
    env = {**production_inputs["env"], "FAKE_MUTATE_REPO": "1"}
    result = _run(
        "stage_06_training_records.py",
        env,
        **_record_flags(
            tmp_path,
            production_inputs,
            source,
            lengths,
            output_dir=output,
        ),
    )
    assert result.returncode != 0
    assert "compiler identity changed" in result.stderr
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("absent", "output set"),
        ("empty", "counts must be positive"),
        ("corrupt", "invalid Omegalax ArrayRecord"),
        ("crc", "cannot read Omegalax record"),
        ("count_mismatch", "record count"),
        ("bad_metadata", "metadata values"),
        ("length_overflow", "expected source chunk"),
        ("unsupervised", "expected source chunk"),
        ("noncanonical", "noncanonical Omegalax record"),
        ("missing_sidecar", "output set"),
        ("extra_file", "output set"),
    ],
)
def test_stage_06_rejects_invalid_record_artifacts(
    tmp_path: Path, production_inputs, mode: str, message: str
) -> None:
    source = make_source(tmp_path / "source", production_inputs["image_store"])
    lengths = _measure(tmp_path, production_inputs, source)
    output = tmp_path / "records"
    output.mkdir()
    (output / "manifest.json").write_text("stale")
    env = {**production_inputs["env"], "FAKE_UV_RECORDS": mode}
    result = _run(
        "stage_06_training_records.py",
        env,
        **_record_flags(tmp_path, production_inputs, source, lengths, output_dir=output),
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert not (output / "manifest.json").exists()
