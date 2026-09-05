from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import grammars
import msgpack
import pytest
import synthetic_clip as clip
from grammars.deltatype_v2 import CODEC
from image_domain import jpeg_q92_height_domain
from PIL import Image

from pipeline.annotation import stage_annotate
from pipeline.annotation.lib.labeler import LabelResult
from pipeline.lib.image_store import make_arrayrecord_image_uri, read_jpeg_bytes
from pipeline.lib.manifest import (
    file_sha256_short,
    make_artifact_id,
    resolve_chat_artifact,
)
from pipeline.lib.source_clips import resolve_source_clips
from pipeline.lib.views import FilterArtifact
from pipeline.stage_01_master_frames import build_segment_master

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PIPELINE_DIR = REPO_ROOT / "data_pipeline"
STAGES = REPO_ROOT / "pipeline"


def _run_stage(script: str, *args: object) -> None:
    roots = [REPO_ROOT, DATA_PIPELINE_DIR]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, roots)))
    process = subprocess.run(
        [sys.executable, str(STAGES / script), *map(str, args)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_stage_00_seals_noncanonical_and_orphan_inputs(tmp_path: Path):
    source = clip.build_uploads_tree(tmp_path / "source")
    orphan = source["keylog_path"].with_name("input_orphan_seg0000.msgpack")
    orphan.write_bytes(b"orphan")
    test_video = source["video_path"].with_name("test_video.mp4")
    shutil.copy2(source["video_path"], test_video)
    test_keylog = source["keylog_path"].with_name("test_keylogs.msgpack")
    test_keylog.write_bytes(b"fixture")
    output = tmp_path / "stage_00" / "clips_manifest.jsonl"
    process = subprocess.run(
        [
            sys.executable,
            str(STAGES / "stage_00_clip_manifest.py"),
            "--dataset-root",
            str(tmp_path / "source"),
            "--out",
            str(output),
            "--workers",
            "1",
        ],
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    exclusions = _jsonl(output.parent / "exclusions.jsonl")
    assert {row["reason"] for row in exclusions} == {
        "noncanonical_keylog_name",
        "noncanonical_video_name",
        "orphan_keylog",
    }
    manifest = json.loads((output.parent / "manifest.json").read_text())
    assert manifest["n_segments"] == 1
    assert manifest["n_source_videos"] == 2
    assert manifest["n_source_keylogs"] == 3
    assert manifest["exclusion_counts"] == {
        "noncanonical_keylog_name": 1,
        "noncanonical_video_name": 1,
        "orphan_keylog": 1,
    }
    resolve_source_clips(output)


def test_stage_00_excludes_an_empty_keylog_before_realign(tmp_path: Path):
    source = clip.build_uploads_tree(tmp_path / "source")
    duplicate_video = source["video_path"].with_name(f"recording_{clip.RECORDING_ID}_seg0001.mp4")
    duplicate_keylog = source["keylog_path"].with_name(f"input_{clip.RECORDING_ID}_seg0001.msgpack")
    shutil.copy2(source["video_path"], duplicate_video)
    duplicate_keylog.write_bytes(msgpack.packb([]))
    output = tmp_path / "stage_00" / "clips_manifest.jsonl"

    _run_stage(
        "stage_00_clip_manifest.py",
        "--dataset-root",
        source["dataset_root"],
        "--out",
        output,
        "--workers",
        1,
    )

    assert [row["segment_id"] for row in _jsonl(output)] == [clip.SEGMENT_ID]
    assert _jsonl(output.parent / "exclusions.jsonl")[0]["reason"] == "empty_keylog"
    manifest = json.loads((output.parent / "manifest.json").read_text())
    assert manifest["exclusion_counts"] == {"empty_keylog": 1}
    resolve_source_clips(output)


class Chain:
    def __init__(self, root: Path) -> None:
        self.source = clip.build_uploads_tree(root / "uploads")
        self.stage_00 = root / "stage_00" / "clips_manifest.jsonl"
        _run_stage(
            "stage_00_clip_manifest.py",
            "--dataset-root",
            root / "uploads",
            "--out",
            self.stage_00,
            "--workers",
            1,
        )
        self.clip_rows = _jsonl(self.stage_00)
        self.stage_02 = root / "stage_02"
        _run_stage(
            "stage_02_realign.py",
            "--clips-manifest",
            self.stage_00,
            "--output-dir",
            self.stage_02,
            "--num-workers",
            1,
        )
        self.realigned = self.stage_02 / "clips_manifest.jsonl"
        self.master = clip.build_master_store(
            root / "stage_01",
            self.clip_rows[0],
            self.source["frames"],
            self.stage_00,
        )
        self.filter = root / "stage_03"
        _run_stage(
            "stage_03_filter.py",
            "--frames_master_dir",
            self.master,
            "--clips_manifest",
            self.realigned,
            "--output_dir",
            self.filter,
            "--num_workers",
            1,
        )
        self.goals = clip.write_goals(
            root / "goals",
            [
                clip.goal_row("g_head", 0, 12, "open the synthetic thing"),
                clip.goal_row("g_mid", 13, 40, "finish the synthetic thing"),
            ],
            master_store_id=make_artifact_id(self.master),
            filter_id=make_artifact_id(self.filter),
        )
        self.conversations = root / "stage_04"
        _run_stage(
            "stage_04_build_conversations.py",
            "--filter_dir",
            self.filter,
            "--goals_dir",
            self.goals,
            "--fps",
            clip.TRAIN_FPS,
            "--output_dir",
            self.conversations,
            "--num_workers",
            1,
        )


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Chain:
    return Chain(tmp_path_factory.mktemp("crowdcast"))


def _assistant_texts(row: dict[str, Any]) -> list[str]:
    return [
        message["content"][0]["text"]
        for message in row["messages"]
        if message["role"] == "assistant"
    ]


def _images(row: dict[str, Any]) -> list[str]:
    return [
        part["image"]
        for message in row["messages"]
        if message["role"] == "user"
        for part in message["content"]
        if part["type"] == "image"
    ]


def _terminal_keyboard_state(row: dict[str, Any]) -> tuple[list[str], set[str]]:
    held: set[str] = set()
    orphaned: list[str] = []
    for text in _assistant_texts(row):
        for element in CODEC.parse(text).elements:
            if element.kind != "event":
                continue
            if element.pressed:
                held.add(element.name)
            elif element.name in held:
                held.remove(element.name)
            else:
                orphaned.append(element.name)
    return orphaned, held


def test_discovery_realign_and_filter_artifacts_are_joined(chain: Chain):
    (source,) = chain.clip_rows
    (realigned,) = _jsonl(chain.realigned)
    assert source["segment_id"] == clip.SEGMENT_ID
    assert realigned["raw_keylog_path"] == source["keylog_path"]
    assert realigned["alignment_status"] == "aligned"
    source_id = make_artifact_id(chain.stage_00.parent)
    source_sha256 = hashlib.sha256(chain.stage_00.read_bytes()).hexdigest()
    realigned_manifest = json.loads((chain.stage_02 / "manifest.json").read_text())
    master_manifest = json.loads((chain.master / "manifest.json").read_text())
    assert realigned_manifest["source_clips_id"] == source_id
    assert master_manifest["source_clips_id"] == source_id
    assert realigned_manifest["source_clips_sha256"] == source_sha256
    assert master_manifest["source_clips_sha256"] == source_sha256
    filter_manifest = json.loads((chain.filter / "manifest.json").read_text())
    assert filter_manifest["master_store_id"] == make_artifact_id(chain.master)
    segment = json.loads((chain.filter / "filter" / f"{clip.SEGMENT_ID}.json").read_text())
    assert {item["reason"] for item in segment["dropped"]} == {
        "black",
        "idle_interior",
    }


def test_stage_04_is_only_goal_conditioned_canonical_deltatype(chain: Chain):
    rows = _jsonl(chain.conversations / "chat.jsonl")
    assert len(rows) == 2
    assert {row["goal_id"] for row in rows} == {"g_head", "g_mid"}
    prompt = grammars.describe("deltatype_v2")
    for row in rows:
        assert row["action_format"] == "canonical"
        assert row["messages"][0]["content"][0]["text"] == prompt
        assert row["messages"][1]["content"][0]["text"] == row["instruction"]
        assert all(CODEC.format(CODEC.parse(text)) == text for text in _assistant_texts(row))


def test_goal_slices_never_emit_a_release_without_its_press(chain: Chain):
    rows = _jsonl(chain.conversations / "chat.jsonl")
    assert all(_terminal_keyboard_state(row) == ([], set()) for row in rows)
    mid = next(row for row in rows if row["goal_id"] == "g_mid")
    first = CODEC.parse(_assistant_texts(mid)[0])
    assert any(
        element.kind == "event" and element.name == "KeyA" and element.pressed
        for element in first.elements
    )


def test_stage_04_attests_prompt_inputs_and_q92_images(chain: Chain):
    manifest = json.loads((chain.conversations / "manifest.json").read_text())
    assert manifest["artifact_type"] == "crowdcast_stage_04_conversations"
    assert manifest["master_store_id"] == make_artifact_id(chain.master)
    assert manifest["filter_id"] == make_artifact_id(chain.filter)
    assert manifest["goals_id"] == make_artifact_id(chain.goals)
    assert manifest["grammar"] == "deltatype_v2"
    prompt = grammars.describe("deltatype_v2")
    assert manifest["system_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert manifest["image_domain"] == jpeg_q92_height_domain(clip.FRAME_H)
    assert resolve_chat_artifact(chain.conversations) == chain.conversations / "chat.jsonl"
    for image in _images(_jsonl(chain.conversations / "chat.jsonl")[0]):
        with Image.open(io.BytesIO(read_jpeg_bytes(image))) as frame:
            assert frame.format == "JPEG"


def test_chat_resolver_refuses_a_forged_crowdcast_image(chain: Chain, tmp_path: Path):
    artifact = tmp_path / "conversations"
    shutil.copytree(chain.conversations, artifact)
    chat = artifact / "chat.jsonl"
    rows = _jsonl(chat)
    image = next(
        part
        for message in rows[0]["messages"]
        for part in message["content"]
        if part["type"] == "image"
    )
    image["image"] = "ar:///outside.array_record#0"
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chat_sha256"] = hashlib.sha256(chat.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="chat rows do not match"):
        resolve_chat_artifact(artifact)


def test_stage_04_requires_the_describe_extract_artifact_contract(chain: Chain, tmp_path: Path):
    manifest_path = chain.goals / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["method"] = "other_method"
    bad = tmp_path / "bad-goals"
    bad.mkdir()
    (bad / "manifest.json").write_text(json.dumps(manifest))
    (bad / "goals.jsonl").write_bytes((chain.goals / "goals.jsonl").read_bytes())
    process = subprocess.run(
        [
            sys.executable,
            str(STAGES / "stage_04_build_conversations.py"),
            "--filter_dir",
            str(chain.filter),
            "--goals_dir",
            str(bad),
            "--fps",
            str(clip.TRAIN_FPS),
            "--output_dir",
            str(tmp_path / "out"),
        ],
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "goals contract mismatch" in process.stderr


def test_stage_04_rejects_mutated_goals_and_filter_payloads(chain: Chain, tmp_path: Path):
    goals = tmp_path / "goals"
    shutil.copytree(chain.goals, goals)
    with (goals / "goals.jsonl").open("a") as target:
        target.write("{}\n")
    process = subprocess.run(
        [
            sys.executable,
            str(STAGES / "stage_04_build_conversations.py"),
            "--filter_dir",
            str(chain.filter),
            "--goals_dir",
            str(goals),
            "--fps",
            str(clip.TRAIN_FPS),
            "--output_dir",
            str(tmp_path / "out"),
        ],
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "goals digest mismatch" in process.stderr

    segment_path = chain.filter / "filter" / f"{clip.SEGMENT_ID}.json"
    original = segment_path.read_bytes()
    try:
        segment_path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="filter digest mismatch"):
            FilterArtifact(chain.filter).load_segment(clip.SEGMENT_ID)
    finally:
        segment_path.write_bytes(original)


def test_stage_04_refuses_an_unprojectable_goal(chain: Chain, tmp_path: Path):
    goals = tmp_path / "goals"
    shutil.copytree(chain.goals, goals)
    goals_path = goals / "goals.jsonl"
    rows = _jsonl(goals_path)
    rows[0]["start_master_idx"] = 10_000
    rows[0]["end_master_idx"] = 10_001
    rows.sort(key=lambda row: (row["start_master_idx"], row["end_master_idx"]))
    goals_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = goals / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["goals_sha256"] = hashlib.sha256(goals_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    process = subprocess.run(
        [
            sys.executable,
            str(STAGES / "stage_04_build_conversations.py"),
            "--filter_dir",
            str(chain.filter),
            "--goals_dir",
            str(goals),
            "--fps",
            str(clip.TRAIN_FPS),
            "--output_dir",
            str(tmp_path / "out"),
            "--num_workers",
            "1",
        ],
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "goal projection failed" in process.stderr


def test_stage_01_cache_validates_closed_payload(chain: Chain, tmp_path: Path):
    source_segment = chain.master / "frames" / clip.SEGMENT_ID
    frames_dir = tmp_path / "frames"
    segment_dir = frames_dir / clip.SEGMENT_ID
    segment_dir.mkdir(parents=True)
    for name in ("images.array_record", "frame_manifest.jsonl"):
        shutil.copy2(source_segment / name, segment_dir / name)
    copied_shard = segment_dir / "images.array_record"
    copied_manifest = segment_dir / "frame_manifest.jsonl"
    copied_rows = _jsonl(copied_manifest)
    for index, row in enumerate(copied_rows):
        row["shard_path"] = str(copied_shard)
        row["image"] = make_arrayrecord_image_uri(copied_shard, index)
    copied_manifest.write_text("".join(json.dumps(row) + "\n" for row in copied_rows))
    (index_row,) = _jsonl(chain.master / "segment_index.jsonl")
    inputs = {
        "jpeg_quality": index_row["jpeg_quality"],
        "master_fps": index_row["master_fps"],
        "target_height": index_row["target_height"],
        "video_sha256": index_row["video_sha256"],
    }
    outputs = {
        key: index_row[key]
        for key in (
            "frame_manifest_sha256",
            "num_records",
            "shard_sha256",
            "total_jpeg_bytes",
        )
    }
    outputs["frame_manifest_sha256"] = file_sha256_short(copied_manifest, n=64)
    (segment_dir / "segment_manifest.json").write_text(
        json.dumps({"schema_version": 1, "inputs": inputs, "outputs": outputs})
    )
    task = {
        "row": chain.clip_rows[0],
        "frames_dir": str(frames_dir),
        "master_fps": index_row["master_fps"],
        "target_height": index_row["target_height"],
        "jpeg_quality": index_row["jpeg_quality"],
        "ffmpeg_bin": "must-not-run",
    }
    assert build_segment_master(task)["status"] == "ok"
    source_video = Path(chain.clip_rows[0]["video_path"])
    original_video = source_video.read_bytes()
    try:
        source_video.write_bytes(original_video + b"mutated")
        with pytest.raises(ValueError, match="source video digest mismatch"):
            build_segment_master(task)
    finally:
        source_video.write_bytes(original_video)
    with (segment_dir / "frame_manifest.jsonl").open("a") as target:
        target.write("{}\n")
    with pytest.raises(ValueError, match="frame manifest digest mismatch"):
        build_segment_master(task)


def test_real_describe_extract_artifact_builds_stage_04(chain: Chain, tmp_path: Path, monkeypatch):
    class FakeLabeler:
        def __init__(self, config):
            self.config = config

        def call_full(self, *args, **kwargs):
            return LabelResult("Observed work.", "", "stop", {"total_tokens": 3}, self.config.model)

        def call_json_full(self, *args, **kwargs):
            goal = {
                "instruction": "complete the observed work",
                "anchor": "The user starts the work.",
                "grounding": "The screen shows the work complete.",
                "start_frame": 0,
                "end_frame": 1,
            }
            return {"goals": [goal]}, LabelResult(
                json.dumps({"goals": [goal]}),
                "",
                "stop",
                {"total_tokens": 5},
                self.config.model,
            )

    monkeypatch.setattr(stage_annotate, "Labeler", FakeLabeler)
    monkeypatch.setenv("LABELER_BASE_URL", "https://labeler.example/v1")
    monkeypatch.setenv("LABELER_API_KEY", "secret")
    goals = tmp_path / "annotated-goals"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_annotate.py",
            "--filter_dir",
            str(chain.filter),
            "--fps",
            str(clip.TRAIN_FPS),
            "--output_dir",
            str(goals),
            "--model",
            "test-model",
            "--target_tpm",
            "100000",
            "--max_workers",
            "1",
        ],
    )
    stage_annotate.main()
    produced = _jsonl(goals / "goals.jsonl")
    assert len(produced) == 1
    assert set(produced[0]) == {
        "goal_id",
        "segment_id",
        "recording_id",
        "start_master_idx",
        "end_master_idx",
        "instruction",
        "anchor",
        "grounding",
        "method",
        "model",
        "prompt_pack_sha",
    }

    conversations = tmp_path / "annotated-conversations"
    _run_stage(
        "stage_04_build_conversations.py",
        "--filter_dir",
        chain.filter,
        "--goals_dir",
        goals,
        "--fps",
        clip.TRAIN_FPS,
        "--output_dir",
        conversations,
        "--num_workers",
        1,
    )
    rows = _jsonl(conversations / "chat.jsonl")
    assert len(rows) == 1
    assert rows[0]["instruction"] == "complete the observed work"
