from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import grammars
import pytest
import synthetic_clip as clip
from grammars.deltatype_v2 import CODEC
from image_domain import image_domain

from pipeline.annotation import stage_annotate
from pipeline.annotation.lib.labeler import LabelResult
from pipeline.lib import config
from pipeline.lib.image_store import open_image_pil
from pipeline.lib.manifest import make_artifact_id, resolve_chat_artifact
from pipeline.lib.views import FilterArtifact
from pipeline.stage_01_master_frames import build_segment_master

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PIPELINE_DIR = REPO_ROOT / "data_pipeline"
STAGES = REPO_ROOT / "pipeline"


def _run_stage(script: str, *args: object) -> None:
    roots = [REPO_ROOT, DATA_PIPELINE_DIR, REPO_ROOT.parent / "desktop"]
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
            root / "stage_01", self.clip_rows[0], self.source["frames"]
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
    assert manifest["image_domain"] == image_domain(
        media="jpeg",
        quality=config.DEFAULT_JPEG_QUALITY,
        geometry="height",
        extent=clip.FRAME_H,
    )
    assert resolve_chat_artifact(chain.conversations) == chain.conversations / "chat.jsonl"
    for image in _images(_jsonl(chain.conversations / "chat.jsonl")[0]):
        with open_image_pil(image) as frame:
            assert frame.format == "JPEG"


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


def test_stage_01_cache_validates_closed_payload(chain: Chain, tmp_path: Path):
    source_segment = chain.master / "frames" / clip.SEGMENT_ID
    frames_dir = tmp_path / "frames"
    segment_dir = frames_dir / clip.SEGMENT_ID
    segment_dir.mkdir(parents=True)
    for name in ("images.array_record", "frame_manifest.jsonl"):
        shutil.copy2(source_segment / name, segment_dir / name)
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
        "force": False,
    }
    assert build_segment_master(task)["status"] == "ok"
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
