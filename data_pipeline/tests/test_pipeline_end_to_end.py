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
from grammars.deltatype_v2 import CODEC
from image_domain import image_domain

from pipeline.lib import config
from pipeline.lib.image_store import open_image_pil
from pipeline.lib.manifest import make_artifact_id

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


def _orphaned_releases(row: dict[str, Any]) -> list[str]:
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
    return orphaned


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
    assert len(rows) == 6
    assert {row["goal_id"] for row in rows} == {"g_head", "g_mid"}
    assert {row["variant_idx"] for row in rows} == {0, 1, 2}
    prompt = grammars.describe("deltatype_v2")
    for row in rows:
        assert row["action_format"] == "canonical"
        assert row["messages"][0]["content"][0]["text"] == prompt
        assert row["messages"][1]["content"][0]["text"] == row["instruction"]
        assert all(CODEC.format(CODEC.parse(text)) == text for text in _assistant_texts(row))


def test_goal_slices_never_emit_a_release_without_its_press(chain: Chain):
    rows = _jsonl(chain.conversations / "chat.jsonl")
    assert all(_orphaned_releases(row) == [] for row in rows)
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
