from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Bag:
    def __init__(self) -> None:
        object.__setattr__(self, "_items", {})

    def __getattr__(self, name: str) -> _Bag:
        items = object.__getattribute__(self, "_items")
        if name not in items:
            items[name] = _Bag()
        return items[name]

    def __setattr__(self, name: str, value: object) -> None:
        object.__getattribute__(self, "_items")[name] = value

    def to_dict(self) -> dict:
        return {
            key: value.to_dict() if isinstance(value, _Bag) else value
            for key, value in object.__getattribute__(self, "_items").items()
        }


@pytest.fixture
def chain(monkeypatch, tmp_path: Path):
    schema = types.ModuleType("pmanager.configs.schema")
    schema.pipeline_task = _Bag
    configs = types.ModuleType("pmanager.configs")
    pmanager = types.ModuleType("pmanager")
    monkeypatch.setitem(sys.modules, "pmanager", pmanager)
    monkeypatch.setitem(sys.modules, "pmanager.configs", configs)
    monkeypatch.setitem(sys.modules, "pmanager.configs.schema", schema)

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    master = tmp_path / "master"
    master.mkdir()
    source_id = "/source::0123456789abcdef"
    source_sha256 = "0" * 64
    (master / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "juergen_annotation_frames_master",
                "schema_version": 1,
                "target_height": 720,
                "jpeg_quality": 92,
                "source_clips_id": source_id,
                "source_clips_sha256": source_sha256,
            }
        )
    )
    realigned = tmp_path / "realigned"
    realigned.mkdir()
    clips = realigned / "clips_manifest.jsonl"
    clips.write_text(
        json.dumps(
            {
                "segment_id": "s",
                "alignment_closed": True,
                "alignment_status": "aligned",
            }
        )
        + "\n"
    )
    (realigned / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "juergen_annotation_clip_manifest_realigned",
                "schema_version": 1,
                "clips_file": clips.name,
                "clips_sha256": hashlib.sha256(clips.read_bytes()).hexdigest(),
                "source_clips_id": source_id,
                "source_clips_sha256": source_sha256,
            }
        )
    )
    omegalax = tmp_path / "omegalax"
    scripts = omegalax / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "measure_message_lengths_from_chat.py",
        "build_sft_records_from_chat.py",
    ):
        (scripts / name).touch()
    snapshot = tmp_path / "model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    for name in (
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ):
        value = {"merge_size": 2} if name == "preprocessor_config.json" else {}
        (snapshot / name).write_text(json.dumps(value))

    monkeypatch.setenv("LABCTL_DATASETS_ROOT", str(datasets))
    monkeypatch.setenv("JUERGEN_REPO", str(REPO_ROOT))
    monkeypatch.setenv("CROWDCAST_MASTER_DIR", str(master))
    monkeypatch.setenv("CROWDCAST_CLIPS_MANIFEST", str(clips))
    monkeypatch.setenv("OMEGALAX_REPO", str(omegalax))
    monkeypatch.setenv("SFT_PROCESSOR_SNAPSHOT", str(snapshot))
    sys.modules.pop("configs.chain_crowdcast", None)
    module = importlib.import_module("configs.chain_crowdcast")
    monkeypatch.setattr(
        module,
        "attest_omegalax",
        lambda path, processor_snapshot: {"path": str(path)},
    )
    return module, master, clips


def _child(config: _Bag | dict) -> dict:
    return config.children[0] if isinstance(config, _Bag) else config["children"][0]


def test_chain_declares_stage01_and_stage02_prerequisites(chain):
    module, master, clips = chain
    stage_03 = module.get_config()
    stage_03b = _child(stage_03)
    stage_04 = _child(stage_03b)
    stage_05 = _child(stage_04)
    stage_06 = _child(stage_05)

    assert stage_03.entrypoint.args.frames_master_dir == str(master)
    assert stage_03.entrypoint.args.clips_manifest == str(clips)
    assert stage_03.entrypoint.path == "pipeline/stage_03_filter.py"
    assert stage_03b["entrypoint"]["path"] == "pipeline/annotation/stage_annotate.py"
    assert stage_04["entrypoint"]["path"] == "pipeline/stage_04_build_conversations.py"
    assert stage_05["entrypoint"]["path"] == "pipeline/stage_05_measure_lengths.py"
    assert stage_06["entrypoint"]["path"] == "pipeline/stage_06_training_records.py"
    assert stage_03b["entrypoint"]["args"]["filter_dir"].endswith(
        "crowdcast_canonical_v1_stage_03_filter"
    )
    assert stage_04["entrypoint"]["args"]["goals_dir"].endswith(
        "crowdcast_canonical_v1_stage_03b_describe_extract"
    )


def test_chain_refuses_a_non_q92_master(chain):
    module, master, _ = chain
    manifest = json.loads((master / "manifest.json").read_text())
    manifest["jpeg_quality"] = 80
    (master / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Stage01 contract mismatch"):
        module.get_config()


def test_chain_refuses_a_noncanonical_stage02_file(chain, monkeypatch, tmp_path: Path):
    module, _, _ = chain
    alternate = tmp_path / "realigned" / "alternate.jsonl"
    alternate.touch()
    monkeypatch.setenv("CROWDCAST_CLIPS_MANIFEST", str(alternate))
    with pytest.raises(RuntimeError, match="canonical clips file"):
        module.get_config()


def test_chain_refuses_unclosed_stage02_alignment(chain):
    module, _, clips = chain
    clips.write_text(
        json.dumps(
            {
                "segment_id": "s",
                "alignment_closed": False,
                "alignment_status": "needs_review",
            }
        )
        + "\n"
    )
    manifest_path = clips.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["clips_sha256"] = hashlib.sha256(clips.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unclosed alignment"):
        module.get_config()


def test_chain_refuses_mismatched_stage00_provenance(chain):
    module, master, _ = chain
    manifest_path = master / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_clips_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="source inventories differ"):
        module.get_config()
