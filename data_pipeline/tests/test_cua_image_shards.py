from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest
from image_domain import encode_jpeg_q92
from PIL import Image

from pipeline.cua_gym import stage_01_image_store
from pipeline.cua_gym.stage_01_image_store import (
    build_inventory_shard,
    build_store,
    finalize_store,
    prepare_inventory,
    validate_image_store,
)
from pipeline.cua_gym.stage_04_build_conversations import ImageIndex
from pipeline.lib.image_store import read_jpeg_bytes


def _tar(root: Path, index: int, color: tuple[int, int, int]) -> None:
    root.mkdir(exist_ok=True)
    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), color).save(buffer, "PNG")
    payload = buffer.getvalue()
    with tarfile.open(root / f"screenshots-{index:04d}.tar", "w") as archive:
        info = tarfile.TarInfo(f"task-{index}/step_000.png")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def _inventory(sources: Path, output: Path) -> tuple[Path, str]:
    path = prepare_inventory(sources, output)
    return path, json.loads(path.read_text())["inventory_sha256"]


def test_shards_resume_independently(tmp_path: Path):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    _tar(sources, 1, (40, 50, 60))
    inventory, digest = _inventory(sources, output)
    first = build_inventory_shard(inventory, digest, output, 0, workers=1)
    with mock.patch(
        "pipeline.cua_gym.stage_01_image_store._transcode",
        side_effect=AssertionError("resumed shard was transcoded"),
    ):
        assert build_inventory_shard(inventory, digest, output, 0, workers=1) == first
    with pytest.raises(ValueError, match="incomplete image shard"):
        finalize_store(inventory, digest, output)
    build_inventory_shard(inventory, digest, output, 1, workers=1)
    assert finalize_store(inventory, digest, output)["num_tars"] == 2
    with mock.patch(
        "array_record.python.array_record_module.ArrayRecordReader.read",
        side_effect=AssertionError("consumer replayed JPEG bytes"),
    ):
        assert ImageIndex(output).uri("screenshots-0000.tar", "task-0/step_000.png")
    uri = ImageIndex(output).uri("screenshots-0000.tar", "task-0/step_000.png")
    expected = encode_jpeg_q92(Image.new("RGB", (1920, 1080), (10, 20, 30)))
    assert read_jpeg_bytes(uri) == expected


def test_changed_source_rebuilds_only_affected_shard(tmp_path: Path):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    _tar(sources, 1, (40, 50, 60))
    inventory, digest = _inventory(sources, output)
    old = [build_inventory_shard(inventory, digest, output, i, workers=1) for i in range(2)]
    finalize_store(inventory, digest, output)
    old_uri = ImageIndex(output).uri("screenshots-0001.tar", "task-1/step_000.png")
    _tar(sources, 1, (70, 80, 90))
    inventory, digest = _inventory(sources, output)
    with mock.patch(
        "pipeline.cua_gym.stage_01_image_store._transcode",
        side_effect=AssertionError("unchanged shard was transcoded"),
    ):
        assert build_inventory_shard(inventory, digest, output, 0, workers=1) == old[0]
    changed = build_inventory_shard(inventory, digest, output, 1, workers=1)
    assert changed["directory"] != old[1]["directory"]
    assert ImageIndex(output).uri("screenshots-0001.tar", "task-1/step_000.png") == old_uri


def test_finalizer_refuses_corrupt_receipt(tmp_path: Path):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    inventory, digest = _inventory(sources, output)
    receipt = build_inventory_shard(inventory, digest, output, 0, workers=1)
    receipt_path = output / "shards" / receipt["directory"] / "receipt.json"
    corrupt = json.loads(receipt_path.read_text())
    corrupt["arrayrecord_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="incomplete image shard"):
        finalize_store(inventory, digest, output)


def test_encoding_contract_change_rebuilds_shard(tmp_path: Path, monkeypatch):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    inventory, digest = _inventory(sources, output)
    old = build_inventory_shard(inventory, digest, output, 0, workers=1)
    monkeypatch.setattr(stage_01_image_store, "ENCODING_SHA256", "0" * 64)
    inventory, digest = _inventory(sources, output)
    new = build_inventory_shard(inventory, digest, output, 0, workers=1)
    assert new["directory"] != old["directory"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "swapped", "stale"])
def test_manifest_shards_are_bound_to_exact_inventory(tmp_path: Path, mutation: str):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    _tar(sources, 1, (40, 50, 60))
    inventory, digest = _inventory(sources, output)
    for index in range(2):
        build_inventory_shard(inventory, digest, output, index, workers=1)
    manifest = finalize_store(inventory, digest, output)
    if mutation == "missing":
        manifest["shards"].pop("screenshots-0001")
    elif mutation == "extra":
        manifest["shards"]["screenshots-9999"] = manifest["shards"]["screenshots-0000"]
    elif mutation == "swapped":
        left = manifest["shards"]["screenshots-0000"]
        manifest["shards"]["screenshots-0000"] = manifest["shards"]["screenshots-0001"]
        manifest["shards"]["screenshots-0001"] = left
    else:
        receipt = manifest["shards"]["screenshots-0000"]
        receipt["source_size"] += 1
        receipt_path = output / "shards" / receipt["directory"] / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_image_store(output)


def test_worker_and_finalizer_reject_external_inventory(tmp_path: Path):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    inventory, digest = _inventory(sources, output)
    external = tmp_path / inventory.name
    shutil.copyfile(inventory, external)
    with pytest.raises(ValueError, match="directly under output_dir"):
        build_inventory_shard(external, digest, output, 0, workers=1)
    with pytest.raises(ValueError, match="directly under output_dir"):
        finalize_store(external, digest, output)


def test_invalid_producer_cleans_its_temporary_directory(tmp_path: Path):
    sources, output = tmp_path / "sources", tmp_path / "output"
    _tar(sources, 0, (10, 20, 30))
    inventory, digest = _inventory(sources, output)
    with (
        mock.patch(
            "pipeline.cua_gym.stage_01_image_store._transcode",
            side_effect=ValueError("invalid PNG"),
        ),
        pytest.raises(ValueError, match="invalid PNG"),
    ):
        build_inventory_shard(inventory, digest, output, 0, workers=1)
    assert not list((output / "shards").glob("*.tmp"))


def test_cli_rejects_workers_outside_worker_invocation(tmp_path: Path, monkeypatch):
    screenshots, output = tmp_path / "screenshots", tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_01_image_store.py",
            "--screenshots_dir",
            str(screenshots),
            "--output_dir",
            str(output),
            "--workers",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="only valid with --tar_index"):
        stage_01_image_store.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_01_image_store.py",
            "--inventory",
            str(output / "source_inventory-digest.json"),
            "--inventory_sha256",
            "digest",
            "--finalize",
            "--output_dir",
            str(output),
            "--workers",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="only valid with --tar_index"):
        stage_01_image_store.main()


def test_build_store_rejects_workers_before_inventory(tmp_path: Path):
    with (
        mock.patch(
            "pipeline.cua_gym.stage_01_image_store.prepare_inventory",
            side_effect=AssertionError("inventory was hashed"),
        ),
        pytest.raises(ValueError, match="workers must be positive"),
    ):
        build_store(tmp_path / "screenshots", tmp_path / "output", workers=0)
