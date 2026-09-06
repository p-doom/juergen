"""Build independently resumable CUA-Gym JPEG ArrayRecord shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from cua_parity_contract import JPEG_QUALITY, OBSERVATION_CONTRACT, OBSERVATION_SIZE
from image_domain import encode_jpeg_q92, validate_jpeg_q92
from pipeline.lib.image_store import (
    make_arrayrecord_image_uri,
    parse_arrayrecord_image_uri,
)

SCREEN = OBSERVATION_SIZE
SHARD_NAME, INDEX_NAME, RECEIPT_NAME = (
    "images.array_record",
    "index.jsonl",
    "receipt.json",
)
INVENTORY_PREFIX, SCHEMA_VERSION = "source_inventory", 2
ENCODING_CONTRACT = {
    "image_domain": OBSERVATION_CONTRACT,
    "jpeg_quality": JPEG_QUALITY,
    "jpeg_subsampling": "4:2:0",
    "width": SCREEN[0],
    "height": SCREEN[1],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


ENCODING_SHA256 = _canonical_sha256(ENCODING_CONTRACT)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_inventory(screenshots_dir: Path, output_dir: Path) -> Path:
    sources = sorted(screenshots_dir.glob("screenshots-*.tar"))
    if not sources:
        raise ValueError(f"no screenshots-*.tar files under {screenshots_dir}")
    entries = [
        {
            "name": p.name,
            "path": str(p.resolve()),
            "size": p.stat().st_size,
            "sha256": _sha256(p),
        }
        for p in sources
    ]
    payload = {
        "artifact_type": "cuagym_stage_01_source_inventory",
        "schema_version": SCHEMA_VERSION,
        "encoding": ENCODING_CONTRACT,
        "encoding_sha256": ENCODING_SHA256,
        "sources": entries,
    }
    payload["inventory_sha256"] = _canonical_sha256(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{INVENTORY_PREFIX}-{payload['inventory_sha256']}.json"
    _atomic_json(path, payload)
    return path


def _read_inventory(path: Path, expected_sha256: str) -> dict:
    if path.name != f"{INVENTORY_PREFIX}-{expected_sha256}.json":
        raise ValueError(f"source inventory filename does not bind digest: {path}")
    inventory = json.loads(path.read_text())
    digest = inventory.pop("inventory_sha256", None)
    if digest != expected_sha256 or _canonical_sha256(inventory) != digest:
        raise ValueError(f"source inventory digest mismatch: {path}")
    if (
        inventory.get("artifact_type") != "cuagym_stage_01_source_inventory"
        or inventory.get("schema_version") != SCHEMA_VERSION
        or inventory.get("encoding") != ENCODING_CONTRACT
        or inventory.get("encoding_sha256") != ENCODING_SHA256
        or not isinstance(inventory.get("sources"), list)
        or not inventory["sources"]
    ):
        raise ValueError(f"source inventory contract mismatch: {path}")
    inventory["inventory_sha256"] = digest
    return inventory


def _source_entry(inventory: dict, index: int) -> dict:
    sources = inventory["sources"]
    if not 0 <= index < len(sources):
        raise ValueError(f"tar index {index} outside inventory")
    source = sources[index]
    if (
        not isinstance(source, dict)
        or set(source) != {"name", "path", "size", "sha256"}
        or Path(source["name"]).name != source["name"]
        or not source["name"].endswith(".tar")
    ):
        raise ValueError(f"invalid source inventory entry {index}")
    return source


def _transcode(job: tuple[str, bytes]) -> tuple[str, bytes]:
    from PIL import Image

    member, source = job
    with Image.open(io.BytesIO(source)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"{member} must be PNG, got {image.format!r}")
        rgb = image.convert("RGB")
    if rgb.size != SCREEN:
        raise ValueError(f"{member} must be {SCREEN[0]}x{SCREEN[1]}")
    return member, encode_jpeg_q92(rgb)


def _members(path: Path):
    seen = set()
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            if member.name in seen:
                raise ValueError(f"duplicate PNG member in {path}: {member.name!r}")
            seen.add(member.name)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read tar member {member.name!r}")
            yield member.name, source.read()


def _validate_receipt(directory: Path, receipt: dict) -> None:
    from array_record.python.array_record_module import ArrayRecordReader

    required = {
        "source",
        "source_size",
        "source_sha256",
        "encoding_sha256",
        "directory",
        "num_images",
        "index_sha256",
        "arrayrecord_sha256",
    }
    if set(receipt) != required or receipt.get("directory") != directory.name:
        raise ValueError(f"invalid image shard receipt: {directory}")
    count = receipt.get("num_images")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("invalid image count")
    index_path, shard_path = directory / INDEX_NAME, directory / SHARD_NAME
    if _sha256(index_path) != receipt["index_sha256"]:
        raise ValueError(f"image index digest mismatch: {index_path}")
    if _sha256(shard_path) != receipt["arrayrecord_sha256"]:
        raise ValueError(f"ArrayRecord digest mismatch: {shard_path}")
    reader = ArrayRecordReader(str(shard_path))
    try:
        if reader.num_records() != count:
            raise ValueError(f"ArrayRecord count mismatch: {shard_path}")
        observed = 0
        with index_path.open(encoding="utf-8") as index_file:
            for expected_index, line in enumerate(index_file):
                row = json.loads(line)
                if set(row) != {"member", "uri", "jpeg_sha256"}:
                    raise ValueError("invalid image index row")
                path, index = parse_arrayrecord_image_uri(row["uri"])
                if path.resolve() != shard_path.resolve() or index != expected_index:
                    raise ValueError("image index URI mismatch")
                observed += 1
        if observed != count:
            raise ValueError(f"image count mismatch: {index_path}")
    finally:
        reader.close()


def _validate_new_output(directory: Path, published: Path, receipt: dict) -> None:
    from array_record.python.array_record_module import ArrayRecordReader

    temporary_index = directory / INDEX_NAME
    published_shard = (published / SHARD_NAME).resolve()
    if _sha256(temporary_index) != receipt["index_sha256"]:
        raise ValueError(f"image index digest mismatch: {temporary_index}")
    if _sha256(directory / SHARD_NAME) != receipt["arrayrecord_sha256"]:
        raise ValueError(f"ArrayRecord digest mismatch: {directory / SHARD_NAME}")
    reader = ArrayRecordReader(str(directory / SHARD_NAME))
    try:
        if reader.num_records() != receipt["num_images"]:
            raise ValueError("ArrayRecord count mismatch")
        observed = 0
        with temporary_index.open(encoding="utf-8") as index_file:
            for record_index, line in enumerate(index_file):
                row = json.loads(line)
                if set(row) != {"member", "uri", "jpeg_sha256"}:
                    raise ValueError("invalid image index row")
                path, index = parse_arrayrecord_image_uri(row["uri"])
                if path.resolve() != published_shard or index != record_index:
                    raise ValueError("image index URI mismatch")
                jpeg = reader.read([record_index])[0]
                if hashlib.sha256(jpeg).hexdigest() != row["jpeg_sha256"]:
                    raise ValueError("JPEG digest mismatch")
                with validate_jpeg_q92(jpeg) as image:
                    if image.size != SCREEN:
                        raise ValueError("invalid JPEG dimensions")
                observed += 1
        if observed != receipt["num_images"]:
            raise ValueError("image index count mismatch")
    finally:
        reader.close()


def _directory_name(source: dict) -> str:
    return f"{source['name'].removesuffix('.tar')}-{source['sha256'][:16]}-{ENCODING_SHA256[:16]}"


def _validate_inventory_location(inventory_path: Path, output_dir: Path) -> None:
    if inventory_path.resolve().parent != output_dir.resolve():
        raise ValueError(
            f"source inventory must be directly under output_dir: {inventory_path}"
        )


def _validate_source_receipt(source: dict, receipt: dict) -> None:
    expected = {
        "source": source["name"],
        "source_size": source["size"],
        "source_sha256": source["sha256"],
        "encoding_sha256": ENCODING_SHA256,
        "directory": _directory_name(source),
    }
    if {key: receipt.get(key) for key in expected} != expected:
        raise ValueError(f"stale image shard receipt for {source['name']}")


def build_inventory_shard(
    inventory_path: Path,
    inventory_sha256: str,
    output_dir: Path,
    index: int,
    *,
    workers: int,
) -> dict:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    _validate_inventory_location(inventory_path, output_dir)
    source_entry = _source_entry(
        _read_inventory(inventory_path, inventory_sha256), index
    )
    source = Path(source_entry["path"])
    if (
        source.stat().st_size != source_entry["size"]
        or _sha256(source) != source_entry["sha256"]
    ):
        raise ValueError(f"source tar changed since inventory: {source}")
    directory_name = _directory_name(source_entry)
    final = output_dir / "shards" / directory_name
    if (final / RECEIPT_NAME).is_file():
        try:
            receipt = json.loads((final / RECEIPT_NAME).read_text())
            _validate_receipt(final, receipt)
            try:
                _validate_source_receipt(source_entry, receipt)
            except ValueError:
                pass
            else:
                return receipt
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{directory_name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    from array_record.python.array_record_module import ArrayRecordWriter

    try:
        writer = ArrayRecordWriter(str(temporary / SHARD_NAME), "group_size:1")
        pool = multiprocessing.Pool(workers) if workers > 1 else None
    except BaseException:
        if "writer" in locals():
            writer.close()
        shutil.rmtree(temporary)
        raise
    count = 0
    try:
        encoded = (
            pool.imap(_transcode, _members(source), chunksize=8)
            if pool
            else map(_transcode, _members(source))
        )
        with (temporary / INDEX_NAME).open("w") as index_file:
            for member, jpeg in encoded:
                writer.write(jpeg)
                index_file.write(
                    json.dumps(
                        {
                            "member": member,
                            "uri": make_arrayrecord_image_uri(
                                final / SHARD_NAME, count
                            ),
                            "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
    finally:
        failed = sys.exc_info()[0] is not None
        writer.close()
        if pool:
            pool.close()
            pool.join()
        if failed and temporary.exists():
            shutil.rmtree(temporary)
    if count == 0:
        shutil.rmtree(temporary)
        raise ValueError(f"screenshot tar contains no PNG members: {source}")
    try:
        receipt = {
            "source": source_entry["name"],
            "source_size": source_entry["size"],
            "source_sha256": source_entry["sha256"],
            "encoding_sha256": ENCODING_SHA256,
            "directory": directory_name,
            "num_images": count,
            "index_sha256": _sha256(temporary / INDEX_NAME),
            "arrayrecord_sha256": _sha256(temporary / SHARD_NAME),
        }
        (temporary / RECEIPT_NAME).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        _validate_new_output(temporary, final, receipt)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    backup = final.parent / f".{directory_name}.{os.getpid()}.old"
    if final.exists():
        final.replace(backup)
    try:
        temporary.replace(final)
    except BaseException:
        if backup.exists():
            backup.replace(final)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return receipt


def validate_image_store(output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": SCHEMA_VERSION,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        **ENCODING_CONTRACT,
        "encoding_sha256": ENCODING_SHA256,
    }
    shards = manifest.get("shards")
    if (
        {k: manifest.get(k) for k in required} != required
        or not isinstance(shards, dict)
        or not shards
    ):
        raise ValueError(f"image-store contract mismatch: {manifest_path}")
    inventory_sha256 = manifest.get("inventory_sha256")
    inventory_name = manifest.get("inventory")
    if (
        not isinstance(inventory_sha256, str)
        or inventory_name != f"{INVENTORY_PREFIX}-{inventory_sha256}.json"
    ):
        raise ValueError(f"invalid image-store inventory: {manifest_path}")
    inventory = _read_inventory(output_dir / inventory_name, inventory_sha256)
    sources = {}
    for index in range(len(inventory["sources"])):
        source = _source_entry(inventory, index)
        sources[source["name"].removesuffix(".tar")] = source
    if set(shards) != set(sources):
        raise ValueError(
            f"image-store shard set does not match inventory: {manifest_path}"
        )
    for name, receipt in shards.items():
        if not isinstance(name, str) or not isinstance(receipt, dict):
            raise ValueError(f"invalid image shard entry: {name!r}")
        directory = output_dir / "shards" / str(receipt.get("directory"))
        if (
            receipt.get("source") != f"{name}.tar"
            or json.loads((directory / RECEIPT_NAME).read_text()) != receipt
        ):
            raise ValueError(f"image shard receipt mismatch: {directory}")
        _validate_source_receipt(sources[name], receipt)
        _validate_receipt(directory, receipt)
    if manifest.get("num_tars") != len(shards) or manifest.get("total_images") != sum(
        r["num_images"] for r in shards.values()
    ):
        raise ValueError("image-store count mismatch")
    return manifest


def finalize_store(
    inventory_path: Path, inventory_sha256: str, output_dir: Path
) -> dict:
    _validate_inventory_location(inventory_path, output_dir)
    inventory = _read_inventory(inventory_path, inventory_sha256)
    shards = {}
    for index in range(len(inventory["sources"])):
        source = _source_entry(inventory, index)
        directory = output_dir / "shards" / _directory_name(source)
        try:
            receipt = json.loads((directory / RECEIPT_NAME).read_text())
            _validate_receipt(directory, receipt)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"incomplete image shard for {source['name']}: {exc}"
            ) from exc
        _validate_source_receipt(source, receipt)
        shards[source["name"].removesuffix(".tar")] = receipt
    manifest = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": SCHEMA_VERSION,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        **ENCODING_CONTRACT,
        "encoding_sha256": ENCODING_SHA256,
        "inventory": inventory_path.name,
        "inventory_sha256": inventory_sha256,
        "num_tars": len(shards),
        "total_images": sum(r["num_images"] for r in shards.values()),
        "shards": shards,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def build_store(screenshots_dir: Path, output_dir: Path, *, workers: int) -> dict:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    inventory_path = prepare_inventory(screenshots_dir, output_dir)
    digest = json.loads(inventory_path.read_text())["inventory_sha256"]
    inventory = _read_inventory(inventory_path, digest)
    for index in range(len(inventory["sources"])):
        build_inventory_shard(
            inventory_path, digest, output_dir, index, workers=workers
        )
    return finalize_store(inventory_path, digest, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory_sha256")
    parser.add_argument("--tar_index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--screenshots_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is not None and (args.screenshots_dir or args.finalize):
        raise SystemExit("--workers is only valid with --tar_index")
    if args.screenshots_dir:
        if any(
            (
                args.inventory,
                args.inventory_sha256,
                args.tar_index is not None,
                args.finalize,
            )
        ):
            raise SystemExit("--screenshots_dir only prepares an inventory")
        result = json.loads(
            prepare_inventory(args.screenshots_dir, args.output_dir).read_text()
        )
    else:
        if (
            args.inventory is None
            or args.inventory_sha256 is None
            or args.finalize == (args.tar_index is not None)
        ):
            raise SystemExit(
                "provide inventory digest and exactly one of --tar_index/--finalize"
            )
        result = (
            finalize_store(args.inventory, args.inventory_sha256, args.output_dir)
            if args.finalize
            else build_inventory_shard(
                args.inventory,
                args.inventory_sha256,
                args.output_dir,
                args.tar_index,
                workers=(os.cpu_count() or 1) if args.workers is None else args.workers,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
