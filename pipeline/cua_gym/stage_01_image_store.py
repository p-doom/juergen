"""Build the fixed JPEG-q92 image store for CUA-Gym trajectories."""

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

from cua_parity_contract import (
    JPEG_QUALITY,
    OBSERVATION_CONTRACT,
    OBSERVATION_SIZE,
)
from pipeline.lib.image_store import make_arrayrecord_image_uri

SCREEN = OBSERVATION_SIZE
SHARD_NAME = "images.array_record"
INDEX_NAME = "index.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return _bytes_sha256(encoded)


def _transcode(job: tuple[str, bytes]) -> tuple[str, bytes]:
    from PIL import Image

    member, source = job
    with Image.open(io.BytesIO(source)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"{member} must be PNG, got {image.format!r}")
        rgb = image.convert("RGB")
    if rgb.size != SCREEN:
        raise ValueError(
            f"{member} must be {SCREEN[0]}x{SCREEN[1]}, got {rgb.size[0]}x{rgb.size[1]}"
        )
    encoded = io.BytesIO()
    rgb.save(
        encoded,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=2,
        optimize=False,
    )
    return member, encoded.getvalue()


def _members(path: Path):
    seen: set[str] = set()
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


def _validate_shard(
    physical: Path,
    published: Path,
    expected: dict[str, object],
) -> None:
    from array_record.python.array_record_module import ArrayRecordReader
    from PIL import Image

    index_path = physical / INDEX_NAME
    shard_path = physical / SHARD_NAME
    try:
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read image index {index_path}: {exc}") from exc
    if len(rows) != expected["num_images"] or not rows:
        raise ValueError(f"image count mismatch in {index_path}")
    if _sha256(index_path) != expected["index_sha256"]:
        raise ValueError(f"image index digest mismatch: {index_path}")
    if _sha256(shard_path) != expected["arrayrecord_sha256"]:
        raise ValueError(f"ArrayRecord digest mismatch: {shard_path}")

    reader = ArrayRecordReader(str(shard_path))
    try:
        if reader.num_records() != len(rows):
            raise ValueError(f"ArrayRecord count mismatch: {shard_path}")
        for record_index, row in enumerate(rows):
            if set(row) != {"member", "uri", "jpeg_sha256"}:
                raise ValueError(f"invalid image index row {record_index}: {row!r}")
            expected_uri = make_arrayrecord_image_uri(
                published / SHARD_NAME, record_index
            )
            if row["uri"] != expected_uri:
                raise ValueError(
                    f"image index URI mismatch at row {record_index}: {row['uri']!r}"
                )
            jpeg = reader.read([record_index])[0]
            if _bytes_sha256(jpeg) != row["jpeg_sha256"]:
                raise ValueError(
                    f"JPEG digest mismatch at {shard_path} record {record_index}"
                )
            with Image.open(io.BytesIO(jpeg)) as image:
                image.load()
                if (
                    image.format != "JPEG"
                    or image.mode != "RGB"
                    or image.size != SCREEN
                ):
                    raise ValueError(
                        f"invalid JPEG at {shard_path} record {record_index}"
                    )
    finally:
        reader.close()


def _validate_generation(
    physical: Path,
    published: Path,
    shards: dict[str, dict[str, object]],
) -> None:
    observed = {path.name for path in physical.iterdir() if path.is_dir()}
    if observed != set(shards):
        raise ValueError(
            f"image-store shard set mismatch: expected {set(shards)}, got {observed}"
        )
    for name, expected in shards.items():
        _validate_shard(physical / name, published / name, expected)


def validate_image_store(
    output_dir: Path, *, manifest_path: Path | None = None
) -> dict[str, object]:
    manifest_path = manifest_path or output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read image-store manifest {manifest_path}: {exc}"
        ) from exc
    required = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": 1,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        "jpeg_quality": JPEG_QUALITY,
        "width": SCREEN[0],
        "height": SCREEN[1],
        "image_domain": OBSERVATION_CONTRACT,
    }
    if {key: manifest.get(key) for key in required} != required:
        raise ValueError(f"image-store contract mismatch: {manifest!r}")
    generation = manifest.get("generation")
    shards = manifest.get("shards")
    source_tars = manifest.get("source_tars")
    if (
        not isinstance(generation, str)
        or not generation.startswith("generation-")
        or Path(generation).name != generation
        or not isinstance(shards, dict)
        or not shards
        or not isinstance(source_tars, dict)
    ):
        raise ValueError(f"invalid image-store manifest: {manifest_path}")
    if manifest.get("num_tars") != len(shards):
        raise ValueError(f"image-store tar count mismatch: {manifest_path}")
    if manifest.get("total_images") != sum(
        int(item["num_images"]) for item in shards.values()
    ):
        raise ValueError(f"image-store image count mismatch: {manifest_path}")
    for name, expected in shards.items():
        if not isinstance(name, str) or not isinstance(expected, dict):
            raise TypeError(f"invalid shard entry in {manifest_path}")
        if set(expected) != {
            "source",
            "source_sha256",
            "num_images",
            "index_sha256",
            "arrayrecord_sha256",
        }:
            raise ValueError(f"invalid shard contract for {name!r}")
        if expected["source"] != f"{name}.tar":
            raise ValueError(f"shard/source mismatch for {name!r}")
        if source_tars.get(expected["source"]) != expected["source_sha256"]:
            raise ValueError(f"source digest mismatch for {name!r}")
    if set(source_tars) != {f"{name}.tar" for name in shards}:
        raise ValueError(f"image-store source/shard set mismatch: {manifest_path}")
    path = output_dir / generation
    generations = {item.name for item in output_dir.glob("generation-*")}
    if generations != {generation}:
        raise ValueError(
            f"image-store generation set mismatch: expected {generation!r}, got {generations}"
        )
    _validate_generation(path, path, shards)
    return manifest


def _existing_manifest(
    output_dir: Path,
    source_tars: dict[str, str],
    previous_manifest: Path,
) -> dict[str, object] | None:
    try:
        manifest = validate_image_store(output_dir, manifest_path=previous_manifest)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return manifest if manifest["source_tars"] == source_tars else None


def _build_shard(
    source: Path,
    physical: Path,
    published: Path,
    *,
    workers: int,
) -> dict[str, object]:
    from array_record.python.array_record_module import ArrayRecordWriter

    physical.mkdir(parents=True)
    writer = ArrayRecordWriter(str(physical / SHARD_NAME), "group_size:1")
    count = 0

    def write(encoded) -> None:
        nonlocal count
        for member, jpeg in encoded:
            writer.write(jpeg)
            uri = make_arrayrecord_image_uri(published / SHARD_NAME, count)
            index.write(
                json.dumps(
                    {
                        "member": member,
                        "uri": uri,
                        "jpeg_sha256": _bytes_sha256(jpeg),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1

    try:
        with (physical / INDEX_NAME).open("w", encoding="utf-8") as index:
            jobs = _members(source)
            if workers == 1:
                write(map(_transcode, jobs))
            else:
                with multiprocessing.Pool(workers) as pool:
                    write(pool.imap(_transcode, jobs, chunksize=8))
    finally:
        writer.close()
    if count == 0:
        raise ValueError(f"screenshot tar contains no PNG members: {source}")
    return {
        "source": source.name,
        "source_sha256": _sha256(source),
        "num_images": count,
        "index_sha256": _sha256(physical / INDEX_NAME),
        "arrayrecord_sha256": _sha256(physical / SHARD_NAME),
    }


def _publish_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    temporary = output_dir / f".manifest.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_dir / "manifest.json")


def build_store(screenshots_dir: Path, output_dir: Path, *, workers: int) -> dict:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous_manifest = output_dir / ".manifest.previous.json"
    if manifest_path.exists():
        manifest_path.replace(previous_manifest)
    sources = sorted(screenshots_dir.glob("screenshots-*.tar"))
    if not sources:
        raise ValueError(f"no screenshots-*.tar files under {screenshots_dir}")
    source_tars = {source.name: _sha256(source) for source in sources}
    if manifest := _existing_manifest(output_dir, source_tars, previous_manifest):
        _publish_manifest(output_dir, manifest)
        previous_manifest.unlink(missing_ok=True)
        return manifest

    generation_name = f"generation-{_canonical_sha256(source_tars)[:16]}"
    generation = output_dir / generation_name
    temporary = output_dir / f".{generation_name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    shards: dict[str, dict[str, object]] = {}
    try:
        for source in sources:
            name = source.name.removesuffix(".tar")
            shards[name] = _build_shard(
                source,
                temporary / name,
                generation / name,
                workers=workers,
            )
        _validate_generation(temporary, generation, shards)
        if generation.exists():
            backup = output_dir / f".{generation_name}.{os.getpid()}.old"
            generation.replace(backup)
            temporary.replace(generation)
            shutil.rmtree(backup)
        else:
            temporary.replace(generation)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    manifest: dict[str, object] = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": 1,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        "jpeg_quality": JPEG_QUALITY,
        "width": SCREEN[0],
        "height": SCREEN[1],
        "image_domain": OBSERVATION_CONTRACT,
        "generation": generation_name,
        "num_tars": len(shards),
        "total_images": sum(int(item["num_images"]) for item in shards.values()),
        "source_tars": source_tars,
        "shards": shards,
    }
    for path in output_dir.glob("generation-*"):
        if path != generation:
            shutil.rmtree(path)
    _publish_manifest(output_dir, manifest)
    previous_manifest.unlink(missing_ok=True)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshots_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_store(args.screenshots_dir, args.output_dir, workers=args.workers),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
