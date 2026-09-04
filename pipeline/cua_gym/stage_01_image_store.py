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

from pipeline.lib.image_store import make_arrayrecord_image_uri

JPEG_QUALITY = 92
SCREEN = (1920, 1080)
SHARD_NAME = "images.array_record"
INDEX_NAME = "index.jsonl"
SUMMARY_NAME = "summary.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transcode(job: tuple[str, bytes]) -> tuple[str, bytes]:
    from PIL import Image

    member, source = job
    with Image.open(io.BytesIO(source)) as image:
        image.load()
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
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read tar member {member.name!r}")
            yield member.name, source.read()


def _complete(output: Path, source_sha256: str) -> bool:
    try:
        summary = json.loads((output / SUMMARY_NAME).read_text(encoding="utf-8"))
        rows = (output / INDEX_NAME).read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError):
        return False
    return (
        (output / SHARD_NAME).is_file()
        and summary.get("source_sha256") == source_sha256
        and summary.get("jpeg_quality") == JPEG_QUALITY
        and summary.get("num_images") == len(rows)
        and len(rows) > 0
    )


def process_tar(source: Path, output_root: Path, *, workers: int) -> dict:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    source_sha256 = _sha256(source)
    name = source.name.removesuffix(".tar")
    final = output_root / name
    if _complete(final, source_sha256):
        return json.loads((final / SUMMARY_NAME).read_text(encoding="utf-8"))
    if final.exists():
        shutil.rmtree(final)
    temporary = output_root / f".{name}.{os.getpid()}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    from array_record.python.array_record_module import ArrayRecordWriter

    writer = ArrayRecordWriter(str(temporary / SHARD_NAME), "group_size:1")
    count = 0

    def write(encoded) -> None:
        nonlocal count
        for member, jpeg in encoded:
            writer.write(jpeg)
            uri = make_arrayrecord_image_uri(final / SHARD_NAME, count)
            index.write(json.dumps({"member": member, "uri": uri}) + "\n")
            count += 1

    try:
        with (temporary / INDEX_NAME).open("w", encoding="utf-8") as index:
            jobs = _members(source)
            if workers == 1:
                write(map(_transcode, jobs))
            else:
                with multiprocessing.Pool(workers) as pool:
                    write(pool.imap(_transcode, jobs, chunksize=8))
    except BaseException:
        writer.close()
        shutil.rmtree(temporary)
        raise
    writer.close()
    if count == 0:
        shutil.rmtree(temporary)
        raise ValueError(f"screenshot tar contains no PNG members: {source}")
    summary = {
        "source": source.name,
        "source_sha256": source_sha256,
        "jpeg_quality": JPEG_QUALITY,
        "num_images": count,
    }
    (temporary / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(final)
    return summary


def build_store(screenshots_dir: Path, output_dir: Path, *, workers: int) -> dict:
    sources = sorted(screenshots_dir.glob("screenshots-*.tar"))
    if not sources:
        raise ValueError(f"no screenshots-*.tar files under {screenshots_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [process_tar(source, output_dir, workers=workers) for source in sources]
    manifest = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": 1,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        "jpeg_quality": JPEG_QUALITY,
        "width": SCREEN[0],
        "height": SCREEN[1],
        "image_domain": "jpeg_q92_1920x1080",
        "num_tars": len(summaries),
        "total_images": sum(item["num_images"] for item in summaries),
        "source_tars": {item["source"]: item["source_sha256"] for item in summaries},
    }
    temporary = output_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_dir / "manifest.json")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screenshots-dir", "--screenshots_dir", type=Path, required=True
    )
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True)
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
