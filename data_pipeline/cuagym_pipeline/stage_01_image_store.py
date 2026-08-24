"""Stage 01 (cuagym): screenshot tars -> JPEG ArrayRecord image store.

Each input tar shard (screenshots-NNNN.tar, members "<task_id>/step_NNN.png")
becomes one output subdir under --output_root:

    <output_root>/screenshots-NNNN/
        images.array_record   raw JPEG bytes per record (quality 92, RGB)
        index.jsonl           {"member": "<task_id>/step_NNN.png",
                               "uri": "ar:///abs/.../images.array_record#idx"}
        summary.json          per-tar counts + failures

URIs follow realigned_pipeline/lib/image_store.py, so the existing ar://
resolver reads this store unchanged. Tars are streamed (never extracted to
disk); each PNG is decode-verified during transcode; failures are counted and
listed in summary.json, never written to the shard. Output is atomic: a tar is
built in a tmp dir and renamed into place, so a subdir that exists with its
summary.json is complete and gets skipped on resume.

Run one tar with --tar_path, or --tar_dir + --tar_index for SLURM arrays.
--finalize_expected=N writes the top-level manifest.json once N complete tar
subdirs exist (idempotent; safe under array-task races).
"""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

from absl import app, flags, logging

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from realigned_pipeline.lib.image_store import make_arrayrecord_image_uri  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string("tar_path", None, "Path to one screenshots-NNNN.tar to process.")
flags.DEFINE_string("tar_dir", None, "Directory holding screenshots-*.tar (used with --tar_index).")
flags.DEFINE_integer("tar_index", -1, "Index into the sorted screenshots-*.tar listing of --tar_dir.")
flags.DEFINE_string("output_root", None, "Output root; one subdir per input tar is created under it.")
flags.DEFINE_integer("jpeg_quality", 92, "JPEG quality for the transcoded records.")
flags.DEFINE_integer("num_workers", 0, "Transcode worker processes (0 = cpu count).")
flags.DEFINE_integer(
    "finalize_expected",
    0,
    "If > 0: after this tar finishes, write <output_root>/manifest.json once this many complete tar subdirs exist.",
)
flags.DEFINE_boolean("finalize_only", False, "Skip tar processing; only attempt the manifest.json finalize.")

flags.mark_flag_as_required("output_root")

SHARD_NAME = "images.array_record"
INDEX_NAME = "index.jsonl"
SUMMARY_NAME = "summary.json"


def _transcode(job: tuple[str, bytes, int]) -> tuple[str, bytes | None, str | None]:
    member, png_bytes, quality = job
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(io.BytesIO(png_bytes)) as im:
            im.load()
            rgb = im.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
        return member, buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001
        return member, None, f"{type(exc).__name__}: {exc}"


def _iter_png_members(tar_path: Path, quality: int):
    with tarfile.open(tar_path, mode="r|") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            yield member.name, fobj.read(), quality


def tar_output_complete(out_dir: Path) -> bool:
    summary_path = out_dir / SUMMARY_NAME
    index_path = out_dir / INDEX_NAME
    shard_path = out_dir / SHARD_NAME
    if not (summary_path.exists() and index_path.exists() and shard_path.exists()):
        return False
    try:
        summary = json.loads(summary_path.read_text())
        with index_path.open() as f:
            n_lines = sum(1 for _ in f)
        return int(summary.get("num_images", -1)) == n_lines
    except Exception:  # noqa: BLE001
        return False


def process_tar(tar_path: Path, output_root: Path, quality: int, num_workers: int) -> None:
    tar_name = tar_path.name.removesuffix(".tar")
    final_dir = output_root / tar_name
    final_shard = final_dir / SHARD_NAME

    if tar_output_complete(final_dir):
        logging.info("skip %s: complete output at %s", tar_path.name, final_dir)
        return
    if final_dir.exists():
        logging.warning("removing incomplete output %s", final_dir)
        shutil.rmtree(final_dir)

    for stale in output_root.glob(f".tmp_{tar_name}_*"):
        logging.warning("removing stale tmp dir %s", stale)
        shutil.rmtree(stale)

    tmp_dir = output_root / f".tmp_{tar_name}_{os.getpid()}"
    tmp_dir.mkdir(parents=True)

    from array_record.python.array_record_module import ArrayRecordWriter  # noqa: PLC0415

    t0 = time.monotonic()
    num_images = 0
    failures: list[dict[str, str]] = []
    writer = ArrayRecordWriter(str(tmp_dir / SHARD_NAME), "group_size:1")
    workers = num_workers or mp.cpu_count()
    try:
        with (tmp_dir / INDEX_NAME).open("w") as index_f, mp.Pool(workers) as pool:
            for member, jpeg, error in pool.imap(
                _transcode, _iter_png_members(tar_path, quality), chunksize=8
            ):
                if error is not None:
                    failures.append({"member": member, "error": error})
                    logging.warning("FAIL %s: %s", member, error)
                    continue
                writer.write(jpeg)
                uri = make_arrayrecord_image_uri(final_shard, num_images)
                index_f.write(json.dumps({"member": member, "uri": uri}) + "\n")
                num_images += 1
    finally:
        writer.close()

    elapsed_s = round(time.monotonic() - t0, 1)
    summary = {
        "tar": tar_path.name,
        "shard": str(final_shard),
        "num_images": num_images,
        "num_failures": len(failures),
        "failures": failures,
        "jpeg_quality": quality,
        "jpeg_bytes_total": (tmp_dir / SHARD_NAME).stat().st_size,
        "elapsed_s": elapsed_s,
    }
    (tmp_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp_dir, final_dir)
    logging.info(
        "done %s: %d images, %d failures, %.1fs -> %s",
        tar_path.name,
        num_images,
        len(failures),
        elapsed_s,
        final_dir,
    )


def maybe_finalize(output_root: Path, expected: int) -> None:
    complete = sorted(d for d in output_root.glob("screenshots-*") if tar_output_complete(d))
    if len(complete) < expected:
        logging.info("finalize: %d/%d tars complete, not writing manifest.json", len(complete), expected)
        return
    per_tar = {}
    total_images = 0
    total_failures = 0
    for d in complete:
        s = json.loads((d / SUMMARY_NAME).read_text())
        per_tar[d.name] = {"num_images": s["num_images"], "num_failures": s["num_failures"]}
        total_images += s["num_images"]
        total_failures += s["num_failures"]
    manifest = {
        "artifact_type": "cuagym_stage_01_image_store",
        "schema_version": 1,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        "jpeg_quality": json.loads((complete[0] / SUMMARY_NAME).read_text())["jpeg_quality"],
        "num_tars": len(complete),
        "total_images": total_images,
        "total_failures": total_failures,
        "tars": per_tar,
    }
    tmp = output_root / f".tmp_manifest_{os.getpid()}.json"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, output_root / "manifest.json")
    logging.info(
        "finalize: manifest.json written (%d tars, %d images, %d failures)",
        len(complete),
        total_images,
        total_failures,
    )


def resolve_tar_path() -> Path:
    if FLAGS.tar_path:
        p = Path(FLAGS.tar_path)
        if not p.exists():
            raise SystemExit(f"--tar_path does not exist: {p}")
        return p
    if not FLAGS.tar_dir or FLAGS.tar_index < 0:
        raise SystemExit("provide --tar_path, or --tar_dir with --tar_index >= 0")
    tars = sorted(Path(FLAGS.tar_dir).glob("screenshots-*.tar"))
    if not tars:
        raise SystemExit(f"no screenshots-*.tar under {FLAGS.tar_dir}")
    if FLAGS.tar_index >= len(tars):
        raise SystemExit(f"--tar_index {FLAGS.tar_index} out of range ({len(tars)} tars)")
    return tars[FLAGS.tar_index]


def main(argv: list[str]) -> None:
    del argv
    output_root = Path(FLAGS.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not FLAGS.finalize_only:
        process_tar(resolve_tar_path(), output_root, FLAGS.jpeg_quality, FLAGS.num_workers)
    if FLAGS.finalize_expected > 0 or FLAGS.finalize_only:
        maybe_finalize(output_root, max(FLAGS.finalize_expected, 1))


if __name__ == "__main__":
    app.run(main)
