"""Measure the shared chat artifact into a tokenizer-bound length cache."""

from __future__ import annotations

import heapq
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path

from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.lib.manifest import (
    file_sha256_short,
    make_artifact_id,
    resolve_chat_artifact,
    write_manifest,
)
from pipeline.lib.omegalax import (
    attest_omegalax,
    attest_processor_snapshot,
    isolated_subprocess_environment,
    omegalax_python,
    validate_message_length_row,
)

FLAGS = flags.FLAGS
MESSAGE_LENGTHS_FILENAME = "message_lengths.jsonl"

flags.DEFINE_string(
    "output_dir", None, "Message-length cache output dir.", required=True
)
flags.DEFINE_string("source_path", None, "Stage 04 conversations root.", required=True)
flags.DEFINE_string("omegalax_repo", None, "Omegalax repository root.", required=True)
flags.DEFINE_string(
    "processor_snapshot", None, "Immutable processor snapshot.", required=True
)
flags.DEFINE_integer(
    "num_workers", None, "Measurement workers (>=2).", required=True, lower_bound=2
)
flags.DEFINE_integer(
    "num_shards", 1, "Number of round-robin measurement shards.", lower_bound=1
)
flags.DEFINE_integer("shard_index", 0, "Shard index for this worker.", lower_bound=0)
flags.DEFINE_bool("merge", False, "Finalize all completed measurement shards.")
flags.DEFINE_string(
    "work_dir", None, "Scratch directory for measurement intermediates."
)


def _tag(index: int, count: int) -> str:
    return f"shard{index:04d}_of_{count:04d}"


def _paths(output_dir: Path, index: int, count: int) -> tuple[Path, Path]:
    tag = _tag(index, count)
    return (
        output_dir / f"message_lengths.{tag}.jsonl",
        output_dir / f"measure_receipt.{tag}.json",
    )


def _iter_chat(chat: Path) -> Iterator[tuple[int, str, int]]:
    with chat.open(encoding="utf-8") as source:
        index = 0
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") if isinstance(row, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"chat messages must be a non-empty list at {chat}:{line_number}"
                )
            yield index, line, len(messages)
            index += 1


def _iter_expected(chat: Path):
    for index, _line, size in _iter_chat(chat):
        for offset in range(size):
            yield index, offset


def _write_slice(chat: Path, target: Path, index: int, count: int) -> int:
    conversations = 0
    with target.open("w", encoding="utf-8") as output:
        for conversation_index, line, _size in _iter_chat(chat):
            if conversation_index % count == index:
                output.write(line if line.endswith("\n") else line + "\n")
                conversations += 1
    return conversations


def _remap(
    source: Path, target: Path, chat: Path, index: int, count: int, merge_size: int
) -> int:
    rows = 0
    expected = _iter_expected(chat)
    with (
        source.open(encoding="utf-8") as measured,
        target.open("w", encoding="utf-8") as output,
    ):
        for line_number, line in enumerate(measured, 1):
            if not line.strip():
                raise ValueError(f"blank cache row at {source}:{line_number}")
            row = json.loads(line)
            local, offset = validate_message_length_row(
                row, source, line_number, merge_size
            )
            key = local, offset
            wanted = next(expected, None)
            if key != wanted:
                raise ValueError(
                    f"{_tag(index, count)} cache keys mismatch: expected {wanted}, got {key}"
                )
            row["conv_idx"] = index + local * count
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows += 1
    missing = next(expected, None)
    if missing is not None:
        raise ValueError(
            f"{_tag(index, count)} cache keys mismatch: expected {missing}, got None"
        )
    return rows


def _identity(
    source_path: Path, chat: Path, processor: dict, omegalax: dict, count: int
) -> dict:
    return {
        "source": str(source_path),
        "source_id": make_artifact_id(source_path),
        "source_sha256": file_sha256_short(chat, n=64),
        "processor_snapshot": processor,
        "omegalax": omegalax,
        "num_workers": FLAGS.num_workers,
        "num_shards": count,
    }


def _iter_keyed(path: Path, merge_size: int, shard_index: int, num_shards: int):
    previous = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"blank cache row at {path}:{line_number}")
            row = json.loads(line)
            key = validate_message_length_row(row, path, line_number, merge_size)
            if key[0] % num_shards != shard_index:
                raise ValueError(f"wrong partition key at {path}:{line_number}: {key}")
            if previous is not None and key <= previous:
                raise ValueError(
                    f"cache keys are not strictly sorted at {path}:{line_number}"
                )
            previous = key
            yield key, shard_index, line.rstrip("\n")


def _validate_receipt(
    receipt_path: Path, cache: Path, identity: dict, index: int
) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or set(receipt) != {
        "identity",
        "shard_index",
        "cache",
    }:
        raise ValueError(f"invalid shard receipt: {receipt_path}")
    if receipt["identity"] != identity or receipt["shard_index"] != index:
        raise ValueError(f"stale or wrong-shard receipt: {receipt_path}")
    info = receipt["cache"]
    fields = {"file", "sha256", "n_messages", "n_conversations", "elapsed_s"}
    if not isinstance(info, dict) or set(info) != fields:
        raise ValueError(f"invalid shard receipt cache: {receipt_path}")
    if any(
        isinstance(info[name], bool)
        or not isinstance(info[name], int)
        or info[name] < 0
        for name in ("n_messages", "n_conversations", "elapsed_s")
    ):
        raise ValueError(f"invalid shard receipt counters: {receipt_path}")
    if not isinstance(info["sha256"], str) or len(info["sha256"]) != 64:
        raise ValueError(f"invalid shard receipt digest: {receipt_path}")
    if (
        info["file"] != cache.name
        or not cache.is_file()
        or info["sha256"] != file_sha256_short(cache, n=64)
    ):
        raise ValueError(f"shard cache does not match receipt: {receipt_path}")
    return info


def _measure(
    chat: Path,
    output_dir: Path,
    identity: dict,
    index: int,
    processor: dict,
    omegalax: dict,
) -> None:
    cache, receipt_path = _paths(output_dir, index, identity["num_shards"])
    if receipt_path.is_file():
        try:
            info = _validate_receipt(receipt_path, cache, identity, index)
            rows = sum(
                1
                for _ in _iter_keyed(
                    cache, processor["merge_size"], index, identity["num_shards"]
                )
            )
            if rows != info["n_messages"]:
                raise ValueError(
                    f"shard cache count does not match receipt: {receipt_path}"
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        else:
            print(
                f"[stage_measure_chat] {_tag(index, identity['num_shards'])} cached",
                flush=True,
            )
            return
    receipt_path.unlink(missing_ok=True)
    scratch = Path(FLAGS.work_dir).resolve() if FLAGS.work_dir else output_dir
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f".{_tag(index, identity['num_shards'])}.", dir=scratch)
    )
    temporary = output_dir / f".{cache.name}.tmp"
    try:
        sliced = work / "chat.jsonl"
        conversations = _write_slice(chat, sliced, index, identity["num_shards"])
        measured = work / "measured"
        measured.mkdir()
        started = time.time()
        if conversations == 0:
            (measured / MESSAGE_LENGTHS_FILENAME).write_text("", encoding="utf-8")
        else:
            cmd = omegalax_python(
                Path(omegalax["path"]),
                "scripts/measure_message_lengths_from_chat.py",
                f"--data_path={sliced}",
                f"--out_dir={measured}",
                f"--model_id={processor['path']}",
                f"--processor={processor['path']}",
                f"--num_workers={FLAGS.num_workers}",
            )
            result = subprocess.run(
                cmd,
                cwd=omegalax["path"],
                env=isolated_subprocess_environment(),
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"measure_message_lengths_from_chat.py failed (rc={result.returncode})"
                )
        elapsed = int(time.time() - started)
        entries = list(measured.iterdir())
        if (
            len(entries) != 1
            or entries[0].name != MESSAGE_LENGTHS_FILENAME
            or not entries[0].is_file()
        ):
            raise RuntimeError(
                f"measurement produced no cache: {measured / MESSAGE_LENGTHS_FILENAME}"
            )
        messages = _remap(
            entries[0],
            temporary,
            sliced,
            index,
            identity["num_shards"],
            processor["merge_size"],
        )
        current = _identity(
            Path(identity["source"]),
            chat,
            attest_processor_snapshot(Path(FLAGS.processor_snapshot)),
            attest_omegalax(Path(FLAGS.omegalax_repo)),
            identity["num_shards"],
        )
        if current != identity:
            raise RuntimeError(
                "Stage05 source or compiler identity changed during execution"
            )
        temporary.replace(cache)
        receipt = {
            "identity": identity,
            "shard_index": index,
            "cache": {
                "file": cache.name,
                "sha256": file_sha256_short(cache, n=64),
                "n_messages": messages,
                "n_conversations": conversations,
                "elapsed_s": elapsed,
            },
        }
        receipt_tmp = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        receipt_tmp.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        receipt_tmp.replace(receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


def _close_rows(path: Path, merge_size: int, shard_index: int, num_shards: int):
    return closing(_iter_keyed(path, merge_size, shard_index, num_shards))


def _merge(
    source_path: Path,
    chat: Path,
    output_dir: Path,
    identity: dict,
    processor: dict,
    omegalax: dict,
) -> None:
    caches = []
    infos = []
    for index in range(identity["num_shards"]):
        cache, receipt = _paths(output_dir, index, identity["num_shards"])
        if not receipt.is_file():
            raise ValueError(f"missing shard receipt: {receipt}")
        infos.append(_validate_receipt(receipt, cache, identity, index))
        caches.append(cache)
    final = output_dir / MESSAGE_LENGTHS_FILENAME
    temporary = output_dir / f".{MESSAGE_LENGTHS_FILENAME}.tmp"
    rows = 0
    shard_rows = [0] * identity["num_shards"]
    shard_conversations = [0] * identity["num_shards"]
    previous_conversation = [None] * identity["num_shards"]
    try:
        with ExitStack() as stack, temporary.open("w", encoding="utf-8") as target:
            streams = [
                stack.enter_context(
                    _close_rows(
                        path, processor["merge_size"], index, identity["num_shards"]
                    )
                )
                for index, path in enumerate(caches)
            ]
            expected = _iter_expected(chat)
            for key, shard_index, line in heapq.merge(
                *streams, key=lambda item: item[0]
            ):
                wanted = next(expected, None)
                if key != wanted:
                    raise ValueError(
                        f"merged cache key mismatch: expected {wanted}, got {key}"
                    )
                target.write(line + "\n")
                rows += 1
                shard_rows[shard_index] += 1
                if key[0] != previous_conversation[shard_index]:
                    shard_conversations[shard_index] += 1
                    previous_conversation[shard_index] = key[0]
            missing = next(expected, None)
            if missing is not None:
                raise ValueError(
                    f"merged cache is incomplete: first missing key {missing}"
                )
            for index, (observed, info) in enumerate(zip(shard_rows, infos)):
                if observed != info["n_messages"]:
                    raise ValueError(
                        f"shard {_tag(index, identity['num_shards'])} cache count does not "
                        f"match receipt: expected {info['n_messages']}, got {observed}"
                    )
                if shard_conversations[index] != info["n_conversations"]:
                    raise ValueError(
                        f"shard {_tag(index, identity['num_shards'])} conversation count "
                        f"does not match receipt: expected {info['n_conversations']}, "
                        f"got {shard_conversations[index]}"
                    )
        current = _identity(
            source_path,
            chat,
            attest_processor_snapshot(Path(FLAGS.processor_snapshot)),
            attest_omegalax(Path(FLAGS.omegalax_repo)),
            identity["num_shards"],
        )
        if current != identity:
            raise RuntimeError(
                "Stage05 source or compiler identity changed during execution"
            )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    cache = {
        "file": MESSAGE_LENGTHS_FILENAME,
        "sha256": file_sha256_short(temporary, n=64),
        "n_messages": rows,
        "elapsed_s": sum(info["elapsed_s"] for info in infos),
    }
    manifest = output_dir / "manifest.json"
    final_backup = output_dir / f".{MESSAGE_LENGTHS_FILENAME}.previous"
    manifest_backup = output_dir / ".manifest.json.previous"
    if final_backup.exists() or manifest_backup.exists():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"interrupted Stage05 publication under {output_dir}")
    final_saved = False
    manifest_saved = False
    try:
        if final.exists():
            final.replace(final_backup)
            final_saved = True
        if manifest.exists():
            manifest.replace(manifest_backup)
            manifest_saved = True
        temporary.replace(final)
        write_manifest(
            output_dir,
            stage="message_lengths",
            params={
                "processor_snapshot": processor,
                "num_workers": FLAGS.num_workers,
                "omegalax_repo": omegalax["path"],
                "omegalax": omegalax,
            },
            inputs={"source": str(source_path), "source_id": identity["source_id"]},
            stats={"cache": cache},
        )
    except BaseException:
        final.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        (output_dir / "manifest.json.tmp").unlink(missing_ok=True)
        if final_saved:
            final_backup.replace(final)
        if manifest_saved:
            manifest_backup.replace(manifest)
        raise
    final_backup.unlink(missing_ok=True)
    manifest_backup.unlink(missing_ok=True)


def main(_) -> None:
    if not 0 <= FLAGS.shard_index < FLAGS.num_shards:
        raise ValueError(f"shard_index must be in [0, {FLAGS.num_shards})")
    if FLAGS.merge:
        unsupported = [
            name for name in ("shard_index", "work_dir") if FLAGS[name].present
        ]
        if unsupported:
            raise ValueError(
                "merge does not accept worker flags: "
                + ", ".join(f"--{name}" for name in unsupported)
            )
    output_dir = Path(FLAGS.output_dir).resolve()
    source_path = Path(FLAGS.source_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chat = resolve_chat_artifact(source_path)
    processor = attest_processor_snapshot(Path(FLAGS.processor_snapshot))
    omegalax = attest_omegalax(Path(FLAGS.omegalax_repo))
    identity = _identity(source_path, chat, processor, omegalax, FLAGS.num_shards)
    if FLAGS.merge:
        _merge(source_path, chat, output_dir, identity, processor, omegalax)
        return
    _measure(chat, output_dir, identity, FLAGS.shard_index, processor, omegalax)
    if FLAGS.num_shards == 1:
        _merge(source_path, chat, output_dir, identity, processor, omegalax)


if __name__ == "__main__":
    app.run(main)
