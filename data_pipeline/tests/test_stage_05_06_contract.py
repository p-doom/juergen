"""Stages 05/06: the wrapper contract, and the patch-count gap it cannot see.

Both stages are thin wrappers: they validate a join, shell out to an omegalax
script under ``uv run --project <omegalax_repo>``, and write a manifest. The
omegalax side needs jax/transformers/torch and a tokenizer download and is not
reachable from the data-pipeline venv, so the ``uv`` invocation is replaced by a
recording stub on PATH. Everything on this side of that boundary — the
chat.jsonl checks, the cache-identity refusal, the split fan-out, the argv, the
manifest — runs for real, in a subprocess each, because both stages define the
same absl flag names on one global ``FLAGS`` and cannot be imported together.

The gap: a stage-06 build once recorded a max vision-patch count 792 BELOW what
the collator produced for the same records, and a training run sized from the
recorded number overflowed. The two numbers come from two independent
expressions over two independently constructed image processors —
``vision_patches += t*h*w`` per message in omegalax's ``_MessageLengthFn``
(cached in ``message_lengths.jsonl``, rolled up into ``token_stats.json`` as
``per_chunk.vision_patches.max``) versus ``pixel_values.shape[0]`` in
``VLMSFTCollator._pad_vision_arrays`` at train time. Given ONE image geometry
they are the same number; they diverge exactly when the geometry differs, and
the geometry is set by the image processor's ``smart_resize`` budget.

What this file can assert from the juergen side is the shape of the hole: stage
06 forwards ``--model_id``/``--processor`` and no geometry knob at all, its only
cross-stage identity check is the chat artifact, and its manifest records no
patch statistic — so a cache measured under one geometry and consumed under
another passes every check this stage makes. The arithmetic test then shows the
two expressions agreeing under one geometry and the shortfall they produce under
two.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import grammars
import pytest
import synthetic_clip as clip

from pipeline.lib.manifest import make_artifact_id
from pipeline.stage_04_build_conversations import build_messages

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PIPELINE_DIR = REPO_ROOT / "data_pipeline"
STAGES = REPO_ROOT / "pipeline"

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CROWD_CAST_IMAGE_DOMAIN = "jpeg_q80_height_720"

# Qwen2-VL image-processor defaults, the geometry the builder gets when nothing
# overrides it (transformers ``Qwen2VLImageProcessor``).
PATCH_SIZE = 14
MERGE_SIZE = 2
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280


def _smart_resize(height: int, width: int, *, factor: int, min_pixels: int,
                  max_pixels: int) -> tuple[int, int]:
    """``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.

    Reproduced rather than imported: ``transformers`` is not a dependency of the
    data-pipeline venv. Both numbers this file compares are functions of its
    output, which is the whole point — it is the only thing that moves.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _grid_thw(height: int, width: int, *, min_pixels: int, max_pixels: int) -> tuple[int, int, int]:
    """``image_grid_thw`` for one still image under a given processor budget."""
    h_bar, w_bar = _smart_resize(
        height, width, factor=PATCH_SIZE * MERGE_SIZE,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    return 1, h_bar // PATCH_SIZE, w_bar // PATCH_SIZE


def vision_patches(grid: tuple[int, int, int]) -> int:
    """``t * h * w`` over an ``image_grid_thw``.

    Both sides of the gap reduce to this one product, which is why asserting the
    two against each other would prove nothing: omegalax's ``_MessageLengthFn``
    accumulates ``vision_patches += t * h * w`` at measure time, and
    ``VLMSFTCollator._pad_vision_arrays`` reads ``pixel_values.shape[0]``, which
    HF's preprocessor guarantees is ``grid_t * grid_h * grid_w``. The grid is
    the only thing that can differ between them.
    """
    t, h, w = grid
    return t * h * w


def vision_tokens(grid: tuple[int, int, int]) -> int:
    """omegalax ``_MessageLengthFn``: ``t * (h // merge) * (w // merge)``."""
    t, h, w = grid
    return t * (h // MERGE_SIZE) * (w // MERGE_SIZE)


# --------------------------------------------------------------------------
# A stage-04-shaped source artifact and a recording stand-in for ``uv``.
# --------------------------------------------------------------------------


def _chat_row(index: int, n_turns: int) -> dict[str, Any]:
    shard = f"/nonexistent/{clip.SEGMENT_ID}/images.array_record"
    turns = [(f"ar://{shard}#{t * clip.STRIDE}", "NO_OP") for t in range(n_turns)]
    return {
        "conversation_id": f"{clip.SEGMENT_ID}:{index}",
        "recording_id": clip.RECORDING_ID,
        "segment_id": clip.SEGMENT_ID,
        "n_frames": n_turns,
        "n_turns": n_turns,
        "messages": build_messages(
            turns,
            instruction="do the synthetic thing",
            system_prompt=grammars.describe("deltatype_v2"),
        ),
    }


def make_source(root: Path, *, n_conversations: int = 2, n_turns: int = 3) -> Path:
    """A stage-04 conversations artifact: chat.jsonl + the manifest that gives
    it an id. Stages 05/06 read exactly these two files."""
    root.mkdir(parents=True, exist_ok=True)
    rows = [_chat_row(i, n_turns) for i in range(n_conversations)]
    with (root / "chat.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (root / "conversations.jsonl").write_bytes((root / "chat.jsonl").read_bytes())
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "juergen_annotation_conversations",
                "schema_version": 2,
                "chat": "chat.jsonl",
                "n_conversations": len(rows),
                "image_domain": CROWD_CAST_IMAGE_DOMAIN,
            },
            indent=2,
        )
        + "\n"
    )
    return root


_FAKE_UV = '''#!{python}
"""Stands in for ``uv run --project <omegalax> python scripts/<name>.py ...``.

Records the argv it was handed and writes only the artefacts the juergen
wrapper counts afterwards; it computes nothing.
"""
import json, os, sys
from pathlib import Path

Path(os.environ["FAKE_UV_LOG"]).open("a").write(json.dumps(sys.argv[1:]) + "\\n")
flags = dict(
    a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a
)
script = next((a for a in sys.argv[1:] if a.endswith(".py")), "")
out = Path(flags["out_dir"])
out.mkdir(parents=True, exist_ok=True)
if "measure_message_lengths" in script:
    rows = int(os.environ.get("FAKE_UV_N_MESSAGES", "6"))
    with (out / "message_lengths.jsonl").open("w") as f:
        for i in range(rows):
            f.write(json.dumps({{"conv_idx": 0, "msg_offset": i}}) + "\\n")
else:
    (out / "part-00000.array_record").write_bytes(b"")
'''


@pytest.fixture
def omegalax(tmp_path: Path) -> dict[str, Any]:
    """A fake omegalax checkout plus the recording ``uv`` that fronts it."""
    repo = tmp_path / "omegalax"
    (repo / "scripts").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(_FAKE_UV.format(python=sys.executable))
    uv.chmod(0o755)
    log = tmp_path / "uv_argv.jsonl"
    roots = [REPO_ROOT, DATA_PIPELINE_DIR, REPO_ROOT.parent / "desktop"]
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        PYTHONPATH=os.pathsep.join(str(r) for r in roots),
        FAKE_UV_LOG=str(log),
    )
    return {"repo": repo, "env": env, "log": log}


def _run(stage: str, env: dict[str, str], **flags: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STAGES / stage), *[f"--{k}={v}" for k, v in flags.items()]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _invocations(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _flags_of(argv: list[str]) -> dict[str, str]:
    return dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)


def _builds(log: Path) -> list[list[str]]:
    """The record-builder invocations, in order."""
    return [a for a in _invocations(log) if any("build_sft_records" in x for x in a)]


def _measure(tmp_path: Path, omegalax: dict[str, Any], source: Path, *,
             out: str = "lengths", processor: str = MODEL_ID) -> Path:
    out_dir = tmp_path / out
    proc = _run(
        "stage_05_measure_lengths.py", omegalax["env"],
        output_dir=out_dir, source_path=source, omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID, processor=processor, num_workers=2,
    )
    assert proc.returncode == 0, proc.stderr
    return out_dir


# --------------------------------------------------------------------------
# Stage 05 — measure
# --------------------------------------------------------------------------


def test_stage_05_refuses_a_source_without_a_chat_file(tmp_path: Path, omegalax) -> None:
    source = make_source(tmp_path / "conv")
    (source / "chat.jsonl").unlink()
    proc = _run(
        "stage_05_measure_lengths.py", omegalax["env"],
        output_dir=tmp_path / "lengths", source_path=source,
        omegalax_repo=omegalax["repo"], model_id=MODEL_ID, processor=MODEL_ID,
        num_workers=2,
    )
    assert proc.returncode != 0
    assert "no chat.jsonl" in proc.stderr
    assert _invocations(omegalax["log"]) == []


def test_stage_05_invokes_the_measure_script_and_fingerprints_its_source(
    tmp_path: Path, omegalax
) -> None:
    source = make_source(tmp_path / "conv")
    out_dir = _measure(tmp_path, omegalax, source)

    (argv,) = _invocations(omegalax["log"])
    assert argv[:5] == [
        "run", "--project", str(omegalax["repo"]), "python",
        "scripts/measure_message_lengths_from_chat.py",
    ]
    assert _flags_of(argv) == {
        "data_path": str(source / "chat.jsonl"),
        "out_dir": str(out_dir),
        "model_id": MODEL_ID,
        "processor": MODEL_ID,
        "num_workers": "2",
    }

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["stage"] == "message_lengths"
    assert manifest["inputs"]["source_id"] == make_artifact_id(source)
    assert manifest["params"]["processor"] == MODEL_ID
    assert manifest["stats"]["per_split"][0]["n_messages"] == 6
    assert (out_dir / "message_lengths.jsonl").is_file()


# --------------------------------------------------------------------------
# Stage 06 — records
# --------------------------------------------------------------------------


def test_stage_06_refuses_a_cache_measured_from_another_dataset(
    tmp_path: Path, omegalax
) -> None:
    source = make_source(tmp_path / "conv")
    other = make_source(tmp_path / "other", n_conversations=3)
    stale = _measure(tmp_path, omegalax, other)
    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=tmp_path / "records", source_path=source,
        omegalax_repo=omegalax["repo"], model_id=MODEL_ID, processor=MODEL_ID,
        max_length=4096, records_per_shard=8, num_workers=2,
        message_lengths_path=stale,
    )
    assert proc.returncode != 0
    assert "message-length cache source mismatch" in proc.stderr
    # One invocation, the measure of ``other``: the refusal happens before any
    # build is launched.
    assert len(_invocations(omegalax["log"])) == 1


def test_stage_06_preserves_crowd_cast_image_domain_in_the_records_manifest(
    tmp_path: Path, omegalax
) -> None:
    source = make_source(tmp_path / "conv")
    out_dir = tmp_path / "records"
    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=out_dir, source_path=source,
        omegalax_repo=omegalax["repo"], model_id=MODEL_ID, processor=MODEL_ID,
        max_length=4096, records_per_shard=8, num_workers=2,
    )

    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["inputs"]["image_domain"] == CROWD_CAST_IMAGE_DOMAIN
    assert [_flags_of(argv)["split"] for argv in _builds(omegalax["log"])] == ["train"]


def test_stage_06_refuses_a_source_with_no_image_domain_before_the_record_builder(
    tmp_path: Path, omegalax
) -> None:
    source = make_source(tmp_path / "conv")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["image_domain"]
    manifest_path.write_text(json.dumps(manifest))

    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=tmp_path / "records", source_path=source,
        omegalax_repo=omegalax["repo"], model_id=MODEL_ID, processor=MODEL_ID,
        max_length=4096, records_per_shard=8, num_workers=2,
    )

    assert proc.returncode != 0
    assert "source conversations must declare image_domain" in proc.stderr
    assert _builds(omegalax["log"]) == []


def test_stage_06_fans_out_one_build_per_split_and_reuses_the_cache(
    tmp_path: Path, omegalax
) -> None:
    source = make_source(tmp_path / "conv")
    lengths = _measure(tmp_path, omegalax, source)
    out_dir = tmp_path / "records"
    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=out_dir, source_path=source, omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID, processor=MODEL_ID, max_length=4096,
        records_per_shard=8, num_workers=2, overflow_mode="split",
        message_lengths_path=lengths, val_fraction=0.25,
    )
    assert proc.returncode == 0, proc.stderr

    builds = _builds(omegalax["log"])
    assert [_flags_of(a)["split"] for a in builds] == ["train", "val"]
    for argv, split in zip(builds, ("train", "val"), strict=True):
        flags = _flags_of(argv)
        assert flags["data_path"] == str(source / "chat.jsonl")
        assert flags["out_dir"] == str(out_dir / split)
        assert flags["message_lengths_path"] == str(lengths / "message_lengths.jsonl")
        assert flags["max_length"] == "4096"
        assert flags["overflow_mode"] == "split"
        assert flags["val_fraction"] == "0.25"
        assert "--overwrite" in argv

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["stage"] == "inline_records"
    assert manifest["inputs"]["source_id"] == make_artifact_id(source)
    assert manifest["inputs"]["image_domain"] == CROWD_CAST_IMAGE_DOMAIN
    assert [s["split"] for s in manifest["stats"]["per_split"]] == ["train", "val"]
    assert all(s["n_shards"] == 1 for s in manifest["stats"]["per_split"])


def test_stage_06_writes_only_train_without_a_val_fraction(tmp_path: Path, omegalax) -> None:
    source = make_source(tmp_path / "conv")
    out_dir = tmp_path / "records"
    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=out_dir, source_path=source, omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID, processor=MODEL_ID, max_length=4096,
        records_per_shard=8, num_workers=2,
    )
    assert proc.returncode == 0, proc.stderr
    builds = _builds(omegalax["log"])
    assert [_flags_of(a)["split"] for a in builds] == ["train"]
    # No cache given: the builder is left to tokenize in line.
    assert "message_lengths_path" not in _flags_of(builds[0])


# --------------------------------------------------------------------------
# The patch-count gap
# --------------------------------------------------------------------------


def test_the_records_stage_forwards_no_image_geometry_and_records_no_patch_count(
    tmp_path: Path, omegalax
) -> None:
    """The shape of the 792 gap, at the juergen boundary.

    Every knob that moves ``smart_resize`` — ``preprocessor_config``,
    ``min_pixels``, ``max_pixels``, ``patch_size``, ``merge_size`` — is absent
    from the argv this stage builds, so the record builder always measures under
    the processor's defaults while ``train_vlm_sft.py`` takes its own
    ``--preprocessor_config``. Nothing about patches is recorded here either, so
    the shortfall is not detectable from this side at all: it surfaces as a
    collator error on an allocated node.
    """
    source = make_source(tmp_path / "conv")
    lengths = _measure(tmp_path, omegalax, source)
    out_dir = tmp_path / "records"
    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=out_dir, source_path=source, omegalax_repo=omegalax["repo"],
        model_id=MODEL_ID, processor=MODEL_ID, max_length=4096,
        records_per_shard=8, num_workers=2, message_lengths_path=lengths,
    )
    assert proc.returncode == 0, proc.stderr
    (build,) = _builds(omegalax["log"])
    flags = _flags_of(build)
    assert set(flags) == {
        "data_path", "out_dir", "model_id", "processor", "max_length",
        "records_per_shard", "num_workers", "overflow_mode", "val_fraction",
        "split", "message_lengths_path",
    }
    assert not {"preprocessor_config", "min_pixels", "max_pixels", "patch_size",
                "merge_size"} & set(flags)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert set(manifest["params"]) == {
        "model_id", "processor", "max_length", "records_per_shard", "num_workers",
        "omegalax_repo", "overflow_mode", "message_lengths_path", "val_fraction",
    }
    assert set(manifest["stats"]["per_split"][0]) == {"split", "n_shards", "elapsed_s"}


def test_a_cache_measured_under_a_different_processor_is_still_accepted(
    tmp_path: Path, omegalax
) -> None:
    """The identity check is the chat artifact, not the geometry.

    Stage 06 refuses a cache measured from another dataset (asserted above) and
    accepts one measured from this dataset under another image processor, which
    is the configuration that produced a recorded max below the collator's. Both
    processors are recorded — in two manifests nothing compares.
    """
    source = make_source(tmp_path / "conv")
    lengths = _measure(tmp_path, omegalax, source, processor="OtherOrg/OtherProcessor")
    proc = _run(
        "stage_06_training_records.py", omegalax["env"],
        output_dir=tmp_path / "records", source_path=source,
        omegalax_repo=omegalax["repo"], model_id=MODEL_ID, processor=MODEL_ID,
        max_length=4096, records_per_shard=8, num_workers=2,
        message_lengths_path=lengths,
    )
    assert proc.returncode == 0, proc.stderr
    measured = json.loads((lengths / "manifest.json").read_text())
    records = json.loads((tmp_path / "records" / "manifest.json").read_text())
    assert measured["inputs"]["source_id"] == records["inputs"]["source_id"]
    assert measured["params"]["processor"] != records["params"]["processor"]


def test_the_patch_count_a_record_carries_is_a_function_of_the_processor_budget() -> None:
    """The 792 gap's mechanism, on the fixture's frame geometry.

    The measured count and the collated count are the same product over
    ``image_grid_thw``, so nothing is asserted by comparing them; the numbers
    below are asserted against the grid instead. Measure under the processor's
    defaults, collate under a training-side floor that is 4x higher, and the
    same 64x48 frame carries 80 patches where 16 were recorded — a max the
    training run sized its vision buffers from and overflowed by 64.
    """
    default = _grid_thw(clip.FRAME_H, clip.FRAME_W, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert default == (1, 4, 4)
    recorded_max = vision_patches(default)
    assert recorded_max == 16
    # Tokens are patches over merge_size**2 exactly, because smart_resize
    # returns multiples of patch_size * merge_size — so a token budget that
    # fits says nothing about whether the patch budget does.
    assert recorded_max % (MERGE_SIZE**2) == 0
    assert vision_tokens(default) == recorded_max // MERGE_SIZE**2 == 4

    raised = _grid_thw(
        clip.FRAME_H, clip.FRAME_W, min_pixels=16 * 28 * 28, max_pixels=MAX_PIXELS
    )
    assert raised == (1, 8, 10)
    produced = vision_patches(raised)
    assert produced == 80
    assert produced - recorded_max == 64
    # The token count moves too, but stays inside the same max_length budget,
    # which is the only budget stage 06 passes across the boundary.
    assert vision_tokens(raised) == 20
