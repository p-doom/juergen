"""The gate's arms, as `DesktopHarnessConfig`s over one taskset.

The calibration semantics, per tier — an arm runs one tier at a time
(`--tier`), and a reading is only calibrated by controls from the same tier:

  * the oracle arm must pass every cell of the tier — it is solvable, the executor
    dispatches, and the guest probe can observe success;
  * the negative arm must fail every cell — a plausibly-wrong action through the
    same parse/compile/executor path is scored as a failure, so a pass is not an
    artefact of the harness;
  * both exist per grammar, since a control that certified only one grammar would
    leave the other's parse/compile path unmeasured;
  * the model arms are what the controls calibrate. Scored-tier readings, all over
    3 trials: off-the-shelf Qwen3-VL-4B native = 3/4 (run
    019fd5be788b7793a8a777da2a0f7531, original qcow -- `focus_terminal_and_type`
    0/3, every draw `model_terminate_without_postcondition`) and Phase-B-compact =
    2/4 (run 01a01e7c171e7dc1b681725c7066d0bd). The 4/4 once recorded here for the
    off-the-shelf arm is not what its result.json says. A model number is
    uncalibrated without its two controls in the same configuration and the same
    tier. Every native reading above is also STALE for a second, independent
    reason: it was read while `native_absolute` declared absolute pixels and
    consumed the model's 0-999 answer as one, so every native arm — oracle,
    negative and model — needs re-running before any of those numbers is quoted
    again. Only `desktop_open_chrome`'s dock icon sits where the two conventions
    nearly agree, which is how a wrong convention scored at all.

Baseline incomparability — read this before quoting a number. The only calibrated
external reference we have is off-the-shelf Qwen3-VL-8B = 33.9% OSWorld-Verified,
and it was measured through the sealed prompts. `native_absolute.describe()`,
`move_rel.describe()` and `compact_absolute.describe()` are docstring-derived
and are not byte-identical to those prompts, so numbers must not be compared across
that boundary and the baseline needs re-measuring through the new prompt before any
model arm here is read against 33.9%. Every episode records this on
`trace.info["prompt"]` (`comparable_to_sealed_baseline: false`).

The matched pair is machine-asserted. `compact_raw` and `compact_absolute`
declare each other via `PAIRED_WITH`, contribute byte-identical handler sets, and
share the line-extraction rule, the `" ; "` separator and the element vocabulary; a
`matched_pair` vector pins one intent in two
encodings to one operation sequence, so any difference the gate measures between
those two arms is the encoding, not the executor. Two consequences:
`0 0 0 ; +LMB -LMB` is a click at the top-left corner in the absolute arm and
"don't move, click here" in `compact_raw` — same bytes, different action; and
`compact_raw.from_target` needs a fresh cursor read and is wrong if that read is
stale, while `compact_absolute.from_target` needs only element geometry. The
scripted arms render one intent per step so that asymmetry is exercised rather than
papered over.

Six control arms + three model arms over one taskset: the arm is the config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evals.harness import (
    ArtifactConfig,
    DesktopHarnessConfig,
    HistoryConfig,
    ImageBudgetConfig,
    ScriptedConfig,
    SettleConfig,
)

__all__ = [
    "ARMS",
    "CONTROL_ARMS",
    "COMPACT_CODEC",
    "MODEL_ARMS",
    "NATIVE_CODEC",
    "ORDERED_CODEC",
    "PHASEB_EXPORT_MANIFEST_SHA256",
    "PHASEB_SYSTEM_PROMPT_SHA256",
    "verify_phaseb_provenance",
]

NATIVE_CODEC = "native_absolute"
COMPACT_CODEC = "deltatype_v2"
"""The Phase-B compact grammar. `compact_raw` is its NO_OP-free sibling and is the
arm paired with `compact_absolute`; `deltatype_v2` is the one the s900
checkpoint was trained on, so it is the one the model arm uses."""
ORDERED_CODEC = "ordered_events_v3"
"""The production format. It had no eval leg at all — no model arm and no scripted
renderer — while a training job was already running on it, so every number it
produced would have been uncalibrated."""

ORDERED_SYSTEM_PROMPT_SHA256 = (
    "ce1ef849674019c3a365eb5aaf4ddc569084937aea41a54d6ebacd19610245be"
)
"""`ordered_events_v3.describe()` — the prompt stage 04 writes into `chat.jsonl`,
so a checkpoint trained under any other one is refused rather than scored.

This is trunk's CURRENT digest, and
`test_every_shipped_arm_passes_its_own_prompt_digest_check` is what keeps it so:
the value first proposed here was `13e761cb…`, already 63 tokens stale, and that
test is what caught it. This arm sets no `expect_prompt_mismatch`, so any pin
that is not the digest `describe()` renders today fails there rather than at the
first VM dispatch."""

PHASEB_SYSTEM_PROMPT_SHA256 = (
    "57f7d0b230974068618b48151b73215d5517d5445a99dbf5abdc05557e3482e6"
)
PHASEB_ARTIFACT_ID = "artifact_896c1de00b60c27c"
PHASEB_PRODUCER_RUN_ID = "run_019fba52e90778e0b8ae170058c814e7"
PHASEB_EXPORT_MANIFEST_SHA256 = (
    "9c141897dec6b468c35d9eb522907b32cde34d925d8771e49dd018943cf5530c"
)
PHASEB_EXPECTED_MANIFEST = {
    "arm": "raw_v2",
    "step": 900,
    "lora_rank": 256,
    "lora_alpha": 256,
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "status": "complete",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_phaseb_provenance(model_path: Path) -> dict[str, Any]:
    """Fail closed unless the checkpoint is the registered step-900 export.

    Called at dispatch, because the harness does not launch the server. The compact
    arm's claim is that this specific checkpoint reads 2/4, so a silently
    substituted export invalidates it.
    """
    root = model_path.parent
    metadata_path = root / ".meta.json"
    manifest_path = root / "export_manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Phase-B checkpoint registration metadata is missing")
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = _sha256_file(manifest_path)
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in PHASEB_EXPECTED_MANIFEST.items()
        if manifest.get(key) != value
    }
    if metadata.get("id") != PHASEB_ARTIFACT_ID:
        mismatches["artifact_id"] = {
            "expected": PHASEB_ARTIFACT_ID,
            "observed": metadata.get("id"),
        }
    if metadata.get("producer_run_id") != PHASEB_PRODUCER_RUN_ID:
        mismatches["producer_run_id"] = {
            "expected": PHASEB_PRODUCER_RUN_ID,
            "observed": metadata.get("producer_run_id"),
        }
    if manifest_sha != PHASEB_EXPORT_MANIFEST_SHA256:
        mismatches["export_manifest_sha256"] = {
            "expected": PHASEB_EXPORT_MANIFEST_SHA256,
            "observed": manifest_sha,
        }
    weights = manifest.get("weights")
    weight_path = model_path / "model.safetensors"
    if (
        not isinstance(weights, list)
        or len(weights) != 1
        or weights[0].get("name") != "model.safetensors"
        or not weight_path.is_file()
        or weight_path.stat().st_size != weights[0].get("size")
    ):
        mismatches["weights"] = {
            "expected": weights,
            "observed_size": weight_path.stat().st_size if weight_path.is_file() else None,
        }
    if mismatches:
        raise RuntimeError(f"Phase-B checkpoint provenance mismatch: {mismatches}")
    return {
        "artifact_id": metadata["id"],
        "producer_run_id": metadata["producer_run_id"],
        "export_manifest": str(manifest_path),
        "export_manifest_sha256": manifest_sha,
        "config_sha256": _sha256_file(model_path / "config.json"),
        "weight_sha256": weights[0]["sha256"],
        "weight_bytes": weights[0]["size"],
        "arm": manifest["arm"],
        "step": manifest["step"],
        "lora_rank": manifest["lora_rank"],
        "lora_alpha": manifest["lora_alpha"],
    }


def _settle() -> SettleConfig:
    # 2.0 s only for the Chrome cell (a launch needs time to become foreground),
    # 0.75 s elsewhere. A global 2.0 s would triple every other cell's wall clock.
    return SettleConfig(min_delay_s=0.75, per_kind={"open_chrome": 2.0})


def _control(codec: str, *, negative: bool, name: str) -> DesktopHarnessConfig:
    return DesktopHarnessConfig(
        id=name,
        codec=codec,
        scripted=ScriptedConfig(enabled=True, negative=negative),
        history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
        settle=_settle(),
        require_unsolved_start=True,
        artifacts=ArtifactConfig(save_prompts=False, write_gif=True),
    )


CONTROL_ARMS: dict[str, DesktopHarnessConfig] = {
    "native_oracle": _control(NATIVE_CODEC, negative=False, name="sol_native_oracle"),
    "native_negative": _control(NATIVE_CODEC, negative=True, name="sol_native_negative"),
    "compact_oracle": _control(COMPACT_CODEC, negative=False, name="sol_compact_oracle"),
    "compact_negative": _control(COMPACT_CODEC, negative=True, name="sol_compact_negative"),
    "ordered_oracle": _control(ORDERED_CODEC, negative=False, name="sol_ordered_oracle"),
    "ordered_negative": _control(ORDERED_CODEC, negative=True, name="sol_ordered_negative"),
}
"""The six control arms: {oracle, negative} x {native, compact, ordered}.

Not the suite cells — those live in `suite.json` behind `suite.load_suite()`, and
every arm here runs a whole tier of them.

A model arm never ships without its pair: the pair is what says a reading is the
model's and not the harness's. `ordered_events_v3` had a model arm's worth of
training behind it and no arm at all, which is the same defect one step earlier."""


MODEL_ARMS: dict[str, DesktopHarnessConfig] = {
    "offshelf_native": DesktopHarnessConfig(
        id="sol_offshelf_native",
        codec=NATIVE_CODEC,
        history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
        images=ImageBudgetConfig(max_images=8),
        settle=_settle(),
        max_tokens=256,
        # Inherited: the command-line default every recorded run of this arm took.
        # Never validated as a choice -- nothing here was measured against another
        # temperature, so it claims only reproducibility.
        temperature=0.0,
        top_p=1.0,
        artifacts=ArtifactConfig(save_prompts=True, write_gif=True),
    ),
    "phaseb_compact": DesktopHarnessConfig(
        id="sol_phaseb_compact",
        codec=COMPACT_CODEC,
        # The checkpoint's sealed training contract: five images (four completed
        # turns plus the current screen), earlier actions carried as prose, and the
        # assistant's prose preserved ahead of the final bare action line.
        history=HistoryConfig(name="prose_summarised_window"),
        images=ImageBudgetConfig(max_images=5),
        system_prompt_sha256=PHASEB_SYSTEM_PROMPT_SHA256,
        expect_prompt_mismatch=(
            "the s900 checkpoint's prompt was sealed before describe() existed and "
            "cannot be recomputed from the codec, so the two digests differ by "
            "construction. Two known differences: describe() documents type(), "
            "which the sealed prompt omits and its parser accepts anyway, and the "
            "sealed prompt declares bare TERMINATE / FAIL action lines, which are "
            "now the harness control channel. Only the digest differs: the arm "
            "renders the CURRENT describe(), the only prompt this checkpoint is "
            "ever shown, and it complies -- it emits the channel's TERMINATE: "
            "success and the episode stops on that line"
        ),
        settle=_settle(),
        max_tokens=256,
        # Inherited and unvalidated, as for `offshelf_native`.
        temperature=0.0,
        top_p=1.0,
        artifacts=ArtifactConfig(save_prompts=True, write_gif=True),
    ),
    "ordered": DesktopHarnessConfig(
        id="sol_ordered",
        codec=ORDERED_CODEC,
        history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
        images=ImageBudgetConfig(max_images=8),
        system_prompt_sha256=ORDERED_SYSTEM_PROMPT_SHA256,
        settle=_settle(),
        max_tokens=256,
        # Measured, not a preference: greedy collapses this family's mouse deltas
        # onto the {0, ±1, ±10, ±100} lattice and lands no click on its target. See
        # `DesktopHarnessConfig.temperature`.
        temperature=0.7,
        top_p=1.0,
        artifacts=ArtifactConfig(save_prompts=True, write_gif=True),
    ),
}
"""The arms the controls calibrate. Scored-tier reference readings: off-the-shelf
Qwen3-VL-4B native = 3/4, Phase-B step-900 compact = 2/4, both over 3 trials.

READ THIS BEFORE QUOTING ANY CLICK-BASED NUMBER FROM `offshelf_native` MEASURED
BEFORE `native_absolute` NAMED ITS GRID. That arm emits 0-999-per-axis
coordinates, and the grammar's prompt used to declare "Coordinates are ABSOLUTE
screen pixels" with nothing rescaling, so every click landed at roughly (0.51x,
0.92y) of where the model aimed. Measured on the two panel cells over six
episodes (job 141319): emitted (502,614) and (473,463), which de-normalise to
(964,663) and (908,500) -- both INSIDE the measured target, 6 and 22 px from its
centre, against targets 172x31 and 230x23. The model grounded them correctly and
the arm reported a miss.

Why the defect stayed invisible, which is the part that outlives the fix: near
(0,0) the two conventions nearly coincide -- emitted (15,59), de-normalised
(29,64), dock icon at (35,60) -- and this arm's 3/4 survived only because its
clicks landed inside a 1120x720 window either way. Any absolute-grammar reading
whose cells all sit near the origin certifies its convention weakly.

`native_absolute.compile` now resolves the 0-999 grid to pixels itself
(`pixels_from_norm`), so the arm and the model agree and no coordinate-space enum
was added: `compile` still takes exactly one convention per grammar and every
Operation downstream is still absolute pixels.

`ordered` has no reference reading yet, but its prompt digest is pinned: a
checkpoint trained under any other prompt is refused instead of scored."""


ARMS: dict[str, DesktopHarnessConfig] = {**CONTROL_ARMS, **MODEL_ARMS}
