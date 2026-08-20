"""The gate's arms, as `DesktopHarnessConfig`s over one taskset.

The calibration semantics:

  * the oracle arm must read 4/4 — the suite is solvable, the executor dispatches,
    and the guest probe can observe success;
  * the negative arm must read 0/4 — a plausibly-wrong action through the same
    parse/compile/executor path is scored as a failure, so a pass is not an
    artefact of the harness;
  * both exist per grammar, since a control that certified only one grammar would
    leave the other's parse/compile path unmeasured;
  * the model arms are what the four controls calibrate. Their published readings
    are off-the-shelf-4B-native 4/4 and Phase-B-compact 2/4. A model number is
    uncalibrated without its two controls in the same configuration.

Baseline incomparability — read this before quoting a number. The only calibrated
external reference we have is off-the-shelf Qwen3-VL-8B = 33.9% OSWorld-Verified,
and it was measured through the sealed prompts. `native_absolute.describe()`,
`move_rel.describe()` and `native_absolute_control.describe()` are docstring-derived
and are not byte-identical to those prompts, so numbers must not be compared across
that boundary and the baseline needs re-measuring through the new prompt before any
model arm here is read against 33.9%. Every episode records this on
`trace.info["prompt"]` (`comparable_to_sealed_baseline: false`).

The matched pair is machine-asserted. `compact_raw` and `native_absolute_control`
declare each other via `PAIRED_WITH`, contribute byte-identical handler sets, and
share the line-extraction rule, the `" ; "` separator and the element vocabulary; a
`matched_pair` vector pins one intent in two
encodings to one operation sequence, so any difference the gate measures between
those two arms is the encoding, not the executor. Two consequences:
`0 0 0 ; +LMB -LMB` is a click at the top-left corner in the absolute arm and
"don't move, click here" in `compact_raw` — same bytes, different action; and
`compact_raw.from_target` needs a fresh cursor read and is wrong if that read is
stale, while `native_absolute_control.from_target` needs only element geometry. The
scripted arms render one intent per step so that asymmetry is exercised rather than
papered over.

Four control arms + two model arms over one taskset: the arm is the config.
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
    "PHASEB_EXPORT_MANIFEST_SHA256",
    "PHASEB_SYSTEM_PROMPT_SHA256",
    "verify_phaseb_provenance",
]

NATIVE_CODEC = "native_absolute"
COMPACT_CODEC = "deltatype_v2"
"""The Phase-B compact grammar. `compact_raw` is its NO_OP-free sibling and is the
arm paired with `native_absolute_control`; `deltatype_v2` is the one the s900
checkpoint was trained on, so it is the one the model arm uses."""

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
}
"""The four control arms: {oracle, negative} x {native, compact}.

Not the four suite cells — those are `terminal_ls`, `terminal_exact_text`,
`desktop_open_chrome` and `focus_terminal_and_type`, and they live in `suite.json`
behind `suite.load_suite()`. Every arm here runs all four of them."""


MODEL_ARMS: dict[str, DesktopHarnessConfig] = {
    "offshelf_native": DesktopHarnessConfig(
        id="sol_offshelf_native",
        codec=NATIVE_CODEC,
        history=HistoryConfig(name="interleaved_frames", n_history_frames=8),
        images=ImageBudgetConfig(max_images=8),
        settle=_settle(),
        max_tokens=256,
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
            "now the harness control channel and no longer parse as actions -- so "
            "this checkpoint's terminations arrive as parse errors and this cell's "
            "2/4 reading has to be re-measured, not assumed"
        ),
        settle=_settle(),
        max_tokens=256,
        artifacts=ArtifactConfig(save_prompts=True, write_gif=True),
    ),
}
"""The arms the four controls calibrate. Reference readings: off-the-shelf
Qwen3-VL-4B native = 4/4, Phase-B step-900 compact = 2/4."""


ARMS: dict[str, DesktopHarnessConfig] = {**CONTROL_ARMS, **MODEL_ARMS}
