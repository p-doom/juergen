#!/usr/bin/env python3
"""Fail-closed CPU gate for roadmap stage-1.5 endpoint-actuation conformance.

This module deliberately does not launch a VM or model server.  It freezes and
checks the inputs, reconstructs all 320 paired geometry cells, verifies the
canonical PNG bytes, tests arm-specific pyautogui actuation plans against a
stateful fake desktop, and implements the prespecified 5pp paired gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from PIL import Image

try:
    from ..contract import (
        Contract,
        ContractError,
        request_seed,
        sha256_bytes,
        sha256_file,
        strict_schema_ok,
        unit_range_ok,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from contract import (  # type: ignore
        Contract,
        ContractError,
        request_seed,
        sha256_bytes,
        sha256_file,
        strict_schema_ok,
        unit_range_ok,
    )


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
Operation = Literal["click", "drag"]
Semantic = Literal["absolute_toolcall", "move_rel", "deltatype_raw"]
Command = tuple[Any, ...]


@dataclass(frozen=True)
class Cell:
    cell_id: str
    episode_id: str
    episode_index: int
    target_index: int
    cursor: tuple[int, int]
    bbox: tuple[int, int, int, int]
    target: tuple[int, int]
    image_path: Path
    image_sha256: str
    request_seed: int


class GateError(RuntimeError):
    pass


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def load_protocol(
    path: Path = PROTOCOL_PATH,
    *,
    require_launch_authorized: bool | None = False,
) -> dict[str, Any]:
    value = _load_object(path)
    if value.get("schema_version") != 1:
        raise GateError("protocol schema drift")
    status = value.get("status")
    authorized = value.get("launch_gate", {}).get("authorized")
    valid_state = (
        (status == "prepared_not_launched" and authorized is False)
        or (status == "authorized_ready" and authorized is True)
    )
    if not valid_state:
        raise GateError("protocol status/authorization state is inconsistent")
    if require_launch_authorized is False and authorized is not False:
        raise GateError("CPU preparation must not authorize launch")
    if require_launch_authorized is True and authorized is not True:
        raise GateError("GPU launch is not explicitly authorized")
    return value


def _nested_integer_values(value: Any) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _nested_integer_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _nested_integer_values(child)]
    return []


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    scope = protocol.get("scope_classification", {})
    required_scope = {
        "roadmap_stage": "1.5",
        "label": "proper-VM endpoint-actuation conformance",
        "is_user_roadmap_stage_2": False,
        "is_free_running_multi_step_closed_loop": False,
    }
    if any(scope.get(key) != value for key, value in required_scope.items()):
        raise GateError("roadmap stage-1.5 scope classification drift")
    source_paths = {
        "aggregate.py": HERE / "aggregate.py",
        "assemble_chunks.py": HERE / "assemble_chunks.py",
        "gate.py": HERE / "gate.py",
        "guest_app.py": HERE / "guest_app.py",
        "lifecycle_stress.py": HERE / "lifecycle_stress.py",
        "live_smoke.py": HERE / "live_smoke.py",
        "run_arm.py": HERE / "run_arm.py",
        "run_arm_stage.sh": HERE / "run_arm_stage.sh",
        "proper_vm_stage2_absolute_prepared.toml": (
            HERE.parent / "labctl" / "recipes" / "proper_vm_stage2_absolute_prepared.toml"
        ),
        "proper_vm_stage2_live_smoke_prepared.toml": (
            HERE.parent / "labctl" / "recipes" / "proper_vm_stage2_live_smoke_prepared.toml"
        ),
        "proper_vm_stage1_5_lifecycle_stress_recovery_prepared.toml": (
            HERE.parent
            / "labctl"
            / "recipes"
            / "proper_vm_stage1_5_lifecycle_stress_recovery_prepared.toml"
        ),
        "proper_vm_stage2_normalized_prepared.toml": (
            HERE.parent / "labctl" / "recipes" / "proper_vm_stage2_normalized_prepared.toml"
        ),
        "proper_vm_stage2_raw_a_to_b_prepared.toml": (
            HERE.parent / "labctl" / "recipes" / "proper_vm_stage2_raw_a_to_b_prepared.toml"
        ),
    }
    for arm in ("absolute", "normalized", "raw_a_to_b"):
        for chunk_index in range(4):
            name = f"proper_vm_stage2_{arm}_chunk{chunk_index}_recovery.toml"
            source_paths[name] = HERE.parent / "labctl" / "recipes" / name
    if set(protocol.get("implementation_sources", {})) != set(source_paths):
        raise GateError("implementation source set drift")
    for name, path in source_paths.items():
        if sha256_file(path) != protocol["implementation_sources"][name]:
            raise GateError(f"implementation source hash drift: {name}")
    expected_arms = {
        "absolute_phase_a": "absolute_toolcall",
        "normalized_phase_a": "move_rel",
        "raw_a_to_b": "deltatype_raw",
    }
    arms = protocol.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(expected_arms):
        raise GateError(f"arm set drift: {set(arms or {})}")
    checked: dict[str, Any] = {}
    for arm_name, semantic in expected_arms.items():
        arm = arms[arm_name]
        if arm.get("semantic") != semantic:
            raise GateError(f"{arm_name}: semantic drift")
        root = Path(arm["checkpoint_root"])
        manifest_path = root / arm["checkpoint_manifest"]
        if not manifest_path.is_file():
            raise GateError(f"{arm_name}: missing checkpoint manifest")
        actual_hash = sha256_file(manifest_path)
        if actual_hash != arm["checkpoint_manifest_sha256"]:
            raise GateError(f"{arm_name}: checkpoint manifest hash drift")
        manifest = _load_object(manifest_path)
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in arm["expected_manifest"].items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise GateError(f"{arm_name}: checkpoint manifest mismatch: {mismatches}")
        hf = root / manifest["hf_subdir"]
        weights = hf / "model.safetensors"
        for needed in (hf / "config.json", hf / "tokenizer.json", weights):
            if not needed.is_file():
                raise GateError(f"{arm_name}: missing HF file {needed}")
        if weights.stat().st_size != arm["model_weights_bytes"]:
            raise GateError(f"{arm_name}: model weight size drift")
        weights_hash = sha256_file(weights)
        if weights_hash != arm["model_weights_sha256"]:
            raise GateError(f"{arm_name}: model weight hash drift")
        checked[arm_name] = {
            "semantic": semantic,
            "manifest_sha256": actual_hash,
            "weights_bytes": weights.stat().st_size,
            "weights_sha256": weights_hash,
        }
    vm = protocol["vm"]
    provider = Path(vm["provider_source"])
    if sha256_file(provider) != vm["provider_sha256"]:
        raise GateError("KVM provider hash drift")
    provider_text = provider.read_text(encoding="utf-8")
    if "snapshot=on" not in provider_text or '"-enable-kvm"' not in provider_text:
        raise GateError("provider is not the pinned snapshot/KVM implementation")
    if "_free_port" in provider_text:
        raise GateError("provider contains forbidden racy port selection")
    if not Path(vm["qcow"]).is_file():
        raise GateError("pinned qcow is missing")
    qemu_bin = Path(vm["qemu_bin"])
    if not qemu_bin.is_file() or not qemu_bin.stat().st_mode & 0o111:
        raise GateError("pinned QEMU executable is absent/nonexecutable")
    smoke = protocol.get("live_smoke_evidence", {})
    smoke_path = Path(str(smoke.get("manifest", "")))
    if (
        smoke.get("status") != "pass"
        or smoke.get("gpu_used") is not False
        or smoke.get("replays") != 6
        or not smoke_path.is_file()
        or sha256_file(smoke_path) != smoke.get("manifest_sha256")
    ):
        raise GateError("live KVM smoke evidence is absent or drifted")
    smoke_manifest = _load_object(smoke_path)
    smoke_required = {
        "status": "pass",
        "cpu_only": True,
        "gpu_used": False,
        "kvm_read_write": True,
        "replay_count": 6,
        "provider_sha256": vm["provider_sha256"],
        "guest_app_sha256": protocol["implementation_sources"]["guest_app.py"],
    }
    if any(smoke_manifest.get(key) != expected for key, expected in smoke_required.items()):
        raise GateError("live KVM smoke manifest contract drift")
    recovery = protocol.get("execution_recovery")
    if not isinstance(recovery, dict) or recovery.get("scientific_design_changed") is not False:
        raise GateError("execution-recovery amendment is absent or changes the design")
    if recovery.get("historical_failure_cells_per_arm") != 115:
        raise GateError("execution-recovery incident boundary drift")
    lifecycle = protocol.get("lifecycle_stress_evidence", {})
    if lifecycle.get("status") == "prepared_not_run":
        if protocol["launch_gate"]["authorized"] is not False:
            raise GateError("GPU launch cannot be authorized before lifecycle stress")
    elif lifecycle.get("status") == "pass":
        lifecycle_path = Path(str(lifecycle.get("manifest", "")))
        if not lifecycle_path.is_file() or sha256_file(lifecycle_path) != lifecycle.get(
            "manifest_sha256"
        ):
            raise GateError("lifecycle-stress evidence is absent or drifted")
        lifecycle_manifest = _load_object(lifecycle_path)
        required_lifecycle = {
            "artifact_type": "synthetic_proper_vm_stage1_5_guest_lifecycle_stress",
            "status": "pass",
            "cpu_only": True,
            "gpu_used": False,
            "exceeded_historical_boundary": True,
            "exact_source_processes_after_each_scene": 0,
        }
        if any(
            lifecycle_manifest.get(key) != value
            for key, value in required_lifecycle.items()
        ):
            raise GateError("lifecycle-stress evidence content drift")
        if int(lifecycle_manifest.get("cells", 0)) < 131 or int(
            lifecycle_manifest.get("scenes", 0)
        ) < 262:
            raise GateError("lifecycle stress did not exceed the failure boundary")
    elif lifecycle.get("status") == "failed_fallback_authorized":
        failure_log = Path(str(lifecycle.get("failure_log", "")))
        if (
            protocol["launch_gate"]["authorized"] is not True
            or lifecycle.get("job_id") != "135624"
            or not failure_log.is_file()
            or sha256_file(failure_log) != lifecycle.get("failure_log_sha256")
            or "Maximum number of clients reached"
            not in failure_log.read_text(encoding="utf-8", errors="replace")
        ):
            raise GateError("failed lifecycle evidence/fallback authorization drift")
        fallback = recovery.get("active_fallback", {})
        if fallback != {
            "kind": "four_disjoint_fresh_vm_chunks",
            "bounds": [[0, 80], [80, 160], [160, 240], [240, 320]],
            "fresh_vm_per_chunk": True,
            "max_cells_per_chunk": 80,
            "strict_exact_coverage_assembly": True,
            "oracle_prefix_independent_at_chunk_boundaries": True,
        }:
            raise GateError("fixed four-chunk fallback plan drift")
        screenshot_recovery = recovery.get("screenshot_readiness_recovery", {})
        screenshot_failure_log = Path(str(screenshot_recovery.get("failure_log", "")))
        screenshot_failure_output = Path(str(screenshot_recovery.get("failure_output", "")))
        required_screenshot_recovery = {
            "status": "bounded_exact_hash_poll_retry_authorized",
            "job_id": "135634",
            "run_id": "run_019fb6bbd1c97ee19d46c022f3e57cbd",
            "arm": "absolute_phase_a",
            "chunk_index": 3,
            "failed_cell": "phasea_short_0060:t00",
            "failed_operation": "click",
            "rows_written": 0,
            "failed_actual_hash_available": False,
            "preserves_exact_decoded_pixel_gate": True,
            "timeout_s": 5.0,
            "poll_s": 0.1,
            "hash_history": 8,
            "prior_protocol_sha256": "a71507ea76aca242df387d7d5daccf1b70872a2ce788cf201efd58d1bb76cc80",
        }
        if (
            any(
                screenshot_recovery.get(key) != value
                for key, value in required_screenshot_recovery.items()
            )
            or not screenshot_failure_log.is_file()
            or sha256_file(screenshot_failure_log)
            != screenshot_recovery.get("failure_log_sha256")
            or "live screenshot pixel mismatch: phasea_short_0060:t00/click"
            not in screenshot_failure_log.read_text(encoding="utf-8", errors="replace")
            or not screenshot_failure_output.is_dir()
            or any(
                (screenshot_failure_output / name).exists()
                for name in ("rows.partial.jsonl", "rows.jsonl", "chunk_manifest.json")
            )
        ):
            raise GateError("screenshot-readiness failure/retry evidence drift")
        if recovery.get("accepted_prior_chunk_protocols") != [
            {
                "protocol_sha256": "a71507ea76aca242df387d7d5daccf1b70872a2ce788cf201efd58d1bb76cc80",
                "scopes": {
                    "absolute_phase_a": [0, 1, 2],
                    "normalized_phase_a": [0, 1, 2, 3],
                    "raw_a_to_b": [0, 1, 2, 3],
                },
            }
        ]:
            raise GateError("accepted prior chunk-protocol lineage drift")
    else:
        raise GateError("unknown lifecycle-stress evidence state")
    no_leak = protocol["no_leak"]
    overlap_path = Path(no_leak["raw_curriculum_transfer_overlap_report"])
    if sha256_file(overlap_path) != no_leak[
        "raw_curriculum_transfer_overlap_report_sha256"
    ]:
        raise GateError("raw curriculum-training overlap report hash drift")
    overlap = _load_object(overlap_path)
    values = _nested_integer_values(overlap.get("overlap_counts"))
    if not values or any(value != no_leak["required_raw_overlap_counts"] for value in values):
        raise GateError("raw curriculum-training overlap counts are not all zero")
    return checked


def load_cells(protocol: dict[str, Any], contract: Contract) -> list[Cell]:
    geometry = protocol["geometry"]
    root = Path(geometry["episode_artifact"])
    manifest_path = root / "build_manifest.json"
    specs_path = root / "episode_specs.jsonl"
    oracle_path = root / "oracle_absolute_toolcall.jsonl"
    if sha256_file(manifest_path) != geometry["build_manifest_sha256"]:
        raise GateError("episode build manifest hash drift")
    if sha256_file(specs_path) != geometry["episode_specs_sha256"]:
        raise GateError("episode specs hash drift")
    if sha256_file(oracle_path) != geometry["absolute_oracle_sha256"]:
        raise GateError("absolute oracle hash drift")
    manifest = _load_object(manifest_path)
    required_manifest = {
        "status": "complete",
        "n_episodes": geometry["episodes"],
        "targets_per_episode": geometry["targets_per_episode"],
        "n_oracle_targets": geometry["paired_cells"],
        "oracle_observations_and_states_identical": True,
        "action_serialization_matched_except_semantics": True,
    }
    mismatch = {
        key: (manifest.get(key), expected)
        for key, expected in required_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatch:
        raise GateError(f"episode manifest contract mismatch: {mismatch}")
    leak_report = manifest.get("leak_report")
    required_leak_fields = protocol["no_leak"]["phase_a_episode_manifest_leak_fields_required_zero"]
    if not isinstance(leak_report, dict):
        raise GateError("episode leak report missing")
    for field in required_leak_fields:
        values = _nested_integer_values(leak_report.get(field))
        if not values or any(values):
            raise GateError(f"episode leak report is nonzero/missing: {field}")
    cells: list[Cell] = []
    lines = specs_path.read_text(encoding="utf-8").splitlines()
    oracle_lines = oracle_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != geometry["episodes"] or any(not line.strip() for line in lines):
        raise GateError("episode row count/blank-line drift")
    if len(oracle_lines) != len(lines) or any(not line.strip() for line in oracle_lines):
        raise GateError("absolute oracle row count/blank-line drift")
    for line_no, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"bad episode JSON line {line_no}: {exc}") from exc
        if row.get("episode_index") != line_no - 1:
            raise GateError(f"episode ordering drift at line {line_no}")
        try:
            oracle_row = json.loads(oracle_lines[line_no - 1])
        except json.JSONDecodeError as exc:
            raise GateError(f"bad oracle JSON line {line_no}: {exc}") from exc
        if oracle_row.get("episode_id") != row.get("episode_id"):
            raise GateError(f"episode/oracle ordering drift at line {line_no}")
        targets = row.get("targets")
        oracle_turns = oracle_row.get("turns")
        if not isinstance(targets, list) or len(targets) != geometry["targets_per_episode"]:
            raise GateError(f"wrong target count: {row.get('episode_id')}")
        if not isinstance(oracle_turns, list) or len(oracle_turns) != len(targets):
            raise GateError(f"wrong oracle turn count: {row.get('episode_id')}")
        for target_index, target_row in enumerate(targets):
            oracle_turn = oracle_turns[target_index]
            if target_row.get("target_index") != target_index:
                raise GateError("target index drift")
            bbox = tuple(int(value) for value in target_row["bbox"])
            target = tuple(int(value) for value in target_row["target_center"])
            cursor = tuple(int(value) for value in oracle_turn["cursor_before"])
            if tuple(oracle_turn["bbox"]) != bbox or tuple(oracle_turn["target_center"]) != target:
                raise GateError("episode/oracle geometry mismatch")
            if target_index == 0 and cursor != tuple(row["initial_cursor"]):
                raise GateError("step-1 oracle cursor mismatch")
            expected_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            if target != expected_center:
                raise GateError("target center/bbox mismatch")
            image_path = root / "images" / f"{row['episode_id']}_t{target_index:02d}.png"
            if not image_path.is_file():
                raise GateError(f"missing frozen cell image {image_path}")
            expected_png = contract.render_png(bbox, cursor)
            actual_png = image_path.read_bytes()
            if actual_png != expected_png:
                raise GateError(f"canonical PNG byte drift: {image_path}")
            if sha256_bytes(actual_png) != oracle_turn["image_sha256"]:
                raise GateError(f"oracle image hash drift: {image_path}")
            cell_id = f"{row['episode_id']}:t{target_index:02d}"
            cells.append(
                Cell(
                    cell_id=cell_id,
                    episode_id=row["episode_id"],
                    episode_index=int(row["episode_index"]),
                    target_index=target_index,
                    cursor=cursor,
                    bbox=bbox,
                    target=target,
                    image_path=image_path,
                    image_sha256=sha256_bytes(actual_png),
                    request_seed=request_seed(row["episode_id"], 0, target_index, 1),
                )
            )
    if len(cells) != geometry["paired_cells"]:
        raise GateError("paired cell count drift")
    identities = {(cell.cursor, cell.bbox) for cell in cells}
    if len(identities) != geometry["unique_cursor_bbox_pairs"]:
        raise GateError("unique cursor/bbox count drift")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise GateError("duplicate cell id")
    if len({(cell.cell_id, cell.request_seed) for cell in cells}) != len(cells):
        raise GateError("duplicate paired seed key")
    return cells


def rgb_sha256(png: bytes) -> str:
    """Hash decoded pixels, which is the live VM screenshot equality contract."""
    import io

    with Image.open(io.BytesIO(png)) as image:
        rgb = image.convert("RGB")
        payload = rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes()
    return hashlib.sha256(payload).hexdigest()


def actuation_plan(
    semantic: Semantic,
    operation: Operation,
    cursor: tuple[int, int],
    endpoint: tuple[int, int],
) -> tuple[Command, ...]:
    """Return an auditable abstract pyautogui plan.

    Drag is an explicitly registered endpoint adapter.  The Phase-A models did
    not receive matched drag supervision, so this does not pretend that the raw
    model emitted a drag action.
    """
    dx, dy = endpoint[0] - cursor[0], endpoint[1] - cursor[1]
    if semantic == "absolute_toolcall":
        move: Command = ("moveTo", endpoint[0], endpoint[1], 0.5 if operation == "drag" else 0.0)
    elif semantic == "move_rel":
        move = ("moveRel", dx, dy, 0.5 if operation == "drag" else 0.0)
    elif semantic == "deltatype_raw":
        move = ("moveTo", endpoint[0], endpoint[1], 0.5 if operation == "drag" else 0.0)
    else:
        raise GateError(f"unknown semantic: {semantic}")
    if operation == "click":
        if semantic == "deltatype_raw":
            return (move, ("mouseDown", "left"), ("mouseUp", "left"))
        return (move, ("click", "left"))
    if operation == "drag":
        return (("mouseDown", "left"), move, ("mouseUp", "left"))
    raise GateError(f"unknown operation: {operation}")


@dataclass
class FakeDesktop:
    cursor: tuple[int, int]
    bbox: tuple[int, int, int, int]
    button_down: bool = False
    down_position: tuple[int, int] | None = None
    click_success: bool = False
    drag_success: bool = False
    event_count: int = 0

    def _inside(self, point: tuple[int, int]) -> bool:
        return self.bbox[0] <= point[0] <= self.bbox[2] and self.bbox[1] <= point[1] <= self.bbox[3]

    def execute(self, command: Command) -> None:
        op = command[0]
        if op == "moveTo":
            self.cursor = (int(command[1]), int(command[2]))
        elif op == "moveRel":
            self.cursor = (self.cursor[0] + int(command[1]), self.cursor[1] + int(command[2]))
        elif op == "click":
            if self.button_down:
                raise GateError("click while button already down")
            self.click_success = self._inside(self.cursor)
        elif op == "mouseDown":
            if self.button_down:
                raise GateError("duplicate mouseDown")
            self.button_down = True
            self.down_position = self.cursor
        elif op == "mouseUp":
            if not self.button_down:
                raise GateError("mouseUp without mouseDown")
            moved = self.down_position != self.cursor
            if moved:
                self.drag_success = self._inside(self.cursor)
            else:
                self.click_success = self._inside(self.cursor)
            self.button_down = False
        else:
            raise GateError(f"unknown fake command: {command}")
        self.event_count += 1


def replay_plan(cell: Cell, plan: Iterable[Command], operation: Operation) -> FakeDesktop:
    desktop = FakeDesktop(cell.cursor, cell.bbox)
    for command in plan:
        desktop.execute(command)
    if desktop.button_down:
        raise GateError("button remained down after replay")
    success = desktop.click_success if operation == "click" else desktop.drag_success
    if not success:
        raise GateError(f"oracle {operation} replay failed: {cell.cell_id}")
    return desktop


def _binomial_cdf(x: int, n: int, probability: float) -> float:
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if x >= n else 0.0
    return sum(
        math.comb(n, k) * probability**k * (1.0 - probability) ** (n - k)
        for k in range(x + 1)
    )


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper confidence limit for a binomial proportion."""
    if not (0 <= successes <= trials) or trials <= 0 or not (0 < alpha < 1):
        raise ValueError("invalid binomial arguments")
    if successes == trials:
        return 1.0
    low, high = successes / trials, 1.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if _binomial_cdf(successes, trials, mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def maximum_confirmatory_harms(
    trials: int, margin: float = 0.05, alpha: float = 0.05
) -> int:
    """Largest harm count whose exact upper bound remains below the margin."""
    maximum = -1
    for harms in range(trials + 1):
        if clopper_pearson_upper(harms, trials, alpha) >= margin:
            break
        maximum = harms
    return maximum


def paired_noninferiority(
    absolute: dict[str, bool],
    treatment: dict[str, bool],
    *,
    margin: float = 0.05,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if set(absolute) != set(treatment) or not absolute:
        raise GateError("nonidentical/empty paired trial ids")
    ids = sorted(absolute)
    absolute_only = sum(absolute[key] and not treatment[key] for key in ids)
    treatment_only = sum(treatment[key] and not absolute[key] for key in ids)
    both_success = sum(absolute[key] and treatment[key] for key in ids)
    both_failure = len(ids) - absolute_only - treatment_only - both_success
    difference = (treatment_only - absolute_only) / len(ids)
    harm_upper = clopper_pearson_upper(absolute_only, len(ids), alpha)
    finite_pass = difference > -margin
    conservative_pass = harm_upper < margin
    return {
        "n": len(ids),
        "both_success": both_success,
        "both_failure": both_failure,
        "absolute_only": absolute_only,
        "treatment_only": treatment_only,
        "treatment_minus_absolute": difference,
        "margin": margin,
        "alpha_one_sided": alpha,
        "absolute_only_rate_cp_upper": harm_upper,
        "finite_benchmark_noninferior": finite_pass,
        "conservative_confirmatory_noninferior": conservative_pass,
        "pass": finite_pass and conservative_pass,
    }


def _oracle_output(contract: Contract, semantic: Semantic, cell: Cell) -> str:
    coord = contract.ideal_coord(semantic, cell.cursor, cell.target)
    if semantic == "deltatype_raw":
        return f"{coord[0]} {coord[1]} 0 ; +LMB -LMB"
    action = "left_click" if semantic == "absolute_toolcall" else "move_rel"
    return (
        '<tool_call>\n{"name": "computer_use", "arguments": '
        f'{{"action": "{action}", "coordinate": [{coord[0]}, {coord[1]}]}}}}\n'
        "</tool_call>"
    )


def run_selftest(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_protocol(protocol_path, require_launch_authorized=None)
    checked_arms = validate_protocol(protocol)
    contract = Contract()
    cells = load_cells(protocol, contract)
    pixel_hashes = {rgb_sha256(cell.image_path.read_bytes()) for cell in cells}
    if len(pixel_hashes) != protocol["geometry"]["unique_cursor_bbox_pairs"]:
        raise GateError("decoded frozen cell image cardinality drift")
    oracle_counts: dict[str, dict[str, int]] = {}
    for arm_name, arm in protocol["arms"].items():
        semantic: Semantic = arm["semantic"]
        counts = {"parse": 0, "schema": 0, "unit": 0, "click": 0, "drag": 0}
        for cell in cells:
            raw = _oracle_output(contract, semantic, cell)
            move = contract.parse(semantic, raw)
            if not move.parse_ok or move.coord is None:
                raise GateError(f"{arm_name}: oracle parse failed")
            counts["parse"] += 1
            if not strict_schema_ok(semantic, raw, move.coord):
                raise GateError(f"{arm_name}: oracle schema failed")
            counts["schema"] += 1
            if not unit_range_ok(semantic, move.coord):
                raise GateError(f"{arm_name}: oracle unit failed")
            counts["unit"] += 1
            endpoint = contract.apply_coord(semantic, cell.cursor, move.coord)
            if not contract.in_bbox(endpoint, cell.bbox):
                raise GateError(f"{arm_name}: oracle endpoint missed")
            click = replay_plan(cell, actuation_plan(semantic, "click", cell.cursor, endpoint), "click")
            drag = replay_plan(cell, actuation_plan(semantic, "drag", cell.cursor, endpoint), "drag")
            if click.cursor != endpoint or drag.cursor != endpoint:
                raise GateError(f"{arm_name}: replay endpoint mismatch")
            counts["click"] += 1
            counts["drag"] += 1
        oracle_counts[arm_name] = counts
    n = len(cells)
    all_success = {cell.cell_id: True for cell in cells}
    zero_harm = paired_noninferiority(all_success, all_success)
    if not zero_harm["pass"]:
        raise GateError("zero-harm NI selftest failed")
    maximum_harms = maximum_confirmatory_harms(n, 0.05, 0.05)
    failing = dict(all_success)
    for cell in cells[: maximum_harms + 1]:
        failing[cell.cell_id] = False
    fail_result = paired_noninferiority(all_success, failing)
    if fail_result["pass"]:
        raise GateError("NI boundary selftest did not fail closed")
    return {
        "schema_version": 1,
        "status": "pass",
        "protocol_sha256": sha256_file(protocol_path),
        "launch_authorized": protocol["launch_gate"]["authorized"],
        "scope_classification": protocol["scope_classification"],
        "arms": checked_arms,
        "cells": len(cells),
        "episodes": len({cell.episode_id for cell in cells}),
        "unique_cursor_bbox_pairs": len({(cell.cursor, cell.bbox) for cell in cells}),
        "unique_decoded_images": len(pixel_hashes),
        "unique_request_seed_pairs": len({(cell.cell_id, cell.request_seed) for cell in cells}),
        "oracle_replays": oracle_counts,
        "actuation_replays_tested": len(cells) * len(checked_arms) * 2,
        "noninferiority": {
            "margin": 0.05,
            "alpha_one_sided": 0.05,
            "zero_harm_upper": zero_harm["absolute_only_rate_cp_upper"],
            "maximum_confirmatory_absolute_only_harms_at_n320": maximum_harms,
            "first_failing_harm_count": maximum_harms + 1,
        },
        "resource_estimate": protocol["resource_estimate"],
        "remaining_launch_gates": (
            [] if protocol["launch_gate"]["authorized"] else protocol["launch_gate"]["required_before_launch"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_selftest(args.protocol)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
