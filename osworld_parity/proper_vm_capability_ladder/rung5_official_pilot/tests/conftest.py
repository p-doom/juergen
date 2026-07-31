from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.contract import (
    ARMS,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING,
    COMPACT_RAW_SUCCESS_FLOOR,
    CONTRACT_ID,
    EXPECTED_EPISODE_COUNT,
    GATE_SCOPE,
    NONINFERIORITY_MARGIN,
    PAIRED_SEEDS,
    PILOT_TASK_COUNT,
    REQUIRED_PREREQUISITE_RUNGS,
    SELECTION_POLICY,
    SIGNATURE_NAMESPACE,
    SOURCE_PROTOCOL,
)
from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.gates import (
    GateBundle,
    SignedGatePaths,
    canonical_json,
)


NOW = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)


@dataclass
class SignedBundleFixture:
    bundle: GateBundle
    private_key: Path
    prerequisites_payload: dict[str, Any]
    release_payload: dict[str, Any]

    def sign(self, path: Path, payload: dict[str, Any]) -> Path:
        path.write_bytes(canonical_json(payload))
        signature = Path(str(path) + ".sig")
        signature.unlink(missing_ok=True)
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(self.private_key),
                "-n",
                SIGNATURE_NAMESPACE,
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return signature


@pytest.fixture
def signed_bundle(tmp_path: Path) -> SignedBundleFixture:
    private_key = tmp_path / "release_key"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "rung5-test-only",
            "-f",
            str(private_key),
        ],
        check=True,
    )
    public_fields = private_key.with_suffix(".pub").read_text(encoding="utf-8").split()
    allowed_signers = tmp_path / "allowed_signers"
    allowed_signers.write_text(
        f"proper-vm-roadmap-release {public_fields[0]} {public_fields[1]}\n",
        encoding="utf-8",
    )
    prerequisites = {
        "schema_version": 1,
        "kind": "proper-vm-prerequisites",
        "gate_id": "gate-rungs-3-1-through-3-4",
        "scope": GATE_SCOPE,
        "decision": "pass",
        "contract_id": CONTRACT_ID,
        "rungs": {rung: "pass" for rung in REQUIRED_PREREQUISITE_RUNGS},
        "issued_at": "2026-07-31T09:00:00Z",
        "expires_at": "2026-08-01T12:00:00Z",
    }
    release = {
        "schema_version": 1,
        "kind": "official-pilot-release",
        "gate_id": "gate-official-pilot-release-v1",
        "parent_gate_id": prerequisites["gate_id"],
        "pilot_id": "pilot-mock-rung5-v1",
        "scope": GATE_SCOPE,
        "decision": "authorize",
        "contract_id": CONTRACT_ID,
        "source_protocol": SOURCE_PROTOCOL,
        "selection_policy": SELECTION_POLICY,
        "task_count": PILOT_TASK_COUNT,
        "paired_seeds": list(PAIRED_SEEDS),
        "arms": list(ARMS),
        "max_episodes": EXPECTED_EPISODE_COUNT,
        "analysis": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "compact_raw_success_floor": COMPACT_RAW_SUCCESS_FLOOR,
            "compact_raw_parse_executor_failure_ceiling": (
                COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING
            ),
        },
        "issued_at": "2026-07-31T10:00:00Z",
        "expires_at": "2026-08-01T12:00:00Z",
    }
    prerequisites_path = tmp_path / "prerequisites.json"
    release_path = tmp_path / "pilot_release.json"
    helper = SignedBundleFixture(
        bundle=GateBundle(
            prerequisites=SignedGatePaths(
                prerequisites_path, Path(str(prerequisites_path) + ".sig")
            ),
            pilot_release=SignedGatePaths(
                release_path, Path(str(release_path) + ".sig")
            ),
            allowed_signers=allowed_signers,
            signer_identity="proper-vm-roadmap-release",
        ),
        private_key=private_key,
        prerequisites_payload=prerequisites,
        release_payload=release,
    )
    helper.sign(prerequisites_path, prerequisites)
    helper.sign(release_path, release)
    return helper


def mock_rows(
    *, compact_parse_failure: bool = False, infrastructure_failure: bool = False
) -> list[dict[str, Any]]:
    from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.contract import (
        COMPACT_RAW_ARM,
        RESET_PROTOCOL,
    )
    from osworld_parity.proper_vm_capability_ladder.rung5_official_pilot.records import (
        expected_arm_order,
        expected_pair_key,
    )

    rows: list[dict[str, Any]] = []
    for cluster_index in range(PILOT_TASK_COUNT):
        for pair_seed in PAIRED_SEEDS:
            arm_order = expected_arm_order(cluster_index, pair_seed)
            for arm in ARMS:
                parse_failure = (
                    compact_parse_failure
                    and arm == COMPACT_RAW_ARM
                    and cluster_index == 0
                    and pair_seed == PAIRED_SEEDS[0]
                )
                step = {
                    "step_index": 0,
                    "parse_success": not parse_failure,
                    "executor_success": not parse_failure,
                    "action_class": "unparsed" if parse_failure else "click",
                    "ineffective": False,
                    "error_code": "parse_error" if parse_failure else None,
                }
                rows.append(
                    {
                        "schema_version": 1,
                        "pilot_id": "pilot-mock-rung5-v1",
                        "source_protocol": SOURCE_PROTOCOL,
                        "cluster_index": cluster_index,
                        "pair_seed": pair_seed,
                        "pair_key": expected_pair_key(cluster_index, pair_seed),
                        "arm": arm,
                        "arm_order": list(arm_order),
                        "reset_protocol": RESET_PROTOCOL,
                        "reset_ordinal": arm_order.index(arm) + 1,
                        "reset_success": not (
                            infrastructure_failure
                            and cluster_index == 0
                            and pair_seed == PAIRED_SEEDS[0]
                            and arm == ARMS[0]
                        ),
                        "setup_success": True,
                        "oracle_evaluated": True,
                        "parse_success": not parse_failure,
                        "executor_success": not parse_failure,
                        "task_success": cluster_index < 6,
                        "termination": "parse_error" if parse_failure else "model_terminate",
                        "steps": [step],
                    }
                )
    return rows
