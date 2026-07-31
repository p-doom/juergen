"""Frozen constants for the preregistered coarse official pilot."""

from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final = 1
CONTRACT_ID: Final = "proper-vm-roadmap-3.5-official-coarse-pilot-v1"
GATE_SCOPE: Final = "proper-vm-official-coarse-pilot-v1"
SOURCE_PROTOCOL: Final = "official-heldout-broker-v1"
SELECTION_POLICY: Final = "broker-sealed-stratified-without-replacement-v1"
RESET_PROTOCOL: Final = "fresh-osworld-ready-snapshot-v1"

NATIVE_ABSOLUTE_ARM: Final = "native_absolute_control"
COMPACT_RAW_ARM: Final = "compact_raw_phaseb"
ARMS: Final = (NATIVE_ABSOLUTE_ARM, COMPACT_RAW_ARM)

# These are paired rollout/setup seeds, not task-selection seeds.  The sealed
# source broker performs task selection only after authorization.
PAIRED_SEEDS: Final = (3501, 3511)
PILOT_TASK_COUNT: Final = 8
EXPECTED_PAIR_COUNT: Final = PILOT_TASK_COUNT * len(PAIRED_SEEDS)
EXPECTED_EPISODE_COUNT: Final = EXPECTED_PAIR_COUNT * len(ARMS)

BOOTSTRAP_SAMPLES: Final = 10_000
BOOTSTRAP_SEED: Final = 20_260_731
CI_LEVEL: Final = 0.95
NONINFERIORITY_MARGIN: Final = -0.15
COMPACT_RAW_SUCCESS_FLOOR: Final = 0.60
COMPACT_RAW_PARSE_EXECUTOR_FAILURE_CEILING: Final = 0.05

REQUIRED_PREREQUISITE_RUNGS: Final = ("3.1", "3.2", "3.3", "3.4")
SIGNATURE_NAMESPACE: Final = "juergen-proper-vm-release-gate-v1"
