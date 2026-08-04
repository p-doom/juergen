"""The model-facing contract is frozen: assert the sealed manifest's hashes.

``dataset_manifest.json`` of the sealed Phase-B dataset records the SHA-256 of
the five implementation files that produced it, plus four contract modules. The
s900 checkpoint's behaviour depends on those exact bytes, so this test is the
tripwire: if anyone edits the codec, the prompt, the converter or the builder,
this fails and the edit has to be justified against the checkpoint.

Files pinned here are committed **verbatim**. Do not reformat them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import ROOT, repo_relative
from prompt import SYSTEM_PROMPT

# manifest.implementation_sha256 of
# phaseb_raw_deltatype_v2_build_audit_v1_run_019fb5a5564e7a71b3ad6e55426af463
IMPLEMENTATION_SHA256 = {
    "action_v2.py": "1ded3d5a7e51da71cf3082049fbdd404971ebf72a95d93f333ebb3ee3075ccb7",
    "build.py": "c3562ebe5fc2661f62a4957ad6714f2abf2f66ca1cc8218f447f6614c27e7c99",
    "converter.py": "7338b12a43250048e07cadfb8c20fa6530ede29f1cab8e857144f3efcce5070e",
    "prompt.py": "c6c32ea22d0f9c06bf3e2c2d852b88e36d7341ee764e5aa11c37c1e06c798072",
    "readiness.py": "4672752d869774b57a85e9242fd85b0d9f948572f5eae7f1d98ac98a341f4408",
}

# manifest.contract_file_sha256 — modules loaded by path at build time. Two of
# them are in-repo after this branch (the production parser and the audited
# action-span converter, vendored); the other two still live outside the repo.
IN_REPO_CONTRACT_SHA256 = {
    "eval/action_parser.py": (
        "f916757d17e4a5f53627510616ffff411e9109e8737d1309067c6338caae4a9a"
    ),
    "osworld_parity/phaseb_deltatype_raw_v2/vendor/action_span_conversion.py": (
        "65397c1dcebdd95431bb53918c0117131f24dfc3cd06c5390e4b321202c84497"
    ),
}

SYSTEM_PROMPT_SHA256 = (
    "57f7d0b230974068618b48151b73215d5517d5445a99dbf5abdc05557e3482e6"
)

# manifest.split_file_sha256
SPLIT_SHA256 = {
    "osworld_parity/split/osworld_train.json": (
        "1a5cb5bf8f27079b50188a3735f5bb7f801b5fd812551d7f328281e6399700ae"
    ),
    "osworld_parity/split/osworld_eval_heldout.json": (
        "9bdb3e466738c06d3f372d7ae4ebadb4d4b575175871cb63af0f4c89f8ba7e7c"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("name", "digest"), sorted(IMPLEMENTATION_SHA256.items()))
def test_implementation_files_are_byte_frozen(name: str, digest: str):
    assert sha256(ROOT / name) == digest


@pytest.mark.parametrize(
    ("relative", "digest"), sorted(IN_REPO_CONTRACT_SHA256.items())
)
def test_in_repo_contract_modules_are_byte_frozen(relative: str, digest: str):
    assert sha256(repo_relative(relative)) == digest


@pytest.mark.parametrize(("relative", "digest"), sorted(SPLIT_SHA256.items()))
def test_task_splits_are_byte_frozen(relative: str, digest: str):
    assert sha256(repo_relative(relative)) == digest


def test_system_prompt_digest_matches_the_sealed_dataset():
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == SYSTEM_PROMPT_SHA256
