import os
import re
import subprocess
from pathlib import Path

import pytest

VENVS = Path(__file__).resolve().parents[1] / "tooling" / "venvs"
SOURCE_COMMIT = "119b2e252b9a91ee4e15124b720daccfd1c9789b"
SOURCE_REF = "refs/remotes/origin/archive/sign-of-life-eval-v2-20260803"


def test_active_serving_venvs_have_one_manifest_and_one_rebuild_script() -> None:
    documented = set(
        re.findall(r"^\| `([^`]+)` \|", (VENVS / "README.md").read_text(), re.MULTILINE)
    )
    assert "sign_of_life_eval_v2_venv" in documented

    manifests = {
        path.name.removesuffix(".requirements.txt")
        for path in VENVS.glob("*.requirements.txt")
    }
    rebuilds = {
        path.name.removeprefix("rebuild-").removesuffix(".sh")
        for path in VENVS.glob("rebuild-*.sh")
    }
    assert manifests == documented
    assert rebuilds == documented


def _run_sign_of_life_rebuild(
    tmp_path: Path, *, source_ref_target: str | None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = VENVS.parents[1]
    source = tmp_path / "source"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", root, source],
        check=True,
    )
    subprocess.run(["git", "-C", source, "update-ref", "-d", SOURCE_REF], check=True)
    if source_ref_target is not None:
        subprocess.run(
            ["git", "-C", source, "update-ref", SOURCE_REF, source_ref_target],
            check=True,
        )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_called = tmp_path / "uv-called"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/sh\ntouch "$UV_CALLED"\nexit 97\n')
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["UV_CALLED"] = str(uv_called)
    result = subprocess.run(
        [
            VENVS / "rebuild-sign_of_life_eval_v2_venv.sh",
            tmp_path / "target",
            source,
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return result, uv_called


@pytest.mark.parametrize(
    ("source_ref_target", "error"),
    [
        (None, "does not contain fetched"),
        ("8be2879c656039168273dc81e1a87df842cbb4ff", "resolves to"),
    ],
)
def test_sign_of_life_rebuild_rejects_missing_or_mismatched_archival_ref(
    tmp_path: Path, source_ref_target: str | None, error: str
) -> None:
    result, uv_called = _run_sign_of_life_rebuild(
        tmp_path, source_ref_target=source_ref_target
    )

    assert result.returncode == 2
    assert error in result.stderr
    assert not uv_called.exists()


def test_sign_of_life_rebuild_accepts_exact_archival_ref(tmp_path: Path) -> None:
    result, uv_called = _run_sign_of_life_rebuild(
        tmp_path, source_ref_target=SOURCE_COMMIT
    )

    assert result.returncode == 97
    assert uv_called.exists()
