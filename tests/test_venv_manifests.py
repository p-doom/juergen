import re
from pathlib import Path

VENVS = Path(__file__).resolve().parents[1] / "tooling" / "venvs"


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
