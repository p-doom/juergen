"""``datasets/convert.py``: the manifest must identify the prompt it wrote.

The eval harness compares its ``system_prompt_sha256`` against the prompt it
renders from the codec. A converter that records only ``codec_digest`` mislabels
every ``--keep_prose`` dataset, because those records carry
``THINKING_PREAMBLE + describe()`` and the bare ``describe()`` digest identifies a
prompt no record contains.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from juergen_doubles import load_convert  # noqa: E402

convert = load_convert()

SCREEN = [1920, 1080]


def _rollout(root: Path, slug: str, n_steps: int = 4) -> None:
    """One rollout dir in the shape ``discover_run_dirs`` looks for."""
    run = root / slug
    (run / "steps").mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "instruction": "open the terminal",
                "screen_size": SCREEN,
                "stop_reason": "terminate",
                "task_success": 1.0,
            }
        )
    )
    rows = []
    for step in range(1, n_steps + 1):
        # Only existence is checked, so an empty file is a frame.
        (run / "steps" / f"step_{step - 1:03d}.png").write_bytes(b"")
        rows.append(
            {
                "step_num": step,
                "action": f"<think>step {step}</think>\n"
                '<tool_call>\n{"name": "computer_use", "arguments": '
                '{"action": "left_click", "coordinate": [100, 200]}}\n</tool_call>',
                "info": {
                    "parsed": {
                        "computer_use": {
                            "action": "left_click",
                            "coordinate": [100 + step, 200],
                        }
                    },
                    "cursor_before": [10, 10],
                    "intended_target": [100 + step, 200],
                },
            }
        )
    (run / "trajectory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def _run(tmp_path: Path, *extra: str) -> tuple[dict, list[dict]]:
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir(parents=True)
    for slug in ("task_a", "task_b", "task_c", "task_d"):
        _rollout(rollouts, slug)
    out = tmp_path / "out"
    assert (
        convert.main(
            [
                "--rollouts_dir", str(rollouts),
                "--out_dir", str(out),
                "--codec", "deltatype_v2",
                "--prose_divergence", "off",
                *extra,
            ]
        )
        == 0
    )
    manifest = json.loads((out / "convert_manifest.json").read_text())
    records = [
        json.loads(line)
        for split in ("train", "val")
        for line in (out / "_normalized" / split / "chat.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert records, "fixture produced no records"
    return manifest, records


def _prompt_of(record: dict) -> str:
    system = record["messages"][0]
    assert system["role"] == "system"
    return system["content"][0]["text"]


@pytest.mark.parametrize("prose", [True, False])
def test_manifest_digest_is_the_prompt_in_the_records(tmp_path, prose):
    manifest, records = _run(tmp_path, *([] if prose else ["--no_keep_prose"]))
    prompts = {_prompt_of(record) for record in records}
    assert len(prompts) == 1
    written = hashlib.sha256(prompts.pop().encode()).hexdigest()
    assert manifest["system_prompt_sha256"] == written


def test_the_two_prose_modes_are_not_the_same_prompt(tmp_path):
    """So a single recorded digest cannot cover both."""
    with_prose, _ = _run(tmp_path / "a")
    without, _ = _run(tmp_path / "b", "--no_keep_prose")
    assert with_prose["system_prompt_sha256"] != without["system_prompt_sha256"]
    # `codec_digest` identifies the grammar revision and is the same either way,
    # which is exactly why it cannot stand in for the prompt digest.
    assert with_prose["codec_digest"] == without["codec_digest"]
    assert without["system_prompt_sha256"] == without["codec_digest"]
