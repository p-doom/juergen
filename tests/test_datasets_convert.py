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


def _rollout(root: Path, slug: str, n_steps: int = 4, **result_extra) -> None:
    """One rollout dir in the shape ``discover_run_dirs`` looks for.

    An external-teacher collection: no ``codec``, so ``source_reader`` picks
    ``teacher_step`` and the steps below carry ``computer_use`` arguments.
    """
    run = root / slug
    (run / "steps").mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "instruction": "open the terminal",
                "screen_size": SCREEN,
                "stop_reason": "terminate",
                **result_extra,
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


def _harness_rollout(root: Path, slug: str, n_steps: int = 4, **result_extra) -> None:
    """One of OUR OWN rollouts, in the shape ``evals/harness.py`` writes.

    It declares a ``codec`` and spells the end of the episode ``outcome``, and its
    steps carry the dispatched Operation stream rather than ``computer_use``
    arguments — the two halves of the source vocabulary ``source_reader`` and
    ``source_stop_reason`` key on. Verified field-for-field against an artifact
    produced by running the real harness.
    """
    run = root / slug
    (run / "steps").mkdir(parents=True)
    (run / "result.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "codec": "deltatype_v2",
                "instruction": "open the terminal",
                "screen_size": SCREEN,
                "outcome": "max_steps",
                **result_extra,
            }
        )
    )
    rows = []
    for step in range(1, n_steps + 1):
        (run / "steps" / f"step_{step - 1:03d}.png").write_bytes(b"")
        rows.append(
            {
                "step_num": step,
                "action": f"<think>step {step}</think>\n0 0 0 ;",
                "info": {
                    "operations": [{"kind": "move_to", "args": [100 + step, 200]}],
                    # Present because the harness always writes it, and read by
                    # nothing on this path: `rollout_step` reads `operations`.
                    "parsed": {
                        "dx": 90 + step,
                        "dy": 190,
                        "scroll": 0,
                        "elements": [],
                        "no_op": False,
                        "terminate": False,
                        "fail": False,
                    },
                    "parse_error": None,
                    "cursor_before": [10, 10],
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


def test_manifest_records_the_image_encoding_in_the_records(tmp_path):
    """The image half of the same claim. The eval harness refuses to score a
    checkpoint through a different encoding, so the manifest has to describe the
    frames these records actually point at -- lossless full-resolution PNG, since
    the harness framebuffers are referenced by path and never re-encoded."""
    from agent.history import ImageBudget

    manifest, records = _run(tmp_path)
    frames = [
        part["image"]
        for record in records
        for message in record["messages"]
        for part in message["content"]
        if part.get("type") == "image"
    ]
    assert frames and all(frame.endswith(".png") for frame in frames)
    assert manifest["image_domain"] == ImageBudget(media="png").domain
    assert manifest["image_domain"] != ImageBudget().domain, (
        "the eval default is JPEG q85; if these ever agree the gate is asleep"
    )


def test_the_two_prose_modes_are_not_the_same_prompt(tmp_path):
    """So a single recorded digest cannot cover both."""
    with_prose, _ = _run(tmp_path / "a")
    without, _ = _run(tmp_path / "b", "--no_keep_prose")
    assert with_prose["system_prompt_sha256"] != without["system_prompt_sha256"]
    # `codec_digest` identifies the grammar revision and is the same either way,
    # which is exactly why it cannot stand in for the prompt digest.
    assert with_prose["codec_digest"] == without["codec_digest"]
    assert without["system_prompt_sha256"] == without["codec_digest"]


def _convert(rollouts: Path, out: Path, *extra: str) -> int:
    return convert.main(
        [
            "--rollouts_dir", str(rollouts),
            "--out_dir", str(out),
            "--codec", "deltatype_v2",
            "--prose_divergence", "off",
            *extra,
        ]
    )


def _records(out: Path) -> list[dict]:
    return [
        json.loads(line)
        for split in ("train", "val")
        for line in (out / "_normalized" / split / "chat.jsonl").read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    "reward",
    [
        pytest.param({}, id="no_task_reward_at_all"),
        pytest.param({"task_reward": None}, id="task_reward_present_but_null"),
    ],
)
def test_min_task_success_refuses_a_collection_it_cannot_score(tmp_path, reward):
    """The flag may not empty the dataset and exit as though it filtered.

    Both spellings of "unscored" are here because they are what the two producers
    actually write. The external teacher collections carry no ``task_reward`` key,
    while our own harness writes it unconditionally and leaves it null unless
    ``evaluate_on_finish`` was set — so a key-presence check passes on every one of
    our rollouts and drops them all. The refusal must name the flag: this trips the
    generic empty-dataset guard too, and that one blames the rollout layout.
    """
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir(parents=True)
    for slug in ("task_a", "task_b"):
        _harness_rollout(rollouts, slug, **reward)
    with pytest.raises(SystemExit, match="--min_task_success"):
        _convert(rollouts, tmp_path / "out", "--min_task_success", "1.0")


def test_min_task_success_filters_on_the_score_the_harness_publishes(tmp_path):
    """`task_reward` is the field, and the threshold is a real comparison.

    A rollout the scorer never reached is dropped rather than fatal — one unscored
    rollout in a scored collection cannot be filtered by a score, but it is not a
    reason to abandon the build.
    """
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir(parents=True)
    _harness_rollout(rollouts, "keep_me", task_reward=1.0)
    _harness_rollout(rollouts, "below_bar", task_reward=0.0)
    _harness_rollout(rollouts, "never_scored", task_reward=None)
    out = tmp_path / "out"
    assert _convert(rollouts, out, "--min_task_success", "1.0") == 0
    manifest = json.loads((out / "convert_manifest.json").read_text())
    assert manifest["n_kept"] == 1
    # Two of the three carried a score; only one cleared the bar.
    assert manifest["n_scored"] == 2
    assert [r["recording_id"] for r in _records(out)] == ["keep_me"]


def test_the_stop_reason_is_read_in_the_producers_own_spelling(tmp_path):
    """Our harness writes `outcome`, the teacher collections write `stop_reason`.

    Reading only one left the field null for every rollout of the other producer.
    """
    ours = tmp_path / "ours"
    ours.mkdir(parents=True)
    _harness_rollout(ours, "mine")
    out_ours = tmp_path / "out_ours"
    assert _convert(ours, out_ours) == 0
    assert [r["source_stop_reason"] for r in _records(out_ours)] == ["max_steps"]

    theirs = tmp_path / "theirs"
    theirs.mkdir(parents=True)
    _rollout(theirs, "yours")
    out_theirs = tmp_path / "out_theirs"
    assert _convert(theirs, out_theirs) == 0
    assert [r["source_stop_reason"] for r in _records(out_theirs)] == ["terminate"]


def _drop_parsed(run: Path, step_num: int | None = None) -> None:
    """Remove ``info["parsed"]`` from one row, or from every row.

    It is the teacher payload. Our own rollouts carry it and this path never reads
    it; a teacher row without it has nothing to read at all.
    """
    path = run / "trajectory.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if step_num in (None, row["step_num"]):
            row["info"].pop("parsed", None)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_our_own_rollout_is_read_from_its_operations_not_its_parsed_action(tmp_path):
    """`rollout_step` reads `operations`, so nothing above it may gate on `parsed`.

    `parsed` is the SOURCE grammar's own action dict — seven vocabularies, none of
    them this seam. Requiring it discarded one of our own rollout turns as a parse
    error with the dispatched Operation stream sitting in the same row.
    """
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir(parents=True)
    _harness_rollout(rollouts, "mine")
    _drop_parsed(rollouts / "mine")
    out = tmp_path / "out"
    assert _convert(rollouts, out) == 0
    manifest = json.loads((out / "convert_manifest.json").read_text())
    assert manifest["n_turns"] == 4
    assert manifest["n_turns_teacher_parse_error"] == 0


def test_a_teacher_row_with_no_payload_is_a_counted_parse_error_not_a_crash(tmp_path):
    """`teacher_step` reads the payload, so it is the one that requires it."""
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir(parents=True)
    _rollout(rollouts, "theirs")
    _drop_parsed(rollouts / "theirs", 1)
    out = tmp_path / "out"
    assert _convert(rollouts, out) == 0
    manifest = json.loads((out / "convert_manifest.json").read_text())
    assert manifest["n_turns"] == 3
    assert manifest["n_turns_teacher_parse_error"] == 1
