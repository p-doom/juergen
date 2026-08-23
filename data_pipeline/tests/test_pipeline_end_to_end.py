"""Stages 00 -> 04 driven over one synthetic segment, end to end.

Every stage below stage 03 had import-probe coverage only
(``test_stage_dependencies.py``), so nothing measured what the chain *produces*.
The three defect classes this pins:

  * **orphaned key releases in training conversations.** Assistant turns carry
    key/button transitions, and the label of a turn is a slice of a
    segment-global stream. A release whose press is not in the same conversation
    trains the model to lift a key it never pressed, and the held-set of every
    later turn is wrong from there on. The invariant is one-sided: a press that
    is never released is a real observed transition (the demonstrator was still
    holding the key when the recording ended) and is counted as
    ``n_held_at_end``; a release without a press is not.
  * **black-frame spans corrupting action attribution.** The pixels of a black
    span are not seen by the trainee, so the input made during it must not be
    folded into a visible frame's label. Asserted differentially: the same
    fixture filtered with black masking off attributes the span's move and its
    key transitions to a visible turn, and manufactures an orphaned press on
    the way.
  * **cross-stage contract fields.** Each artifact records the id of the one it
    was built from (``master_store_id`` / ``filter_id``) and the digest of the
    system prompt its labels were rendered against; a join that drifts must
    fail rather than silently mix coordinate systems.

Stage 01's ffmpeg decode is not exercised here — see ``synthetic_clip``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grammars
import pytest
import synthetic_clip as clip
from grammars.deltatype_v2 import CODEC as DELTATYPE_V2
from grammars.ordered_events_v3 import CODEC as ORDERED_EVENTS_V3

from pipeline.lib import config
from pipeline.lib.manifest import make_artifact_id

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PIPELINE_DIR = REPO_ROOT / "data_pipeline"
STAGES = REPO_ROOT / "pipeline"

TERMINAL_TOKEN = f"{grammars.CONTROL_TOKEN}: success"

# The label of each window that carries events, under ``canonical``. Windows
# past W8 are NO_OP (the clip's second half is inactive on purpose).
EXPECTED_CANONICAL_LABELS = [
    "9 3 0",                                        # W0  two moves summed
    "0 0 0 ; +LMB -LMB",                            # W1  a click inside one window
    "2 -1 0 ; +KeyA",                               # W2  autorepeat deduped
    "0 0 0 ; -KeyA +KeyR -KeyR",                    # W3  carried release, clamped release
    "-3 2 0 ; +KeyB -KeyB",                         # W4  press clamped out of the black span
    "0 0 -3",                                       # W5  dangling release absent
    "0 0 0 ; +ShiftLeft +KeyH -KeyH -ShiftLeft +KeyI -KeyI",  # W6
    "1 1 0",                                        # W7
    "0 0 0 ; +KeyC",                                # W8  held at end
]

EXPECTED_V3_LABELS = [
    "move(5,0); move(4,3)",
    "down(LMB); up(LMB)",
    "down(KeyA); move(2,-1)",
    'up(KeyA); type("r")',
    'type("b"); move(-3,2)',
    "scroll(0,-3)",
    f'type("{clip.TYPED_TEXT}")',
    "move(1,1)",
    "down(KeyC)",
]

EXPECTED_DEAD_ZONE_COUNTERS = {
    "n_discarded_black": 1,          # the move at t=4.20
    "n_discarded_no_coverage": 1,    # the move past the end of the video
    "n_discarded_pre_first_frame": 0,
    "n_pairs_dropped_dead_zone": 1,  # KeyM, both endpoints inside the span
    "n_unreleased_press_dropped": 1,  # KeyQ, pressed inside the span, never released
    "n_releases_clamped": 1,         # KeyR
    "n_presses_clamped": 1,          # KeyB
    "n_dangling_release": 1,         # KeyZ
    "n_redundant_press": 1,          # the KeyA autorepeat
    "n_held_at_end": 2,              # KeyQ (dropped) and KeyC (emitted)
    "max_simultaneous_keys": 3,
}


def _env() -> dict[str, str]:
    """Stage subprocesses resolve ``pipeline`` and ``grammars`` from the repo
    root and ``desktop`` from the sibling checkout — the same three roots
    ``conftest.py`` puts on this process's path."""
    roots = [REPO_ROOT, DATA_PIPELINE_DIR, REPO_ROOT.parent / "desktop"]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(str(r) for r in roots))


def _run_stage(script: str, *args: object) -> None:
    proc = subprocess.run(
        [sys.executable, str(STAGES / script), *[str(a) for a in args]],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{script} exited {proc.returncode}\n--- stdout ---\n{proc.stdout}"
            f"\n--- stderr ---\n{proc.stderr}"
        )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _filter(master: Path, clips_manifest: Path, out_dir: Path, *extra: object) -> Path:
    _run_stage(
        "stage_03_filter.py",
        "--frames-master-dir", master,
        "--clips-manifest", clips_manifest,
        "--output-dir", out_dir,
        "--num-workers", 1,
        *extra,
    )
    return out_dir


#: Filter knobs that leave black masking as the ONLY mask, so the conversation
#: assertions below are about attribution and not about idle thinning.
_NO_IDLE = (
    "--idle-activity", "raw",
    "--idle-judgment-bin-s", 0,
    "--idle-min-duration-s", 100_000,
    "--idle-keep-head-s", 0,
    "--idle-keep-tail-s", 0,
)


def _conversations(filter_dir: Path, out_dir: Path, action_format: str, *extra: object) -> Path:
    _run_stage(
        "stage_04_build_conversations.py",
        "--filter-dir", filter_dir,
        "--fps", clip.TRAIN_FPS,
        "--output-dir", out_dir,
        "--action-format", action_format,
        "--num-workers", 1,
        *extra,
    )
    return out_dir


class Chain:
    """Every artifact one run of the chain produced, keyed by stage."""

    def __init__(self, root: Path):
        self.root = root
        self.source = clip.build_uploads_tree(root / "uploads_tree")

        self.stage00 = root / "s00" / "clips_manifest.jsonl"
        _run_stage(
            "stage_00_clip_manifest.py",
            "--dataset-root", root / "uploads_tree",
            "--out", self.stage00,
            "--workers", 1,
        )
        self.clip_rows = _jsonl(self.stage00)

        self.realign_dir = root / "s02"
        _run_stage(
            "stage_02_realign.py",
            "--clips-manifest", self.stage00,
            "--output-dir", self.realign_dir,
            "--num-workers", 1,
        )
        self.realigned_manifest = self.realign_dir / "clips_manifest.jsonl"

        self.master = clip.build_master_store(
            root / "s01", self.clip_rows[0], self.source["frames"]
        )

        self.filter_dir = _filter(
            self.master, self.realigned_manifest, root / "s03", *_NO_IDLE
        )
        self.filter_black_off = _filter(
            self.master, self.realigned_manifest, root / "s03_black_off",
            "--drop-black-frames", "false", *_NO_IDLE,
        )
        self.filter_default = _filter(
            self.master, self.realigned_manifest, root / "s03_default"
        )

        self.conv_canonical = _conversations(
            self.filter_dir, root / "s04_canonical", "canonical",
            "--instruction", "do the synthetic thing",
        )
        self.conv_black_off = _conversations(
            self.filter_black_off, root / "s04_black_off", "canonical",
            "--instruction", "do the synthetic thing",
        )
        self.conv_v3 = _conversations(
            self.filter_dir, root / "s04_v3", "ordered_events_v3",
            "--instruction", "do the synthetic thing",
            "--terminal-token", TERMINAL_TOKEN,
        )

        self.goals_dir = clip.write_goals(
            root / "goals",
            [
                clip.goal_row("g_head", 0, 12, "open the synthetic thing"),
                clip.goal_row("g_mid", 13, 40, "finish the synthetic thing"),
            ],
            master_store_id=make_artifact_id(self.master),
            filter_id=make_artifact_id(self.filter_dir),
        )
        self.conv_goals = _conversations(
            self.filter_dir, root / "s04_goals", "canonical", "--goals-dir", self.goals_dir
        )

    def segment_filter(self, filter_dir: Path) -> dict[str, Any]:
        return json.loads((filter_dir / "filter" / f"{clip.SEGMENT_ID}.json").read_text())

    def rows(self, conv_dir: Path) -> list[dict[str, Any]]:
        return _jsonl(conv_dir / "conversations.jsonl")

    def summary(self, conv_dir: Path) -> dict[str, Any]:
        return json.loads((conv_dir / "conversations_summary.json").read_text())


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> Chain:
    return Chain(tmp_path_factory.mktemp("crowdcast_chain"))


# --------------------------------------------------------------------------
# Reading a built conversation the way training reads it.
# --------------------------------------------------------------------------


def assistant_texts(row: dict[str, Any]) -> list[str]:
    return [
        m["content"][0]["text"] for m in row["messages"] if m["role"] == "assistant"
    ]


def user_images(row: dict[str, Any]) -> list[str]:
    return [
        block["image"]
        for m in row["messages"]
        if m["role"] == "user"
        for block in m["content"]
        if block["type"] == "image"
    ]


def key_transitions(text: str, grammar: str) -> list[tuple[str, str]]:
    """The key/button transitions one assistant turn spells, via the grammar's
    own parser — the same direction the eval harness reads a completion in, so
    this cannot drift from what the emitter wrote."""
    body = grammars.split_control(text).body
    if grammar == "deltatype_v2":
        action = DELTATYPE_V2.parse(body)
        return [
            ("+" if element.pressed else "-", element.name)
            for element in action.elements
            if element.kind == "event"
        ]
    action = ORDERED_EVENTS_V3.parse(body)
    return [
        ("+" if primitive.kind == "down" else "-", primitive.name)
        for primitive in action.primitives
        if primitive.kind in ("down", "up")
    ]


def orphaned_releases(row: dict[str, Any], grammar: str) -> tuple[str, ...]:
    """Names released in this conversation without having been pressed in it."""
    held: set[str] = set()
    orphans: list[str] = []
    for text in assistant_texts(row):
        for sign, name in key_transitions(text, grammar):
            if sign == "+":
                held.add(name)
            elif name in held:
                held.discard(name)
            else:
                orphans.append(name)
    return tuple(orphans)


def unreleased_presses(row: dict[str, Any], grammar: str) -> tuple[str, ...]:
    held: list[str] = []
    for text in assistant_texts(row):
        for sign, name in key_transitions(text, grammar):
            if sign == "+":
                held.append(name)
            elif name in held:
                held.remove(name)
    return tuple(held)


def _all_transitions(rows: Iterable[dict[str, Any]], grammar: str) -> list[tuple[str, str]]:
    return [t for row in rows for text in assistant_texts(row) for t in key_transitions(text, grammar)]


# --------------------------------------------------------------------------
# Stage 00 — discovery + probe
# --------------------------------------------------------------------------


def test_stage_00_probes_the_clip_and_pairs_its_keylog(chain: Chain) -> None:
    (row,) = chain.clip_rows
    assert row["segment_id"] == clip.SEGMENT_ID
    assert row["recording_id"] == clip.RECORDING_ID
    assert row["segment_idx"] == 0
    assert row["user_id"] == clip.USER_ID
    assert row["version"] == clip.VERSION
    assert row["video_ok"] is True
    assert row["video_fps"] == clip.VIDEO_FPS
    assert row["video_frame_count"] == clip.N_FRAMES
    assert row["video_duration_s"] == pytest.approx(clip.N_FRAMES / clip.VIDEO_FPS)
    assert (row["video_width"], row["video_height"]) == (clip.FRAME_W, clip.FRAME_H)
    assert row["keylog_exists"] is True
    assert Path(row["keylog_path"]) == chain.source["keylog_path"].resolve()


def test_stage_00_drops_a_segment_with_no_keylog(tmp_path: Path) -> None:
    source = clip.build_uploads_tree(tmp_path / "tree")
    source["keylog_path"].unlink()
    out = tmp_path / "manifest.jsonl"
    _run_stage(
        "stage_00_clip_manifest.py", "--dataset-root", tmp_path / "tree",
        "--out", out, "--workers", 1,
    )
    assert _jsonl(out) == []
    _run_stage(
        "stage_00_clip_manifest.py", "--dataset-root", tmp_path / "tree",
        "--out", out, "--workers", 1, "--keep-missing-keylog",
    )
    (kept,) = _jsonl(out)
    assert kept["keylog_exists"] is False


# --------------------------------------------------------------------------
# Stage 02 — realignment
# --------------------------------------------------------------------------


def test_stage_02_leaves_a_pause_free_recording_on_its_raw_keylog(chain: Chain) -> None:
    (alignment,) = _jsonl(chain.realign_dir / "alignment.jsonl")
    assert alignment["status"] == "aligned"
    assert alignment["closed"] is True
    assert alignment["n_pauses"] == 0
    assert alignment["splices"] == []
    assert alignment["corrected_keylog_path"] is None
    summary = json.loads((chain.realign_dir / "realign_summary.json").read_text())
    assert summary["n_keylogs_repointed"] == 0
    assert summary["n_corrected"] == 0


def test_stage_02_carries_every_stage_00_field_forward(chain: Chain) -> None:
    (before,) = chain.clip_rows
    (after,) = _jsonl(chain.realigned_manifest)
    assert {k: after[k] for k in before} == before
    assert after["alignment_status"] == "aligned"
    assert after["raw_keylog_path"] == before["keylog_path"]
    assert after["keylog_path"] == before["keylog_path"]  # not repointed


# --------------------------------------------------------------------------
# Stage 01 — the master frame store
# --------------------------------------------------------------------------


def test_stage_01_records_one_frame_per_master_tick_with_luma_metrics(chain: Chain) -> None:
    manifest = _jsonl(chain.master / "frames" / clip.SEGMENT_ID / "frame_manifest.jsonl")
    assert len(manifest) == clip.N_FRAMES
    shard = chain.master / "frames" / clip.SEGMENT_ID / "images.array_record"
    for i, record in enumerate(manifest):
        assert record["record_index"] == i
        assert record["image"] == f"ar://{shard}#{i}"
        assert record["source_time_s"] == pytest.approx(i / clip.MASTER_FPS)
        assert record["source_frame_idx"] == i
        assert len(record["sha256"]) == 64
    black_start, black_end = clip.BLACK_SPAN
    for i, record in enumerate(manifest):
        if black_start <= i < black_end:
            assert record["mean_luma"] == 0.0
            assert record["frac_dark"] == 1.0
        else:
            assert record["mean_luma"] > config.DEFAULT_BLACK_LUMA_MAX
            assert record["frac_dark"] < config.DEFAULT_BLACK_DARK_FRAC_MIN
    summary = json.loads((chain.master / "frames_master_summary.json").read_text())
    assert summary["master_fps"] == clip.MASTER_FPS
    assert summary["n_records_total"] == clip.N_FRAMES


# --------------------------------------------------------------------------
# Stage 03 — the survivor mask
# --------------------------------------------------------------------------


def test_stage_03_masks_exactly_the_black_span(chain: Chain) -> None:
    doc = chain.segment_filter(chain.filter_dir)
    black_start, black_end = clip.BLACK_SPAN
    assert doc["n_master_records"] == clip.N_FRAMES
    assert doc["kept_ranges"] == [[0, black_start], [black_end, clip.N_FRAMES]]
    assert doc["dropped"] == [{"start": black_start, "end": black_end, "reason": "black"}]
    assert doc["n_black"] == black_end - black_start
    assert doc["n_idle_interior"] == 0
    assert doc["n_kept"] == clip.N_FRAMES - (black_end - black_start)
    assert doc["alignment_status"] == "aligned"
    (index_row,) = _jsonl(chain.filter_dir / "filter_index.jsonl")
    assert index_row["status"] == "ok"
    assert index_row["n_kept"] == doc["n_kept"]


def test_stage_03_thins_the_inactive_half_under_the_default_knobs(chain: Chain) -> None:
    # The clip's second half carries no input, so the shipped duration knobs
    # (runs > 4 s thinned, 2 s kept at each end) drop its interior. Black wins
    # over idle where they meet.
    doc = chain.segment_filter(chain.filter_default)
    assert doc["kept_ranges"] == [[0, 16], [20, 48], [72, 80]]
    assert doc["dropped"] == [
        {"start": 16, "end": 20, "reason": "black"},
        {"start": 48, "end": 72, "reason": "idle_interior"},
    ]
    assert doc["n_idle_interior"] == 24
    assert doc["n_black"] == 4


def test_stage_03_fingerprints_the_master_store_it_masked(chain: Chain) -> None:
    manifest = json.loads((chain.filter_dir / "manifest.json").read_text())
    assert manifest["artifact_type"] == "realigned_filter_mask"
    assert manifest["master_store_id"] == make_artifact_id(chain.master)
    assert manifest["master_fps"] == clip.MASTER_FPS


# --------------------------------------------------------------------------
# Stage 04 — conversations
# --------------------------------------------------------------------------


def test_stage_04_emits_one_turn_per_surviving_slot(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_canonical)
    expected_frames = clip.N_FRAMES // clip.STRIDE - 1  # the slot at tick 16 is black
    assert row["n_frames"] == expected_frames
    assert row["n_turns"] == expected_frames
    assert len(assistant_texts(row)) == expected_frames
    assert row["messages"][0]["role"] == "system"
    assert [m["role"] for m in row["messages"][1:]] == ["user", "assistant"] * expected_frames
    first_user = row["messages"][1]["content"]
    assert [b["type"] for b in first_user] == ["text", "image"]
    assert first_user[0]["text"] == "do the synthetic thing"
    assert all(
        [b["type"] for b in m["content"]] == ["image"]
        for m in row["messages"][3::2]
    )


def test_stage_04_points_every_turn_at_a_surviving_master_record(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_canonical)
    shard = chain.master / "frames" / clip.SEGMENT_ID / "images.array_record"
    indices = []
    for uri in user_images(row):
        prefix, _, index = uri.partition("#")
        assert prefix == f"ar://{shard}"
        indices.append(int(index))
    assert indices == sorted(indices)
    assert indices == [t for t in range(0, clip.N_FRAMES, clip.STRIDE) if t != 16]
    black_start, black_end = clip.BLACK_SPAN
    assert not any(black_start <= i < black_end for i in indices)


def test_stage_04_labels_every_event_window(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_canonical)
    texts = assistant_texts(row)
    assert texts[: len(EXPECTED_CANONICAL_LABELS)] == EXPECTED_CANONICAL_LABELS
    assert set(texts[len(EXPECTED_CANONICAL_LABELS) :]) == {"NO_OP"}
    assert row["n_non_noop"] == len(EXPECTED_CANONICAL_LABELS)


def test_stage_04_accounts_for_every_event_the_label_policy_touched(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_canonical)
    assert row["dead_zone_counters"] == EXPECTED_DEAD_ZONE_COUNTERS
    # 5 of 28 events discarded, over the 5% flag threshold.
    assert row["dead_zone_flagged"] is True


def test_stage_04_records_the_join_ids_and_the_prompt_it_trained_against(chain: Chain) -> None:
    manifest = json.loads((chain.conv_canonical / "manifest.json").read_text())
    assert manifest["master_store_id"] == make_artifact_id(chain.master)
    assert manifest["filter_id"] == make_artifact_id(chain.filter_dir)
    assert manifest["grammar"] == "deltatype_v2"
    assert manifest["master_fps"] == clip.MASTER_FPS
    assert manifest["stride"] == clip.STRIDE
    prompt = grammars.describe(manifest["grammar"])
    assert manifest["system_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    (row,) = chain.rows(chain.conv_canonical)
    assert row["messages"][0]["content"][0]["text"] == prompt
    assert (chain.conv_canonical / "chat.jsonl").read_bytes() == (
        chain.conv_canonical / "conversations.jsonl"
    ).read_bytes()


# --------------------------------------------------------------------------
# The orphaned-release class
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attr", "grammar"),
    [("conv_canonical", "deltatype_v2"), ("conv_v3", "ordered_events_v3")],
)
def test_no_release_reaches_a_conversation_without_its_press(
    chain: Chain, attr: str, grammar: str
) -> None:
    rows = chain.rows(getattr(chain, attr))
    assert rows
    for row in rows:
        assert orphaned_releases(row, grammar) == ()


def test_the_carried_pair_lands_in_two_consecutive_turns(chain: Chain) -> None:
    # KeyA is pressed in W2 and released in W3: the split/carry case. Both
    # halves are present, in order, in adjacent assistant turns — the press is
    # not repeated and the release is not dropped.
    (row,) = chain.rows(chain.conv_canonical)
    texts = assistant_texts(row)
    assert ("+", "KeyA") in key_transitions(texts[2], "deltatype_v2")
    assert ("-", "KeyA") in key_transitions(texts[3], "deltatype_v2")
    assert sum(1 for t in _all_transitions([row], "deltatype_v2") if t[1] == "KeyA") == 2


def test_a_press_still_held_when_the_clip_ends_is_kept_and_counted(chain: Chain) -> None:
    # The invariant is one-sided on purpose: KeyC was pressed in front of a
    # frame the trainee sees, so its press is real supervision even though no
    # release follows. KeyQ, pressed inside the black span and never released,
    # is dropped instead of clamped forward — clamping it would have
    # manufactured exactly the orphan the class is about.
    (row,) = chain.rows(chain.conv_canonical)
    assert unreleased_presses(row, "deltatype_v2") == ("KeyC",)
    assert row["dead_zone_counters"]["n_held_at_end"] == 2
    assert row["dead_zone_counters"]["n_unreleased_press_dropped"] == 1
    assert not any(name == "KeyQ" for _, name in _all_transitions([row], "deltatype_v2"))


def test_the_dangling_release_never_reaches_a_label(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_canonical)
    assert not any(name == "KeyZ" for _, name in _all_transitions([row], "deltatype_v2"))
    assert row["dead_zone_counters"]["n_dangling_release"] == 1


# --------------------------------------------------------------------------
# Black-frame spans and action attribution
# --------------------------------------------------------------------------


def test_black_span_input_is_not_attributed_to_a_visible_frame(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_canonical)
    texts = assistant_texts(row)
    # The (7,7) move happened while the screen was black; no label carries it.
    assert not any(text.startswith("7 7 ") for text in texts)
    names = {name for _, name in _all_transitions([row], "deltatype_v2")}
    assert not names & {"KeyM", "KeyQ"}
    counters = row["dead_zone_counters"]
    assert counters["n_discarded_black"] == 1
    assert counters["n_pairs_dropped_dead_zone"] == 1


def test_without_black_masking_the_same_input_corrupts_a_visible_turn(chain: Chain) -> None:
    # The differential. With the span kept, the trainee is shown a black frame
    # and taught the move and the four key transitions that happened behind it;
    # KeyR's release lands one turn late, and KeyQ becomes a press with no
    # release anywhere in the conversation.
    (row,) = chain.rows(chain.conv_black_off)
    texts = assistant_texts(row)
    assert texts[4] == "7 7 0 ; +KeyM -KeyR +KeyB +KeyQ -KeyM"
    assert texts[3] == "0 0 0 ; -KeyA +KeyR"
    assert unreleased_presses(row, "deltatype_v2") == ("KeyQ", "KeyC")
    assert chain.segment_filter(chain.filter_black_off)["n_black"] == 0
    assert row["dead_zone_counters"]["n_discarded_black"] == 0
    # Idle drops are NOT dead zones, so nothing about the inactive half of the
    # clip moves a discard counter either way.
    assert row["dead_zone_counters"]["n_discarded_pre_first_frame"] == 0


# --------------------------------------------------------------------------
# ordered_events_v3 + the control channel
# --------------------------------------------------------------------------


def test_ordered_events_v3_collapses_the_typing_run(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_v3)
    texts = assistant_texts(row)
    assert texts[: len(EXPECTED_V3_LABELS)] == EXPECTED_V3_LABELS
    summary = chain.summary(chain.conv_v3)
    assert summary["grammar"] == "ordered_events_v3"
    assert summary["primitive_counts"] == {
        "move": 5, "scroll": 1, "down": 3, "up": 2, "type": 3
    }
    assert summary["continuous_action_hz"] == 10.0


def test_the_terminal_token_rides_on_the_last_turn_only(chain: Chain) -> None:
    (row,) = chain.rows(chain.conv_v3)
    texts = assistant_texts(row)
    assert not any(TERMINAL_TOKEN in text for text in texts[:-1])
    assert texts[-1].endswith(f"\n{TERMINAL_TOKEN}")
    control = grammars.split_control(texts[-1])
    assert control.status == "success"
    # The action under the token still parses: the token is a channel, not a
    # word of the grammar.
    assert ORDERED_EVENTS_V3.parse(control.body).no_op is True


# --------------------------------------------------------------------------
# Goal projection — where the orphaned-release class came from
# --------------------------------------------------------------------------


def test_goal_projection_snaps_to_the_observation_the_first_action_came_from(
    chain: Chain,
) -> None:
    by_goal = {row["goal_id"]: row for row in chain.rows(chain.conv_goals)}
    assert set(by_goal) == {"g_head", "g_mid"}
    head, mid = by_goal["g_head"], by_goal["g_mid"]
    assert (head["n_turns"], head["snapped_start"]) == (3, False)
    assert (mid["n_turns"], mid["snapped_start"]) == (6, True)
    assert chain.summary(chain.conv_goals)["goal_projection_totals"] == {
        "n_goals": 2, "n_projected": 2, "n_empty_projection": 0,
        "n_too_few_frames": 0, "n_snapped": 1,
    }


def test_a_goal_slice_orphans_the_release_of_a_press_outside_it(chain: Chain) -> None:
    """The 141,702-orphaned-release class, reproduced on one segment.

    Goal mode formats the labels once over the WHOLE segment and then takes
    ``result.labels[f.view_idx]`` for the goal's frames only
    (``stage_04_build_conversations.build_segment_conversations``). ``g_mid``
    starts at master tick 13; ``+KeyA`` was emitted in the window at tick 8 and
    ``-KeyA`` in the window at tick 12. ``snap_start="before"`` pulls the tick-12
    observation in, so the release is carried into the conversation while the
    press that opened it is not.

    This is characterisation, not approval: the invariant itself is asserted
    (and expected to fail) by the test below.
    """
    by_goal = {row["goal_id"]: row for row in chain.rows(chain.conv_goals)}
    assert orphaned_releases(by_goal["g_head"], "deltatype_v2") == ()
    assert orphaned_releases(by_goal["g_mid"], "deltatype_v2") == ("KeyA",)
    assert assistant_texts(by_goal["g_mid"])[0] == "0 0 0 ; -KeyA +KeyR -KeyR"
    assert ("+", "KeyA") in _all_transitions([by_goal["g_head"]], "deltatype_v2")


@pytest.mark.xfail(
    strict=True,
    reason="stage 04 slices segment-global labels per goal without rebalancing the "
    "held set; when this XPASSes the class is fixed — drop the marker and the "
    "characterisation in test_a_goal_slice_orphans_the_release_of_a_press_outside_it",
)
def test_no_release_reaches_a_goal_conversation_without_its_press(chain: Chain) -> None:
    for row in chain.rows(chain.conv_goals):
        assert orphaned_releases(row, "deltatype_v2") == ()
