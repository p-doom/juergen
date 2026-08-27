"""Stage 04d (cuagym): cursor-perturbation grounding drills.

The cursor-perturbation probe (eval/cursor_probe.py) shows the relative-mouse
policy barely reads the pointer out of the pixels: composite the cursor
somewhere else and the predicted delta should absorb the whole shift (slope 1),
but the base model scores 0.02, our best checkpoint 0.55, the LoRA 0.17, and a
third of the time the emitted action line is byte-identical. The model is
dead-reckoning the cursor from the action history and from layout priors.

Demonstrations cannot fix that: on a demonstration the true cursor is always
exactly where the history says it is, so reading the pixels and dead-reckoning
are observationally equivalent, and nothing in the corpus ever shows the policy
what a *missed* click looks like from the inside.

So we manufacture that state. For a step whose executed action was a plain
left click we know the true cursor C, the executed target T, and -- from
another frame of the same rollout -- the real background under the pointer. We
erase the pointer, repaint the same extracted sprite at a new position C'
sampled around T (the post-miss geometry the policy actually visits) and
supervise the corrective action from C' to the unchanged T. Now the history is
silent about where the pointer is and only the pixels carry the answer.

Everything that decides whether the erase is honest is reused from
eval/cursor_probe.py: the sprite matte, the repaint rectangle, the search for a
stray pointer in the donor frame, and the arrow-only reconstruction test. Two
things differ, both because the probe needs 150 pristine pairs while this needs
tens of thousands:

  donor      the probe pastes from frame t-1 only. But C is exactly where the
             last click landed, so the pixels there are the likeliest on the
             whole screen to have just changed -- 62% of candidates die on that
             one test. Any frame of the rollout will do as long as it agrees
             with frame t around C, so we try a window either side (frames
             after t first: the state at C is what step t-1's click put there)
             and keep the first that passes. Pass rate 3.6% -> 24.9%.
  region     the probe demands the whole 84x96 halo be unchanged. Only the
             pasted rectangle has to be right, so agreement is tested over that
             rectangle plus a thin ring (--ring_px), by mean and by worst
             pixel. Residual pointers are still caught by the donor-core
             search, which sweeps +-32 px, wider than the rectangle.

Records are single-turn (no live history turns) but otherwise byte-compatible
with stage_04: same system prompt, same instruction template, same message
layout, same ordered_events_v3 action line, same ar:// image refs. The
perturbed frame is the only new image, written to its own ArrayRecord store
following stage_01 conventions; nothing else is duplicated.

Modes
  sprite    extract the arrow matte and cache it to --sprite_path
  build     write the drill image store + chat.jsonl + manifest.jsonl
  verify    invertibility, parse rate, sprite localization, |delta| spread
  montage   contact sheet: C' and T framed together, or zoomed on either the
            repainted pointer or the erase site
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from absl import app, flags
from PIL import Image, ImageDraw

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATA_PIPELINE_DIR.parent
EVAL_DIR = REPO_ROOT / "eval"
for _p in (str(DATA_PIPELINE_DIR), str(REPO_ROOT), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from action_parser import parse_ordered_action  # noqa: E402
from cuagym_pipeline.oev3_render import (  # noqa: E402
    join_primitives,
    render_down,
    render_move,
    render_up,
)
from cuagym_pipeline.stage_04_build_conversations import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT_PATH,
    INSTRUCTION_TEMPLATE,
    ImageIndex,
    build_step_record,
    translate_episode,
)
from cuagym_pipeline.translate import GRID, norm_to_px, px_to_norm  # noqa: E402
from cursor_probe import (  # noqa: E402
    HOT_X,
    HOT_Y,
    SEARCH_R,
    SPR_H,
    SPR_W,
    _encode,
    _in_bounds,
    _load_rgb,
    _screen_box,
    composite,
    crop_at_hotspot,
    cursor_match_errors,
    estimate_sprite,
    nearest_cursor_err,
    sprite_crop_fits,
    sprite_masks,
)
from realigned_pipeline.lib.image_store import (  # noqa: E402
    make_arrayrecord_image_uri,
    read_jpeg_bytes,
)

FLAGS = flags.FLAGS

SHARD_NAME = "images.array_record"
INDEX_NAME = "index.jsonl"
SUMMARY_NAME = "summary.json"
CLICK_ACTIONS = frozenset({"left_click", "click"})

flags.DEFINE_enum("mode", "build", ["sprite", "build", "verify", "montage"], "What to run.")
flags.DEFINE_string("trajectories", None, "Path to the rollout trajectories.jsonl.")
flags.DEFINE_string("image_index_root", None, "stage_01 image store root (per-tar index.jsonl).")
flags.DEFINE_string("image_output_root", None, "Output root for the perturbed-frame image store.")
flags.DEFINE_string("output_dir", None, "Output dataset dir (chat.jsonl + manifest.jsonl).")
flags.DEFINE_string("sprite_path", None, "Cursor sprite npz (written by --mode=sprite).")
flags.DEFINE_string("system_prompt_path", str(DEFAULT_SYSTEM_PROMPT_PATH), "System prompt file.")
flags.DEFINE_integer("limit", 0, "Max rollouts to read (0 = all).")
flags.DEFINE_integer("num_workers", 0, "Worker processes (0 = cpu count).")
flags.DEFINE_integer("num_parts", 0, "Output shards / work units (0 = 4x --num_workers).")
flags.DEFINE_integer("seed", 0, "Root seed; the per-step stream is keyed by task_id and step.")
flags.DEFINE_integer("max_per_rollout", 3, "Cap on drills kept per rollout.")
flags.DEFINE_integer("keep_percent", 100, "Percent of surviving drills kept (deterministic).")

flags.DEFINE_float("near_frac", 0.75, "Fraction of C' drawn from the post-miss band.")
flags.DEFINE_integer("near_min_px", 30, "Inner radius of the post-miss band around T.")
flags.DEFINE_integer("near_max_px", 400, "Outer radius of the post-miss band around T.")
flags.DEFINE_integer("far_max_px", 1200, "Outer radius of the long-displacement tail.")

flags.DEFINE_integer("donor_window", 6, "How many frames either side of t may donate the background.")
flags.DEFINE_integer("donor_min_px", 60, "A donor's own pointer must be at least this far from C.")
flags.DEFINE_integer("frame_cache", 64, "Frames held per worker (uint8, ~6 MB each).")
flags.DEFINE_integer("ring_px", 6, "Ring around the repaint rectangle that must match the donor.")
flags.DEFINE_float("bg_tol", 1.5, "Max mean abs difference in that ring.")
flags.DEFINE_float("bg_max_tol", 40.0, "Max single-pixel abs difference in that ring.")
flags.DEFINE_float("prev_core_min", 30.0, "Min sprite-core match error in the frame we paste from.")
flags.DEFINE_float("arrow_tol", 15.0, "Max recomposite error: only the plain arrow passes.")

flags.DEFINE_integer("sprite_crops", 800, "Crops used to estimate the sprite matte.")
flags.DEFINE_integer("sprite_rollout_stride", 7, "Rollout stride while collecting sprite crops.")
flags.DEFINE_integer("sprite_steps_per_rollout", 2, "Crops taken from each visited rollout.")

flags.DEFINE_enum(
    "previous_actions_mode",
    "true",
    ["true", "none"],
    "'true': the real elided step list, as stage_04 emits at history_n=0, so the "
    "history stays in distribution and dead-reckoning is actively contradicted. "
    "'none': an empty step list, matching a step-0 record.",
)

flags.DEFINE_integer("verify_localize_n", 400, "Frames sampled for the sprite-localization gate.")
flags.DEFINE_integer("verify_localize_radius", 8, "Search radius for that gate.")

flags.DEFINE_string("montage_out", None, "Where to write the contact sheet PNG.")
flags.DEFINE_integer("montage_n", 20, "Tiles on the contact sheet.")
flags.DEFINE_integer("montage_cols", 4, "Tile columns.")
flags.DEFINE_integer("montage_tile", 380, "Tile width in pixels.")
flags.DEFINE_enum(
    "montage_center",
    "pair",
    ["pair", "cursor", "erased"],
    "'pair': frame C' and T together. 'cursor': zoom on C', where the sprite "
    "was painted. 'erased': zoom on the true C, which must show clean "
    "background and no leftover pointer.",
)
flags.DEFINE_integer("montage_zoom_px", 0, "Source crop width for the zoom centers (0 = 96).")


def _thresholds() -> dict:
    return {
        "ring_px": FLAGS.ring_px,
        "bg_tol": FLAGS.bg_tol,
        "bg_max_tol": FLAGS.bg_max_tol,
        "prev_core_min": FLAGS.prev_core_min,
        "arrow_tol": FLAGS.arrow_tol,
    }


def _sampling() -> dict:
    return {
        "near_frac": FLAGS.near_frac,
        "near_min_px": FLAGS.near_min_px,
        "near_max_px": FLAGS.near_max_px,
        "far_max_px": FLAGS.far_max_px,
    }


def line_offsets(path: Path, limit: int = 0) -> list[int]:
    offsets = []
    with path.open("rb") as fh:
        pos = fh.tell()
        for _ in fh:
            offsets.append(pos)
            pos = fh.tell()
            if limit and len(offsets) >= limit:
                break
    return offsets


def read_at(path_str: str, offset: int) -> dict:
    with open(path_str, "rb") as fh:
        fh.seek(offset)
        return json.loads(fh.readline())


def paint_in_bounds(nx: int, ny: int, sw: int, sh: int, margin: int = 3) -> bool:
    """The sprite frame and the repaint rectangle both fit on screen at C'."""
    x0, y0, x1, y1 = _screen_box(nx, ny)
    x0, y0 = min(x0, nx - HOT_X), min(y0, ny - HOT_Y)
    x1, y1 = max(x1, nx - HOT_X + SPR_W), max(y1, ny - HOT_Y + SPR_H)
    return x0 >= margin and y0 >= margin and x1 <= sw - margin and y1 <= sh - margin


def sample_cursor(rng: random.Random, tx: int, ty: int, sw: int, sh: int, cfg: dict):
    """A pointer position around T: mostly post-miss range, with a long tail."""
    lo, mid, hi = cfg["near_min_px"], cfg["near_max_px"], cfg["far_max_px"]
    for _ in range(64):
        mag = rng.uniform(lo, mid) if rng.random() < cfg["near_frac"] else rng.uniform(mid, hi)
        ang = rng.uniform(0, 2 * math.pi)
        nx = int(round(tx + mag * math.cos(ang)))
        ny = int(round(ty + mag * math.sin(ang)))
        if math.hypot(nx - tx, ny - ty) < lo:
            continue
        if paint_in_bounds(nx, ny, sw, sh):
            return nx, ny
    return None


def erase_diagnostics(cur, donor, sprite_rgb, alpha, masks, cx, cy, ring: int) -> dict:
    """Pixel evidence that pasting the donor frame really recovers the background.

    Same questions as cursor_probe.build_pair -- does the background agree
    where the cursor is not, is there a second pointer hiding in the frame we
    paste from, and is this the plain arrow -- but asked over the repaint
    rectangle plus ``ring`` pixels rather than the probe's full halo, and of a
    donor that need not be the immediately preceding frame.
    """
    core, foot = masks
    sy, sx = cy - HOT_Y, cx - HOT_X
    obs = cur[sy : sy + SPR_H, sx : sx + SPR_W]
    bg = donor[sy : sy + SPR_H, sx : sx + SPR_W]
    bx0, by0, bx1, by1 = _screen_box(cx, cy)
    rx0, ry0 = min(sx, bx0) - ring, min(sy, by0) - ring
    rx1, ry1 = max(sx + SPR_W, bx1) + ring, max(sy + SPR_H, by1) + ring
    ring_diff = np.abs(cur[ry0:ry1, rx0:rx1] - donor[ry0:ry1, rx0:rx1]).mean(axis=2)
    keep = np.ones(ring_diff.shape, bool)
    keep[sy - ry0 : sy - ry0 + SPR_H, sx - rx0 : sx - rx0 + SPR_W] &= ~foot
    recon = alpha[..., None] * sprite_rgb + (1 - alpha[..., None]) * bg
    err = np.abs(recon - obs)
    return {
        "bg_err": float(ring_diff[keep].mean()),
        "bg_max": float(ring_diff[keep].max()),
        "donor_core_err": nearest_cursor_err(donor, sprite_rgb, core, cx, cy),
        "arrow_err": float(err.mean(axis=2)[core].mean()),
        "recon_err": float(err.mean()),
    }


def reject_reason(diag: dict, thr: dict) -> str | None:
    if diag["bg_err"] > thr["bg_tol"] or diag["bg_max"] > thr["bg_max_tol"]:
        return "bg_moved"
    if diag["donor_core_err"] < thr["prev_core_min"]:
        return "donor_cursor_overlap"
    if diag["arrow_err"] > thr["arrow_tol"]:
        return "not_arrow_cursor"
    return None


def repaint(cur, donor, sprite_rgb, alpha, cx: int, cy: int, nx: int, ny: int):
    x0, y0, x1, y1 = _screen_box(cx, cy)
    out = cur.copy()
    out[y0:y1, x0:x1] = donor[y0:y1, x0:x1]
    sx, sy = nx - HOT_X, ny - HOT_Y
    patch = out[sy : sy + SPR_H, sx : sx + SPR_W].astype(np.float32)
    composite(patch, sprite_rgb, alpha, HOT_X, HOT_Y)
    out[sy : sy + SPR_H, sx : sx + SPR_W] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def move_delta_of(line: str) -> tuple[int, int] | None:
    head = line.split(";", 1)[0].strip()
    if not head.startswith("move(") or not head.endswith(")"):
        return None
    try:
        dx, dy = (int(v) for v in head[5:-1].split(","))
    except ValueError:
        return None
    return dx, dy


def drill_think(cursor_norm, target_norm, delta) -> str:
    return (
        f"The cursor tip is at ({cursor_norm[0]},{cursor_norm[1]}) on the {GRID}x{GRID} grid "
        f"and I need to click at ({target_norm[0]},{target_norm[1]}), so I move by "
        f"({target_norm[0]}-{cursor_norm[0]},{target_norm[1]}-{cursor_norm[1]}) "
        f"= ({delta[0]},{delta[1]})."
    )


def keep_by_hash(key: str, percent: int) -> bool:
    if percent >= 100:
        return True
    return hashlib.sha256(key.encode()).digest()[1] % 100 < percent


def aligned_raw_steps(rec: dict, steps: list[dict]) -> list[dict] | None:
    """The raw steps behind ``translate_episode``'s output, index for index."""
    raw = [s for s in (rec.get("steps") or []) if s.get("shard") and s.get("member")]
    return raw if len(raw) == len(steps) else None


def candidate_steps(
    rec: dict, steps: list[dict], raw: list[dict]
) -> list[tuple[int, dict, tuple[int, int]]]:
    """Steps whose gold action is a single left click that moved the pointer."""
    out = []
    for t in range(len(steps)):
        entry = steps[t]
        if entry["target"] is None or not entry["line"]:
            continue
        args = raw[t].get("raw_action_args") or {}
        if args.get("action") not in CLICK_ACTIONS or args.get("coordinate") is None:
            continue
        if raw[t].get("cursor_before") is None or raw[t].get("coordinate_screen") is None:
            continue
        delta = move_delta_of(entry["line"])
        if delta is None or entry["line"] != f"{render_move(*delta)}; down(LMB); up(LMB)":
            continue
        out.append((t, raw[t], delta))
    return out


class FrameCache:
    """LRU of whole rollout frames, kept as uint8 (6 MB, not the 25 MB float)."""

    def __init__(self, images: ImageIndex, maxsize: int):
        self._images = images
        self._maxsize = maxsize
        self._frames: OrderedDict[str, np.ndarray] = OrderedDict()

    def uri(self, steps: list[dict], i: int) -> str:
        return self._images.uri(steps[i]["shard"], steps[i]["member"])

    def get(self, steps: list[dict], i: int) -> np.ndarray:
        uri = self.uri(steps, i)
        hit = self._frames.pop(uri, None)
        if hit is None:
            hit = np.asarray(Image.open(io.BytesIO(read_jpeg_bytes(uri))).convert("RGB"))
        self._frames[uri] = hit
        while len(self._frames) > self._maxsize:
            self._frames.popitem(last=False)
        return hit


def local_window(frame: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Float32 crop around the hotspot, exactly the span every gate looks at.

    ``_in_bounds`` guarantees this fits: its pad is ``max(HALO, SEARCH_R)`` and
    the widest reader here is ``nearest_cursor_err``, which needs SEARCH_R.
    """
    x0, y0 = cx - HOT_X - SEARCH_R, cy - HOT_Y - SEARCH_R
    return frame[
        y0 : y0 + SPR_H + 2 * SEARCH_R, x0 : x0 + SPR_W + 2 * SEARCH_R
    ].astype(np.float32)


LX, LY = HOT_X + SEARCH_R, HOT_Y + SEARCH_R


def donor_order(t: int, n: int, window: int):
    """Frame indices to try as the background donor, nearest first.

    Forward before backward at each distance: the pixels around C are what
    step t-1's click just put there, so the frames that still show that state
    are the ones after it, not the ones before.
    """
    for d in range(1, window + 1):
        for k in (t + d, t - d):
            if 0 <= k < n and k != t:
                yield k


def gate_rollout(
    rec: dict,
    steps: list[dict],
    raw: list[dict],
    frames: FrameCache,
    sprite_rgb,
    alpha,
    masks,
    cfg: dict,
    stats: Counter,
) -> list[dict]:
    """Geometry for every step that survives the erase gates, before the cap.

    Deliberately does not repaint or encode: most of what passes here is
    discarded by the per-rollout cap, and a 1920x1080 JPEG encode costs about
    as much as the decode that got us here.
    """
    screen = tuple(rec.get("screen") or (1920, 1080))
    sw, sh = screen
    cands = candidate_steps(rec, steps, raw)
    if not cands:
        return []
    stats["candidate_steps"] += len(cands)
    thr = cfg["thr"]
    out = []
    for t, raw_step, delta in cands:
        cx, cy = (int(round(v)) for v in raw_step["cursor_before"])
        if not _in_bounds(cx, cy, sw, sh):
            stats["reject_cursor_at_edge"] += 1
            continue
        cursor_norm = px_to_norm((cx, cy), screen)
        target_norm = (cursor_norm[0] + delta[0], cursor_norm[1] + delta[1])
        tx, ty = norm_to_px(target_norm, screen)
        rng = random.Random(f"{cfg['seed']}|{rec['task_id']}|{steps[t]['step']}")
        spot = sample_cursor(rng, tx, ty, sw, sh, cfg["sampling"])
        if spot is None:
            stats["reject_no_valid_spot"] += 1
            continue
        nx, ny = spot
        new_norm = px_to_norm((nx, ny), screen)
        new_delta = (target_norm[0] - new_norm[0], target_norm[1] - new_norm[1])
        if new_delta == (0, 0):
            stats["reject_zero_delta"] += 1
            continue
        win_cur = local_window(frames.get(steps, t), cx, cy)
        best = None
        why = "no_clean_donor"
        for k in donor_order(t, len(steps), cfg["donor_window"]):
            other = raw[k].get("cursor_before")
            if other is None or math.hypot(other[0] - cx, other[1] - cy) < cfg["donor_min_px"]:
                continue
            win_donor = local_window(frames.get(steps, k), cx, cy)
            if win_donor.shape != win_cur.shape:
                continue
            diag = erase_diagnostics(
                win_cur, win_donor, sprite_rgb, alpha, masks, LX, LY, thr["ring_px"]
            )
            reason = reject_reason(diag, thr)
            if reason is None:
                best = (k, diag)
                break
            why = reason
        if best is None:
            stats["reject_" + why] += 1
            continue
        donor_k, diag = best
        line = join_primitives([render_move(*new_delta), render_down("LMB"), render_up("LMB")])
        parse_ordered_action(line)
        out.append(
            {
                "t": t,
                "donor_t": donor_k,
                "step": steps[t]["step"],
                "donor_step": steps[donor_k]["step"],
                "line": line,
                "think": drill_think(new_norm, target_norm, new_delta),
                "cursor_true_px": [cx, cy],
                "cursor_px": [nx, ny],
                "source_image": frames.uri(steps, t),
                "cursor_norm": list(new_norm),
                "target_px": [tx, ty],
                "target_screen_px": [
                    min(max(int(round(v)), 0), lim - 1)
                    for v, lim in zip(raw_step["coordinate_screen"], (sw, sh))
                ],
                "target_norm": list(target_norm),
                "delta": list(new_delta),
                "gold_delta": list(delta),
                "dist_px": round(math.hypot(nx - tx, ny - ty), 2),
                "shift_px": [nx - cx, ny - cy],
                "screen": [sw, sh],
                **{k: round(v, 4) for k, v in diag.items()},
            }
        )
        stats["passed_gates"] += 1
    return out


def worker(job: tuple) -> dict:
    part, offsets, cfg = job
    from array_record.python.array_record_module import ArrayRecordWriter

    images = ImageIndex(Path(cfg["image_index_root"]))
    sprite = np.load(cfg["sprite_path"])
    sprite_rgb, alpha = sprite["rgb"], sprite["alpha"]
    masks = sprite_masks(alpha)
    system_prompt = Path(cfg["system_prompt_path"]).read_text().strip()

    image_root = Path(cfg["image_output_root"])
    part_name = f"drills-{part:04d}"
    final_dir = image_root / part_name
    final_shard = final_dir / SHARD_NAME
    tmp_dir = image_root / f".tmp_{part_name}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    parts_dir = Path(cfg["output_dir"]) / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    n_images = 0
    writer = ArrayRecordWriter(str(tmp_dir / SHARD_NAME), "group_size:1")
    try:
        with (tmp_dir / INDEX_NAME).open("w") as index_f, (
            parts_dir / f"chat-{part:04d}.jsonl"
        ).open("w") as chat_f, (parts_dir / f"manifest-{part:04d}.jsonl").open("w") as man_f:
            for offset in offsets:
                rec = read_at(cfg["trajectories"], offset)
                steps = translate_episode(rec, Counter())
                raw = aligned_raw_steps(rec, steps)
                if raw is None:
                    stats["reject_unaligned_rollout"] += 1
                    continue
                frames = FrameCache(images, cfg["frame_cache"])
                drills = gate_rollout(
                    rec, steps, raw, frames, sprite_rgb, alpha, masks, cfg, stats
                )
                stats[f"rollout_passes_{min(len(drills), 16):02d}"] += 1
                rng = random.Random(f"{cfg['seed']}|pick|{rec['task_id']}")
                rng.shuffle(drills)
                selected = sorted(drills[: cfg["max_per_rollout"]], key=lambda r: r["t"])
                selected = [
                    d
                    for d in selected
                    if keep_by_hash(f"{rec['task_id']}__s{d['step']:03d}__cd", cfg["keep_percent"])
                ]
                stats["reject_keep_percent"] += len(drills[: cfg["max_per_rollout"]]) - len(selected)
                kept = 0
                for d in selected:
                    conv_id = f"{rec['task_id']}__s{d['step']:03d}__cd"
                    writer.write(
                        _encode(
                            repaint(
                                frames.get(steps, d["t"]),
                                frames.get(steps, d["donor_t"]),
                                sprite_rgb,
                                alpha,
                                *d["cursor_true_px"],
                                *d["cursor_px"],
                            )
                        )
                    )
                    uri = make_arrayrecord_image_uri(final_shard, n_images)
                    index_f.write(json.dumps({"member": conv_id, "uri": uri}) + "\n")
                    n_images += 1
                    messages = build_step_record(
                        rec,
                        steps,
                        d["t"],
                        system_prompt=system_prompt,
                        images=images,
                        history_n=0,
                    )
                    messages[-2]["content"][0]["image"] = uri
                    if cfg["previous_actions_mode"] == "none":
                        messages[-2]["content"][1]["text"] = INSTRUCTION_TEMPLATE.format(
                            instruction=rec["instruction"], previous_actions="None"
                        )
                    messages[-1]["content"][0]["text"] = (
                        f"<think>{d['think']}\n</think>\n\n{d['line']}"
                    )
                    reward = rec.get("reward")
                    chat_f.write(
                        json.dumps(
                            {
                                "conversation_id": conv_id,
                                "recording_id": rec["task_id"],
                                "task_id": rec["task_id"],
                                "app": rec.get("app"),
                                "reward": reward,
                                "terminated": rec.get("terminated"),
                                "pool": "success"
                                if (reward is not None and reward > 0)
                                else "failure",
                                "target_step": d["step"],
                                "n_history_turns": 0,
                                "action_format": "ordered_events_v3",
                                "record_kind": "cursor_drill",
                                "messages": messages,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    man_f.write(
                        json.dumps(
                            {
                                "conversation_id": conv_id,
                                "image": uri,
                                "pool": "success"
                                if (reward is not None and reward > 0)
                                else "failure",
                                "app": rec.get("app"),
                                **{k: v for k, v in d.items() if k not in ("jpeg", "t")},
                            }
                        )
                        + "\n"
                    )
                    kept += 1
                stats["records"] += kept
                stats["rollouts"] += 1
    finally:
        writer.close()

    summary = {
        "part": part_name,
        "shard": str(final_shard),
        "num_images": n_images,
        "num_failures": 0,
        "jpeg_quality": 92,
    }
    (tmp_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)
    return {"part": part, "num_images": n_images, "stats": dict(stats)}


def cmd_sprite() -> None:
    """Collect cursor crops over spread-out steps and unmix the arrow.

    The estimator reads the matte off the per-pixel variance, so the crops must
    disagree about the background. Taking one crop per rollout at its first
    step does the opposite: every rollout of an app opens on the same desktop,
    the variance collapses, and the matte comes out as noise at alpha ~= 0.3
    over the whole frame.
    """
    images = ImageIndex(Path(FLAGS.image_index_root))
    offsets = line_offsets(Path(FLAGS.trajectories))
    crops = []
    for offset in offsets[:: FLAGS.sprite_rollout_stride]:
        rec = read_at(FLAGS.trajectories, offset)
        sw, sh = tuple(rec.get("screen") or (1920, 1080))
        usable = [
            s
            for s in (rec.get("steps") or [])
            if s.get("shard")
            and s.get("member")
            and s.get("cursor_before") is not None
            and sprite_crop_fits(*(int(round(v)) for v in s["cursor_before"]), sw, sh)
        ]
        if not usable:
            continue
        rng = random.Random(f"{FLAGS.seed}|sprite|{rec['task_id']}")
        for step in rng.sample(usable, min(FLAGS.sprite_steps_per_rollout, len(usable))):
            cx, cy = (int(round(v)) for v in step["cursor_before"])
            crops.append(
                crop_at_hotspot(_load_rgb(images.uri(step["shard"], step["member"])), cx, cy)
            )
        if len(crops) >= FLAGS.sprite_crops:
            break
    rgb, alpha, stats = estimate_sprite(crops)
    out = Path(FLAGS.sprite_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rgb=rgb, alpha=alpha)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    zoom = 8
    panels = [
        np.clip(rgb, 0, 255).astype(np.uint8),
        np.repeat((alpha * 255).astype(np.uint8)[..., None], 3, axis=2),
        np.clip(alpha[..., None] * rgb + (1 - alpha[..., None]) * 200.0, 0, 255).astype(np.uint8),
    ]
    strip = np.concatenate([np.pad(p, ((2, 2), (2, 2), (0, 0)), constant_values=64) for p in panels], axis=1)
    Image.fromarray(strip).resize(
        (strip.shape[1] * zoom, strip.shape[0] * zoom), Image.NEAREST
    ).save(out.with_suffix(".png"))
    print(json.dumps(stats, indent=2), flush=True)


def cmd_build() -> None:
    t0 = time.time()
    out_dir = Path(FLAGS.output_dir)
    image_root = Path(FLAGS.image_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    offsets = line_offsets(Path(FLAGS.trajectories), FLAGS.limit)
    workers = FLAGS.num_workers or mp.cpu_count()
    cfg = {
        "trajectories": FLAGS.trajectories,
        "image_index_root": FLAGS.image_index_root,
        "image_output_root": FLAGS.image_output_root,
        "output_dir": FLAGS.output_dir,
        "sprite_path": FLAGS.sprite_path,
        "system_prompt_path": FLAGS.system_prompt_path,
        "seed": FLAGS.seed,
        "max_per_rollout": FLAGS.max_per_rollout,
        "keep_percent": FLAGS.keep_percent,
        "previous_actions_mode": FLAGS.previous_actions_mode,
        "donor_window": FLAGS.donor_window,
        "donor_min_px": FLAGS.donor_min_px,
        "frame_cache": FLAGS.frame_cache,
        "thr": _thresholds(),
        "sampling": _sampling(),
    }
    n_parts = FLAGS.num_parts or 4 * workers
    bounds = [len(offsets) * k // n_parts for k in range(n_parts + 1)]
    jobs = [(k, offsets[bounds[k] : bounds[k + 1]], cfg) for k in range(n_parts)]
    stats: Counter = Counter()
    per_part = {}
    done = 0
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(worker, jobs):
            stats.update(res["stats"])
            per_part[res["part"]] = res["num_images"]
            done += 1
            print(
                f"[build] part {res['part']:04d} done ({done}/{n_parts}) "
                f"records={stats['records']}",
                flush=True,
            )

    parts_dir = out_dir / "parts"
    for name, glob in (("chat.jsonl", "chat-*.jsonl"), ("manifest.jsonl", "manifest-*.jsonl")):
        with (out_dir / name).open("wb") as dst:
            for src in sorted(parts_dir.glob(glob)):
                with src.open("rb") as fh:
                    shutil.copyfileobj(fh, dst)
    shutil.rmtree(parts_dir)

    manifest = {
        "artifact_type": "cuagym_stage_04d_cursor_drill_image_store",
        "schema_version": 1,
        "uri_scheme": "ar:///abs/path/images.array_record#idx",
        "jpeg_quality": 92,
        "num_parts": len(per_part),
        "total_images": sum(per_part.values()),
        "parts": {f"drills-{k:04d}": v for k, v in sorted(per_part.items())},
    }
    (image_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "trajectories": FLAGS.trajectories,
        "records": stats["records"],
        "rollouts": stats["rollouts"],
        "candidate_steps": stats["candidate_steps"],
        "passed_gates": stats["passed_gates"],
        "previous_actions_mode": FLAGS.previous_actions_mode,
        "max_per_rollout": FLAGS.max_per_rollout,
        "keep_percent": FLAGS.keep_percent,
        "seed": FLAGS.seed,
        "donor_window": FLAGS.donor_window,
        "donor_min_px": FLAGS.donor_min_px,
        "thresholds": _thresholds(),
        "sampling": _sampling(),
        "gate_pass_rate": (
            stats["passed_gates"] / stats["candidate_steps"] if stats["candidate_steps"] else 0.0
        ),
        "elapsed_s": round(time.time() - t0, 1),
        **{k: v for k, v in sorted(stats.items()) if k.startswith("reject_")},
        "rollout_pass_histogram": {
            k.removeprefix("rollout_passes_"): v
            for k, v in sorted(stats.items())
            if k.startswith("rollout_passes_")
        },
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else 0.0


def best_cursor_offset(img, sprite_rgb, core, cx: int, cy: int, radius: int):
    errs = cursor_match_errors(img, sprite_rgb, core, cx, cy, radius)
    i, j = np.unravel_index(int(np.argmin(errs)), errs.shape)
    return int(j - radius), int(i - radius), float(errs[i, j])


def localization_check(rows: list[dict], sprite_path: str, n: int, radius: int) -> dict:
    """Where does the pointer actually sit, relative to the position we labelled?

    Run on the drill frame at C' and, as a control, on the untouched source
    frame at the logged C. Any constant offset between the OS bitmap and the
    logged hotspot cancels: the sprite was cut at the logged position, so it is
    pasted back at the same relative offset the real pointer has. The control
    is what proves that, and the two histograms have to agree.
    """
    blob = np.load(sprite_path)
    sprite_rgb, alpha = blob["rgb"], blob["alpha"]
    core, _ = sprite_masks(alpha)
    rng = random.Random(0)
    picks = rng.sample(rows, min(n, len(rows)))
    out = {}
    for label, key, pos_key in (
        ("drill_at_perturbed", "image", "cursor_px"),
        ("source_at_true", "source_image", "cursor_true_px"),
    ):
        offsets: Counter = Counter()
        errs = []
        for r in picks:
            if key not in r:
                continue
            img = np.asarray(
                Image.open(io.BytesIO(read_jpeg_bytes(r[key]))).convert("RGB"), dtype=np.float32
            )
            cx, cy = r[pos_key]
            sh, sw = img.shape[:2]
            if not _in_bounds(cx, cy, sw, sh):
                continue
            dx, dy, err = best_cursor_offset(img, sprite_rgb, core, cx, cy, radius)
            offsets[f"{dx},{dy}"] += 1
            errs.append(err)
        total = sum(offsets.values())
        out[label] = {
            "n": total,
            "exact_hotspot_rate": offsets.get("0,0", 0) / total if total else 0.0,
            "within_1px_rate": (
                sum(v for k, v in offsets.items() if max(abs(int(x)) for x in k.split(",")) <= 1)
                / total
                if total
                else 0.0
            ),
            "top_offsets": offsets.most_common(5),
            "match_err_median": _pct(errs, 50),
        }
    return out


def cmd_verify() -> None:
    out_dir = Path(FLAGS.output_dir)
    rows = [json.loads(x) for x in (out_dir / "manifest.jsonl").read_text().splitlines() if x.strip()]
    by_id = {r["conversation_id"]: r for r in rows}
    dist, mag, shift, quant = [], [], [], []
    bad_invert = bad_parse = bad_line = bad_shape = 0
    pools: Counter = Counter()
    apps: Counter = Counter()
    n_chat = 0
    system_prompt = Path(FLAGS.system_prompt_path).read_text().strip()
    with (out_dir / "chat.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            n_chat += 1
            pools[row["pool"]] += 1
            apps[row.get("app")] += 1
            man = by_id.get(row["conversation_id"])
            if man is None:
                bad_shape += 1
                continue
            msgs = row["messages"]
            ok_shape = (
                len(msgs) == 3
                and msgs[0]["role"] == "system"
                and msgs[0]["content"][0]["text"] == system_prompt
                and msgs[1]["role"] == "user"
                and msgs[1]["content"][0]["type"] == "image"
                and msgs[1]["content"][0]["image"] == man["image"]
                and msgs[1]["content"][1]["text"].startswith(
                    "\nPlease generate the next move according to the UI screenshot"
                )
                and msgs[2]["role"] == "assistant"
            )
            bad_shape += not ok_shape
            text = msgs[-1]["content"][0]["text"]
            action = text.rsplit("\n", 1)[-1]
            try:
                parse_ordered_action(action)
            except (ValueError, TypeError):
                bad_parse += 1
                continue
            if action != man["line"]:
                bad_line += 1
            screen = tuple(man["screen"])
            delta = move_delta_of(action)
            back = norm_to_px(
                (
                    px_to_norm(tuple(man["cursor_px"]), screen)[0] + delta[0],
                    px_to_norm(tuple(man["cursor_px"]), screen)[1] + delta[1],
                ),
                screen,
            )
            if max(abs(back[0] - man["target_px"][0]), abs(back[1] - man["target_px"][1])) > 0:
                bad_invert += 1
            gold = man["target_screen_px"]
            quant.append(max(abs(gold[0] - back[0]), abs(gold[1] - back[1])))
            dist.append(man["dist_px"])
            mag.append(math.hypot(*delta))
            shift.append(math.hypot(*man["shift_px"]))

    report = {
        "n_chat_rows": n_chat,
        "n_manifest_rows": len(rows),
        "pools": dict(pools),
        "top_apps": apps.most_common(10),
        "strict_parse_rate": (n_chat - bad_parse) / n_chat if n_chat else 0.0,
        "action_line_matches_manifest": n_chat - bad_line,
        "message_layout_ok": n_chat - bad_shape,
        "invertibility_failures_gt_1px": bad_invert,
        "invertibility_rate": (n_chat - bad_invert) / n_chat if n_chat else 0.0,
        "gold_target_agreement_max_px": max(quant) if quant else None,
        "dist_c_to_t_px": {
            "min": min(dist) if dist else None,
            "p10": _pct(dist, 10),
            "p50": _pct(dist, 50),
            "p90": _pct(dist, 90),
            "p99": _pct(dist, 99),
            "max": max(dist) if dist else None,
            "frac_in_30_400": sum(1 for d in dist if 30 <= d <= 400) / len(dist) if dist else 0.0,
        },
        "abs_delta_thousandths": {
            "min": min(mag) if mag else None,
            "p10": _pct(mag, 10),
            "p50": _pct(mag, 50),
            "p90": _pct(mag, 90),
            "max": max(mag) if mag else None,
        },
        "perturbation_shift_px": {
            "p10": _pct(shift, 10),
            "p50": _pct(shift, 50),
            "p90": _pct(shift, 90),
            "max": max(shift) if shift else None,
        },
    }
    if FLAGS.sprite_path:
        report["sprite_localization"] = localization_check(
            rows, FLAGS.sprite_path, FLAGS.verify_localize_n, FLAGS.verify_localize_radius
        )
    (out_dir / "verify_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def cmd_montage() -> None:
    out_dir = Path(FLAGS.output_dir)
    rows = [json.loads(x) for x in (out_dir / "manifest.jsonl").read_text().splitlines() if x.strip()]
    rng = random.Random(FLAGS.seed)
    picks = rng.sample(rows, min(FLAGS.montage_n, len(rows)))
    picks.sort(key=lambda r: r["dist_px"])

    tile_w = FLAGS.montage_tile
    tile_h = int(tile_w * 3 / 4)
    cols = FLAGS.montage_cols
    rowsn = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rowsn * (tile_h + 18)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    zoom_px = FLAGS.montage_zoom_px or 96
    for i, r in enumerate(picks):
        img = Image.open(io.BytesIO(read_jpeg_bytes(r["image"]))).convert("RGB")
        sw, sh = img.size
        cx, cy = r["cursor_px"]
        tx, ty = r["target_px"]
        if FLAGS.montage_center == "pair":
            pad = 70
            x0, y0 = min(cx, tx) - pad, min(cy, ty) - pad
            x1, y1 = max(cx, tx) + pad, max(cy, ty) + pad
            span = max(x1 - x0, (y1 - y0) * 4 // 3, 160)
            span = min(span, min(sw, sh * 4 // 3))
            mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        else:
            span = zoom_px
            mx, my = (cx, cy) if FLAGS.montage_center == "cursor" else tuple(r["cursor_true_px"])
        bx0 = min(max(mx - span // 2, 0), sw - span)
        by0 = min(max(my - (span * 3 // 4) // 2, 0), sh - span * 3 // 4)
        crop = img.crop((bx0, by0, bx0 + span, by0 + span * 3 // 4)).resize(
            (tile_w, tile_h), Image.LANCZOS
        )
        cd = ImageDraw.Draw(crop)
        sx = tile_w / span
        sy = tile_h / (span * 3 // 4)
        px, py = (cx - bx0) * sx, (cy - by0) * sy
        qx, qy = (tx - bx0) * sx, (ty - by0) * sy
        if FLAGS.montage_center == "erased":
            ex0, ey0, ex1, ey1 = _screen_box(*r["cursor_true_px"])
            cd.rectangle(
                [(ex0 - bx0) * sx, (ey0 - by0) * sy, (ex1 - bx0) * sx, (ey1 - by0) * sy],
                outline=(255, 40, 200),
                width=1,
            )
        else:
            cd.line([(px - 16, py), (px - 4, py)], fill=(255, 40, 40), width=2)
            cd.line([(px + 4, py), (px + 16, py)], fill=(255, 40, 40), width=2)
            cd.line([(px, py - 16), (px, py - 4)], fill=(255, 40, 40), width=2)
            cd.line([(px, py + 4), (px, py + 16)], fill=(255, 40, 40), width=2)
            cd.ellipse([qx - 11, qy - 11, qx + 11, qy + 11], outline=(40, 255, 90), width=2)
            if FLAGS.montage_center == "pair":
                cd.line([(px, py), (qx, qy)], fill=(255, 210, 60), width=1)
        ox, oy = (i % cols) * tile_w, (i // cols) * (tile_h + 18)
        sheet.paste(crop, (ox, oy))
        draw.text(
            (ox + 4, oy + tile_h + 3),
            f"{r['conversation_id'][:26]}  d={r['dist_px']:.0f}px  "
            f"move({r['delta'][0]},{r['delta'][1]})  arrow={r['arrow_err']:.1f}",
            fill=(220, 220, 220),
        )
    dest = Path(FLAGS.montage_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    print(f"[montage] {len(picks)} tiles -> {dest}", flush=True)


def main(argv):
    del argv
    if FLAGS.mode == "sprite":
        cmd_sprite()
    elif FLAGS.mode == "build":
        cmd_build()
    elif FLAGS.mode == "verify":
        cmd_verify()
    else:
        cmd_montage()


if __name__ == "__main__":
    app.run(main)
