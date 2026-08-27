"""Causal cursor test: does the model read the cursor out of the pixels?

The aim probe (bc_offline_score) cannot answer this. A model that has memorised
"the Save icon lives at the top left, so move up-left" scores a decent direction
cosine without ever looking at where the pointer currently is — the relative
action space makes layout priors and cursor reading observationally equivalent
on held-out demonstrations.

So we intervene. For each val record we rebuild the final screenshot twice,
identically except for one thing: where the mouse cursor is drawn.

    A (control)     cursor composited back at its true position C
    B (perturbed)   cursor composited at C' = C + shift instead

If the model reads the cursor it still aims at the same on-screen target, so its
predicted relative delta must absorb the whole shift:

    pred_B - pred_A  ==  -(C' - C)          slope 1  -> reads the cursor
    pred_B - pred_A  ==  0                  slope 0  -> layout / dead-reckoning

The reported slope of (pred_B - pred_A) on -(C' - C) is that number. Deltas are
grid units (1000x1000), so the pixel shift is scaled per axis by the screen.

Ground truth for C comes from the rollout logs, which record ``cursor_before``
in pixels for every step; the hotspot sits exactly on that pixel (verified by
cropping). The cursor is erased by pasting the same rectangle from the previous
screenshot — real background, no inpainting guesswork — and a record is only
kept when the pixels prove that erase is sound: the frames must agree outside
the cursor, disagree under it, and re-compositing the sprite must reproduce the
real pointer. The last test also restricts the probe to the plain arrow, since
I-beam, hand and resize pointers miss it by an order of magnitude.

Subcommands
  index    scrape 'task_id|step' -> cursor pixel from the rollout trajectories
  build    extract the sprite, then write the A/B screenshot pairs + manifest
  run      replay both conditions against sglang, one pass per condition
  score    regress (pred_B - pred_A) on -(C' - C) and write result.json
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openai
from array_record.python.array_record_module import ArrayRecordReader
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from action_parser import parse_ordered_action_tolerant
from bc_offline_score import _median, _ordered_to_action
from bc_teacher_forced_pairs import _joined_text, gold_action, sample_indices
from data_pipeline.realigned_pipeline.lib.image_store import read_jpeg_bytes
from hf_complete import complete_export_dir, find_hf_snapshot
from oev3_agent import extract_action_line
from result import write_result
from sglang_runner import sglang_server

GRID = 1000
JPEG_QUALITY = 92
N_BOOTSTRAP = 5000

# Sprite frame: the cursor hotspot sits at (HOT_X, HOT_Y) inside a SPR_W x SPR_H
# patch, big enough to hold the arrow plus its drop shadow.
HOT_X, HOT_Y = 8, 8
SPR_W, SPR_H = 32, 36
# Rectangle repainted when the cursor is moved, relative to the hotspot.
BOX = (-6, -4, 21, 29)
# Margin around the sprite frame that must also be unchanged between frames.
HALO = 26
# How far around the hotspot we hunt for a pointer in the frame we paste from.
SEARCH_R = 32

_IO_LOCK = threading.Lock()


def _screen_box(cx: int, cy: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = BOX
    return cx + x0, cy + y0, cx + x1, cy + y1


def _in_bounds(cx: int, cy: int, sw: int, sh: int, margin: int = 3) -> bool:
    """Both the repaint rectangle and the whole sprite frame must fit on screen."""
    x0, y0, x1, y1 = _screen_box(cx, cy)
    pad = max(HALO, SEARCH_R)
    x0, y0 = min(x0, cx - HOT_X - pad), min(y0, cy - HOT_Y - pad)
    x1, y1 = max(x1, cx - HOT_X + SPR_W + pad), max(y1, cy - HOT_Y + SPR_H + pad)
    return x0 >= margin and y0 >= margin and x1 <= sw - margin and y1 <= sh - margin


def composite(dst: np.ndarray, sprite_rgb: np.ndarray, alpha: np.ndarray, cx: int, cy: int) -> None:
    """Alpha-blend the cursor sprite into ``dst`` (float32 HxWx3) at hotspot (cx, cy)."""
    x0, y0 = cx - HOT_X, cy - HOT_Y
    a = alpha[..., None]
    dst[y0 : y0 + SPR_H, x0 : x0 + SPR_W] = (
        a * sprite_rgb + (1 - a) * dst[y0 : y0 + SPR_H, x0 : x0 + SPR_W]
    )


def _load_rgb(uri: str) -> np.ndarray:
    with _IO_LOCK:
        raw = read_jpeg_bytes(uri)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32)


def _encode(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=JPEG_QUALITY
    )
    return buf.getvalue()


def final_image_uri(rec: dict) -> str:
    return rec["messages"][-2]["content"][0]["image"]


def prev_image_uri(rec: dict) -> str | None:
    msgs = rec["messages"]
    if len(msgs) < 5 or msgs[-4].get("role") != "user":
        return None
    for part in msgs[-4]["content"]:
        if isinstance(part, dict) and part.get("type") == "image":
            return part["image"]
    return None


# --------------------------------------------------------------------------
# index


def cmd_index(args) -> None:
    index: dict[str, list[int]] = {}
    n = 0
    with Path(args.trajectories).open() as fh:
        for line in fh:
            rec = json.loads(line)
            tid = rec["task_id"]
            sw, sh = rec.get("screen") or (1920, 1080)
            for step in rec.get("steps") or []:
                cb = step.get("cursor_before")
                if cb is None:
                    continue
                index[f"{tid}|{step['step']}"] = [
                    int(round(cb[0])), int(round(cb[1])), int(sw), int(sh)
                ]
            n += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index))
    print(f"[index] {n} rollouts, {len(index)} steps -> {out}", flush=True)


# --------------------------------------------------------------------------
# sprite


def crop_at_hotspot(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """A copy, not a view: a kept slice would pin the whole 25 MB frame."""
    return img[cy - HOT_Y : cy - HOT_Y + SPR_H, cx - HOT_X : cx - HOT_X + SPR_W].copy()


def sprite_crop_fits(cx: int, cy: int, sw: int, sh: int) -> bool:
    return HOT_X <= cx < sw - (SPR_W - HOT_X) and HOT_Y <= cy < sh - (SPR_H - HOT_Y)


def estimate_sprite(crops: list[np.ndarray], iters: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """Recover the arrow's colour and alpha matte from many cursor crops.

    Across crops the cursor pixels are identical while the background is not,
    so per-pixel variance gives the matte: var(obs) = (1-a)^2 var(bg). The
    colour is then unmixed from the crop mean. Crops whose core disagrees with
    the running estimate are dropped, which sheds the non-arrow cursor shapes
    (I-beam, hand) that share the same hotspot convention.
    """
    stack = np.stack(crops)
    keep = np.ones(len(stack), bool)
    alpha = np.zeros((SPR_H, SPR_W), np.float32)
    rgb = np.zeros((SPR_H, SPR_W, 3), np.float32)
    stats = {}
    for _ in range(iters):
        sub = stack[keep]
        var = sub.var(axis=0).mean(axis=2)
        edge = np.concatenate(
            [var[:2].ravel(), var[-2:].ravel(), var[:, :2].ravel(), var[:, -2:].ravel()]
        )
        bg_var = max(float(np.median(edge)), 1e-6)
        alpha = np.clip(1.0 - np.sqrt(np.clip(var / bg_var, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)
        mean_obs = sub.mean(axis=0)
        bg_mean = np.median(
            np.concatenate([mean_obs[:2].reshape(-1, 3), mean_obs[-2:].reshape(-1, 3)]), axis=0
        )
        safe = np.maximum(alpha, 1e-3)[..., None]
        rgb = np.clip((mean_obs - (1 - safe) * bg_mean) / safe, 0, 255).astype(np.float32)
        core = alpha > 0.6
        err = (np.abs(stack - np.median(sub, axis=0)).mean(axis=3) * core).sum(axis=(1, 2))
        err = err / max(int(core.sum()), 1)
        keep = err <= np.percentile(err, 55)
        stats = {
            "n_crops": int(len(stack)),
            "n_kept": int(keep.sum()),
            "bg_var": bg_var,
            "core_px": int(core.sum()),
            "alpha_sum": float(alpha.sum()),
            "core_err_median": float(np.median(err)),
        }
    return rgb, alpha, stats


def extract_sprite(
    reader, index: dict, want: int, stride: int, iters: int = 4
) -> tuple[np.ndarray, np.ndarray, dict]:
    crops = []
    for i in range(0, reader.num_records(), stride):
        rec = json.loads(reader.read([i])[0])
        pos = index.get(f"{rec['recording_id']}|{rec['target_step']}")
        if pos is None:
            continue
        cx, cy, sw, sh = pos
        if not sprite_crop_fits(cx, cy, sw, sh):
            continue
        crops.append(crop_at_hotspot(_load_rgb(final_image_uri(rec)), cx, cy))
        if len(crops) >= want:
            break
    return estimate_sprite(crops, iters)


# --------------------------------------------------------------------------
# build


def sample_shift(rng: random.Random, cx: int, cy: int, sw: int, sh: int,
                 lo: int, hi: int) -> tuple[int, int] | None:
    """A shift of |lo..hi| px in a random direction that keeps C' on screen."""
    for _ in range(64):
        ang = rng.uniform(0, 2 * math.pi)
        mag = rng.uniform(lo, hi)
        dx, dy = int(round(mag * math.cos(ang))), int(round(mag * math.sin(ang)))
        if _in_bounds(cx + dx, cy + dy, sw, sh) and math.hypot(dx, dy) >= lo:
            return dx, dy
    return None


def _dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(iters):
        acc = out.copy()
        acc[1:] |= out[:-1]
        acc[:-1] |= out[1:]
        acc[:, 1:] |= out[:, :-1]
        acc[:, :-1] |= out[:, 1:]
        out = acc
    return out


def sprite_masks(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(core, footprint) masks in sprite coordinates."""
    return alpha > 0.5, _dilate(alpha > 0.08, 2)


def cursor_match_errors(img: np.ndarray, sprite_rgb: np.ndarray, core: np.ndarray,
                        cx: int, cy: int, radius: int = SEARCH_R) -> np.ndarray:
    """Sprite-core match error at every hotspot offset within ``radius``.

    Row i, column j is the error with the hotspot at (cx + j - radius,
    cy + i - radius). The arrow core is half dark and half white, so no flat
    patch of UI scores low here by accident.
    """
    ys, xs = np.nonzero(core)
    vals = sprite_rgb[core]
    y0, x0 = cy - HOT_Y - radius, cx - HOT_X - radius
    region = img[y0 : y0 + SPR_H + 2 * radius, x0 : x0 + SPR_W + 2 * radius]
    span = np.arange(-radius, radius + 1)
    oy, ox = np.meshgrid(span, span, indexing="ij")
    iy = oy.reshape(-1, 1) + radius + ys
    ix = ox.reshape(-1, 1) + radius + xs
    return np.abs(region[iy, ix] - vals).mean(axis=(1, 2)).reshape(oy.shape)


def nearest_cursor_err(img: np.ndarray, sprite_rgb: np.ndarray, core: np.ndarray,
                       cx: int, cy: int, radius: int = SEARCH_R) -> float:
    """Best match of the sprite core anywhere within ``radius`` of the hotspot.

    Used on the frame we paste from: if it holds a pointer close enough to land
    inside the repainted rectangle, the erase would leave a second cursor
    behind.
    """
    return float(cursor_match_errors(img, sprite_rgb, core, cx, cy, radius).min())


def build_pair(cur: np.ndarray, prev: np.ndarray, sprite_rgb, alpha, masks,
               cx: int, cy: int, nx: int, ny: int, thr: dict):
    """Return (control, perturbed, diagnostics). Control is None if unusable.

    Three things have to hold before a record is usable, and each is checked
    against pixels rather than assumed:

      bg_err    the previous frame is identical to this one everywhere the
                cursor is not, so pasting it really does recover the true
                background (and does not smuggle in its own cursor);
      core_diff the two frames genuinely differ under the cursor, i.e. the
                pointer moved between the frames and the erase does something;
      arrow_err re-compositing our sprite on that background reproduces the
                real cursor, which is only true for the plain arrow — I-beam,
                hand and resize pointers land far outside the tolerance.
    """
    core, foot = masks
    sy, sx = cy - HOT_Y, cx - HOT_X
    obs = cur[sy : sy + SPR_H, sx : sx + SPR_W]
    bg = prev[sy : sy + SPR_H, sx : sx + SPR_W]
    diff = np.abs(obs - bg).mean(axis=2)
    # The halo has to be compared too, not just the sprite frame: a pointer that
    # only crept a few pixels between the frames still has its old copy inside
    # the rectangle we paste, and would survive the erase as a second cursor.
    hy, hx = sy - HALO, sx - HALO
    hh, hw = SPR_H + 2 * HALO, SPR_W + 2 * HALO
    halo_diff = np.abs(cur[hy : hy + hh, hx : hx + hw] - prev[hy : hy + hh, hx : hx + hw]).mean(axis=2)
    halo_mask = np.ones((hh, hw), bool)
    halo_mask[HALO : HALO + SPR_H, HALO : HALO + SPR_W] &= ~foot
    diag = {
        "bg_err": float(halo_diff[halo_mask].mean()),
        # A stray pointer 10 px away is a handful of very wrong pixels, which a
        # mean over the whole halo dilutes below any usable threshold; count
        # them instead.
        "bg_hits": float((halo_diff[halo_mask] > 20).mean()),
        "core_diff": float(diff[core].mean()),
        # And if it barely moved at all, its old copy hides inside the new
        # one's footprint, where neither of the two checks above can see it.
        "prev_core_err": nearest_cursor_err(prev, sprite_rgb, core, cx, cy),
    }
    recon = alpha[..., None] * sprite_rgb + (1 - alpha[..., None]) * bg
    diag["arrow_err"] = float(np.abs(recon - obs).mean(axis=2)[core].mean())
    diag["recon_err"] = float(np.abs(recon - obs).mean())
    if (
        diag["bg_err"] > thr["bg_tol"]
        or diag["bg_hits"] > thr["bg_hit_tol"]
        or diag["core_diff"] < thr["core_min"]
        or diag["prev_core_err"] < thr["prev_core_min"]
        or diag["arrow_err"] > thr["arrow_tol"]
    ):
        return None, None, diag

    x0, y0, x1, y1 = _screen_box(cx, cy)
    erased = cur.copy()
    erased[y0:y1, x0:x1] = prev[y0:y1, x0:x1]
    control = erased.copy()
    composite(control, sprite_rgb, alpha, cx, cy)
    perturbed = erased
    composite(perturbed, sprite_rgb, alpha, nx, ny)
    return control, perturbed, diag


def cmd_build(args) -> None:
    index = json.loads(Path(args.cursor_index).read_text())
    reader = ArrayRecordReader(args.val_shard)
    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    sprite_path = out_dir / "sprite.npz"
    if sprite_path.is_file():
        blob = np.load(sprite_path)
        sprite_rgb, alpha = blob["rgb"], blob["alpha"]
        print(f"[sprite] reusing {sprite_path}", flush=True)
    else:
        sprite_rgb, alpha, stats = extract_sprite(
            reader, index, args.sprite_crops, args.sprite_stride
        )
        np.savez(sprite_path, rgb=sprite_rgb, alpha=alpha)
        (out_dir / "sprite_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[sprite] {json.dumps(stats)}", flush=True)

    masks = sprite_masks(alpha)
    thr = {
        "bg_tol": args.bg_tol,
        "bg_hit_tol": args.bg_hit_tol,
        "core_min": args.core_min,
        "prev_core_min": args.prev_core_min,
        "arrow_tol": args.arrow_tol,
    }
    rng = random.Random(args.seed)
    rows = []
    reject: dict[str, int] = {}
    total = reader.num_records()
    order = sample_indices(total, args.scan_records)
    for idx in order:
        if len(rows) >= args.num_records:
            break
        rec = json.loads(reader.read([idx])[0])
        pos = index.get(f"{rec['recording_id']}|{rec['target_step']}")
        if pos is None:
            reject["no_cursor_pos"] = reject.get("no_cursor_pos", 0) + 1
            continue
        cx, cy, sw, sh = pos
        prev_uri = prev_image_uri(rec)
        if prev_uri is None:
            reject["no_prev_frame"] = reject.get("no_prev_frame", 0) + 1
            continue
        if not _in_bounds(cx, cy, sw, sh):
            reject["cursor_at_edge"] = reject.get("cursor_at_edge", 0) + 1
            continue
        shift = sample_shift(rng, cx, cy, sw, sh, args.shift_min, args.shift_max)
        if shift is None:
            reject["no_valid_shift"] = reject.get("no_valid_shift", 0) + 1
            continue
        nx, ny = cx + shift[0], cy + shift[1]
        cur = _load_rgb(final_image_uri(rec))
        prev = _load_rgb(prev_uri)
        if cur.shape != prev.shape:
            reject["frame_shape"] = reject.get("frame_shape", 0) + 1
            continue
        control, perturbed, diag = build_pair(
            cur, prev, sprite_rgb, alpha, masks, cx, cy, nx, ny, thr
        )
        if control is None:
            why = (
                "bg_moved" if diag["bg_err"] > thr["bg_tol"]
                else "stray_pixels" if diag["bg_hits"] > thr["bg_hit_tol"]
                else "cursor_static" if diag["core_diff"] < thr["core_min"]
                else "prev_cursor_overlap" if diag["prev_core_err"] < thr["prev_core_min"]
                else "not_arrow_cursor"
            )
            reject[why] = reject.get(why, 0) + 1
            continue
        stem = out_dir / "images" / f"{idx:06d}"
        (stem.with_suffix(".a.jpg")).write_bytes(_encode(control))
        (stem.with_suffix(".b.jpg")).write_bytes(_encode(perturbed))
        rows.append(
            {
                "idx": idx,
                "gold": gold_action(rec["messages"]),
                "pool": rec.get("pool"),
                "app": rec.get("app"),
                "recording_id": rec.get("recording_id"),
                "target_step": rec.get("target_step"),
                "cursor": [cx, cy],
                "cursor_perturbed": [nx, ny],
                "shift_px": list(shift),
                "screen": [sw, sh],
                **{k: v for k, v in diag.items()},
                "image_a": str(stem.with_suffix(".a.jpg")),
                "image_b": str(stem.with_suffix(".b.jpg")),
            }
        )
        if len(rows) % 25 == 0:
            print(f"[build] {len(rows)}/{args.num_records}", flush=True)

    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report = {
        "val_shard": args.val_shard,
        "n_built": len(rows),
        "n_scanned": len(order),
        "rejected": reject,
        "recon_err_median": _median([r["recon_err"] for r in rows]) if rows else None,
        "recon_err_p90": (
            sorted(r["recon_err"] for r in rows)[int(0.9 * (len(rows) - 1))] if rows else None
        ),
        "thresholds": thr,
        "shift_px_range": [args.shift_min, args.shift_max],
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


# --------------------------------------------------------------------------
# run


def _data_url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def build_messages(rec: dict, final_image_path: str) -> list[dict]:
    """The record's own prompt, with the final screenshot swapped for ours."""
    msgs = rec["messages"][:-1]
    out: list[dict] = []
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    for i, m in enumerate(msgs):
        role, content = m["role"], m["content"]
        if role in ("system", "assistant") or isinstance(content, str):
            out.append({"role": role, "content": _joined_text(content)})
            continue
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "image":
                url = _data_url(final_image_path) if i == last_user else None
                if url is None:
                    with _IO_LOCK:
                        raw = read_jpeg_bytes(p["image"])
                    url = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
                parts.append({"type": "image_url", "image_url": {"url": url}})
            else:
                parts.append({"type": "text", "text": p.get("text", "") if isinstance(p, dict) else str(p)})
        out.append({"role": role, "content": parts})
    return out


def cmd_run(args) -> None:
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    reader = ArrayRecordReader(args.val_shard)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.ExitStack() as stack:
        if args.model_path:
            snapshot = find_hf_snapshot(args.model_id, Path(os.environ["HF_HOME"]))
            if Path(args.model_path).resolve() != snapshot.resolve():
                completion = complete_export_dir(Path(args.model_path), snapshot)
                print(f"[hf_complete] {completion}", flush=True)
            base_url = stack.enter_context(
                sglang_server(
                    model_path=args.model_path,
                    port=args.port,
                    api_key=args.api_key,
                    log_path=out_path.parent / "sglang_server.log",
                    mem_fraction_static=args.mem_fraction_static,
                    chunked_prefill_size=args.chunked_prefill_size,
                    served_model_name=args.model,
                )
            )
        else:
            base_url = args.base_url
        client = openai.OpenAI(base_url=base_url, api_key=args.api_key, timeout=600, max_retries=0)
        done = 0
        lock = threading.Lock()

        def one(row: dict) -> dict:
            nonlocal done
            with _IO_LOCK:
                rec = json.loads(reader.read([row["idx"]])[0])
            out = dict(row)
            for cond, key in (("image_a", "pred_a"), ("image_b", "pred_b")):
                raw, err = "", ""
                messages = build_messages(rec, row[cond])
                for attempt in range(args.retries):
                    try:
                        resp = client.chat.completions.create(
                            model=args.model,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=args.max_tokens,
                        )
                        msg = resp.choices[0].message
                        raw = msg.content or getattr(msg, "reasoning_content", None) or ""
                        break
                    except Exception as e:  # noqa: BLE001
                        err = f"{type(e).__name__}: {e}"
                        time.sleep(min(2**attempt, 30))
                try:
                    out[key] = extract_action_line(raw)
                except ValueError:
                    out[key] = ""
                if not raw and err:
                    out[key + "_error"] = err
            with lock:
                done += 1
                if done % 20 == 0:
                    print(f"[run] {done}/{len(rows)}", flush=True)
            return out

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(one, rows))

    with out_path.open("w") as fh:
        for r in sorted(results, key=lambda r: r["idx"]):
            fh.write(json.dumps(r) + "\n")
    n_err = sum(1 for r in results if r.get("pred_a_error") or r.get("pred_b_error"))
    print(f"[run] wrote {len(results)} rows to {out_path} ({n_err} request failures)", flush=True)
    if n_err > len(results) // 4:
        sys.exit(3)


# --------------------------------------------------------------------------
# score


def net_move(line: str) -> tuple[int, int] | None:
    if not line or line.strip().split("\n", 1)[0].strip() == "TERMINATE":
        return None
    try:
        act = _ordered_to_action(parse_ordered_action_tolerant(line))
    except (ValueError, TypeError):
        return None
    if act.no_op or (act.dx == 0 and act.dy == 0):
        return None
    return act.dx, act.dy


def _slope(xs, ys) -> float:
    sxx = sum(x * x for x in xs)
    return (sum(x * y for x, y in zip(xs, ys)) / sxx) if sxx else 0.0


def _fit(xs: list[float], ys: list[float], groups: list[int] | None = None) -> dict:
    """Slope through the origin, plus an outlier-proof median-of-ratios twin.

    The confidence interval resamples whole records, not single observations:
    the x and y axis of one screenshot share a prediction and are not
    independent draws.
    """
    if not xs:
        return {"n": 0, "slope": 0.0, "slope_robust": 0.0, "r2": 0.0, "intercept": 0.0}
    sxx = sum(x * x for x in xs)
    slope = (sum(x * y for x, y in zip(xs, ys)) / sxx) if sxx else 0.0
    ss_tot = sum(y * y for y in ys)
    ss_res = sum((y - slope * x) ** 2 for x, y in zip(xs, ys))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    varx = sum((x - mx) ** 2 for x in xs)
    ratios = [y / x for x, y in zip(xs, ys) if abs(x) > 20]
    lo = hi = None
    if groups:
        by_group: dict[int, list[tuple[float, float]]] = {}
        for g, x, y in zip(groups, xs, ys):
            by_group.setdefault(g, []).append((x, y))
        keys = list(by_group)
        rng = random.Random(0)
        boots = []
        for _ in range(N_BOOTSTRAP):
            bx: list[float] = []
            by: list[float] = []
            for _ in range(len(keys)):
                for x, y in by_group[keys[rng.randrange(len(keys))]]:
                    bx.append(x)
                    by.append(y)
            boots.append(_slope(bx, by))
        boots.sort()
        lo = boots[int(0.025 * N_BOOTSTRAP)]
        hi = boots[int(0.975 * N_BOOTSTRAP)]
    return {
        "n": n,
        "slope": slope,
        "slope_ci_lo": lo,
        "slope_ci_hi": hi,
        "slope_robust": _median(ratios) if ratios else 0.0,
        "slope_with_intercept": (cov / varx) if varx else 0.0,
        "intercept": my - (cov / varx) * mx if varx else 0.0,
        "r2": (1 - ss_res / ss_tot) if ss_tot else 0.0,
    }


def _cosine(u: tuple[int, int] | None, v: tuple[int, int] | None) -> float | None:
    if u is None or v is None:
        return None
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return None
    return (u[0] * v[0] + u[1] * v[1]) / (nu * nv)


def observations(rows: list[dict]) -> list[dict]:
    """One (expected, observed) pair per axis, for every usable row."""
    out = []
    for i, r in enumerate(rows):
        a, b = net_move(r.get("pred_a", "")), net_move(r.get("pred_b", ""))
        if a is None or b is None:
            continue
        sw, sh = r["screen"]
        sx, sy = r["shift_px"]
        aim = _cosine(net_move(r.get("gold", "")), a)
        for axis, expect, obs in (
            ("x", -sx / sw * GRID, b[0] - a[0]),
            ("y", -sy / sh * GRID, b[1] - a[1]),
        ):
            out.append({
                "row": i,
                "axis": axis,
                "expect": expect,
                "obs": obs,
                "pool": r.get("pool"),
                "aim_cosine": aim,
            })
    return out


def compensation_rates(obs: list[dict], floor: float = 20.0) -> dict:
    """How often the model actually tracks the cursor, outlier-free.

    The mean slope is one number over a heavy-tailed ratio distribution, so a
    handful of wild predictions can carry it either way. Counting where each
    observation falls is steadier and says the thing directly: on what share of
    steps does the prediction move by roughly the amount the cursor moved, and
    on what share does it not budge at all.
    """
    usable = [o for o in obs if abs(o["expect"]) > floor]
    if not usable:
        return {"n": 0, "frac_compensating": 0.0, "frac_ignoring": 0.0}
    ratios = [o["obs"] / o["expect"] for o in usable]
    return {
        "n": len(usable),
        "frac_compensating": sum(1 for r in ratios if 0.5 <= r <= 1.5) / len(ratios),
        "frac_ignoring": sum(1 for r in ratios if abs(r) < 0.1) / len(ratios),
    }


def _fit_of(obs: list[dict], ci: bool = False) -> dict:
    return _fit(
        [o["expect"] for o in obs],
        [o["obs"] for o in obs],
        groups=[o["row"] for o in obs] if ci else None,
    )


def score_rows(rows: list[dict]) -> dict:
    obs = observations(rows)
    n_both_move = len(obs) // 2
    n_changed = sum(1 for r in rows if r.get("pred_a", "") != r.get("pred_b", ""))
    by_pool = {
        pool: _fit_of([o for o in obs if o["pool"] == pool])
        for pool in sorted({o["pool"] for o in obs if o["pool"]})
    }
    aimed = [o for o in obs if o["aim_cosine"] is not None]
    return {
        "n_rows": len(rows),
        "n_both_move": n_both_move,
        "move_pair_rate": n_both_move / len(rows) if rows else 0.0,
        "prediction_changed_rate": n_changed / len(rows) if rows else 0.0,
        "pooled": _fit_of(obs, ci=True),
        "rates": compensation_rates(obs),
        "axis_x": _fit_of([o for o in obs if o["axis"] == "x"]),
        "axis_y": _fit_of([o for o in obs if o["axis"] == "y"]),
        "by_pool": by_pool,
        # Does it read the cursor at least when it is aiming well? If the slope
        # is flat here too, no amount of on-target aiming came from the pixels.
        "gold_aligned": _fit_of([o for o in aimed if o["aim_cosine"] > 0.5]),
        "gold_misaligned": _fit_of([o for o in aimed if o["aim_cosine"] <= 0.5]),
    }


def cmd_score(args) -> None:
    rows = [json.loads(x) for x in Path(args.rows_jsonl).read_text().splitlines() if x.strip()]
    t0 = time.time()
    res = score_rows(rows)
    scores = {
        "cursor_probe/slope": res["pooled"]["slope"],
        "cursor_probe/slope_ci_lo": res["pooled"]["slope_ci_lo"],
        "cursor_probe/slope_ci_hi": res["pooled"]["slope_ci_hi"],
        "cursor_probe/slope_robust": res["pooled"]["slope_robust"],
        "cursor_probe/r2": res["pooled"]["r2"],
        "cursor_probe/slope_x": res["axis_x"]["slope"],
        "cursor_probe/slope_y": res["axis_y"]["slope"],
        "cursor_probe/move_pair_rate": res["move_pair_rate"],
        "cursor_probe/prediction_changed_rate": res["prediction_changed_rate"],
        "cursor_probe/n_both_move": res["n_both_move"],
        "cursor_probe/slope_gold_aligned": res["gold_aligned"]["slope"],
        "cursor_probe/frac_compensating": res["rates"]["frac_compensating"],
        "cursor_probe/frac_ignoring": res["rates"]["frac_ignoring"],
    }
    write_result(
        Path(args.output_dir) / "result.json",
        task=args.task,
        scores=scores,
        params={"grid": GRID},
        inputs={"rows_jsonl": args.rows_jsonl, "n_rows": res["n_rows"]},
        n_samples=res["n_rows"],
        elapsed_s=int(time.time() - t0),
        extra=res,
    )
    print(json.dumps(scores, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_i = sub.add_parser("index", help="scrape cursor_before out of the rollout trajectories")
    p_i.add_argument("--trajectories", required=True)
    p_i.add_argument("--out", required=True)
    p_i.set_defaults(fn=cmd_index)

    p_b = sub.add_parser("build", help="write A/B screenshots + manifest")
    p_b.add_argument("--val_shard", required=True)
    p_b.add_argument("--cursor_index", required=True, help="json of 'task|step' -> [cx,cy,w,h]")
    p_b.add_argument("--out_dir", required=True)
    p_b.add_argument("--num_records", type=int, default=150)
    p_b.add_argument("--scan_records", type=int, default=1200)
    p_b.add_argument("--shift_min", type=int, default=200)
    p_b.add_argument("--shift_max", type=int, default=450)
    p_b.add_argument("--bg_tol", type=float, default=1.5)
    p_b.add_argument("--bg_hit_tol", type=float, default=0.002)
    p_b.add_argument("--prev_core_min", type=float, default=30.0)
    p_b.add_argument("--core_min", type=float, default=8.0)
    p_b.add_argument("--arrow_tol", type=float, default=15.0)
    p_b.add_argument("--sprite_crops", type=int, default=800)
    p_b.add_argument("--sprite_stride", type=int, default=3)
    p_b.add_argument("--seed", type=int, default=0)
    p_b.set_defaults(fn=cmd_build)

    p_r = sub.add_parser("run", help="replay both conditions against sglang")
    p_r.add_argument("--manifest", required=True)
    p_r.add_argument("--val_shard", required=True)
    p_r.add_argument("--out", required=True)
    p_r.add_argument("--base_url", default=None)
    p_r.add_argument("--model_path", default=None)
    p_r.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    p_r.add_argument("--port", type=int, default=0)
    p_r.add_argument("--mem_fraction_static", type=float, default=0.80)
    p_r.add_argument("--chunked_prefill_size", type=int, default=2048)
    p_r.add_argument("--api_key", default="probe")
    p_r.add_argument("--model", default="bc-probe")
    p_r.add_argument("--concurrency", type=int, default=16)
    p_r.add_argument("--max_tokens", type=int, default=2048)
    p_r.add_argument("--retries", type=int, default=5)
    p_r.set_defaults(fn=cmd_run)

    p_s = sub.add_parser("score", help="regress the predicted-delta shift")
    p_s.add_argument("--rows_jsonl", required=True)
    p_s.add_argument("--output_dir", required=True)
    p_s.add_argument("--task", default="bc_cursor_perturbation")
    p_s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    if args.mode == "run" and bool(args.base_url) == bool(args.model_path):
        ap.error("pass exactly one of --base_url or --model_path")
    args.fn(args)


if __name__ == "__main__":
    main()
