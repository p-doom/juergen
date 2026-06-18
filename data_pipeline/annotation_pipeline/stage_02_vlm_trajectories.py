#!/usr/bin/env python3
"""Stage 02: two-pass VLM trajectory extraction.

Pass A (segment): each request covers a wide time window (default 240s) via
sparse frame sampling, so the VLM sees whole task arcs before drawing
boundaries. Overlapping windows plus an interval-merge step join tasks that
cross window edges.

Pass B (name): for each merged candidate segment, a focused request over
frames from exactly that interval produces the imperative instruction, a
completion verdict, and refined bounds. Pass C then verifies each named
trajectory with discrete grounded checks and sets a `verified` flag.

Only pass-B trajectories with an accepted completion verdict are written to
trajectories_raw.json; everything else lands in naming_rejected.json with a
reason.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from annotation_pipeline import config
from annotation_pipeline.common import (
    ensure_dir,
    extract_json_object,
    image_data_url,
    read_jsonl,
    write_json,
    write_jsonl,
)


TRUNCATED_END = {"truncated_end", "truncated_both"}
TRUNCATED_START = {"truncated_start", "truncated_both"}


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def resize_to_height(frame: Any, height: int) -> Any:
    if height <= 0 or frame.shape[0] == height:
        return frame
    scale = height / frame.shape[0]
    width = max(2, round((frame.shape[1] * scale) / 2) * 2)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def resize_to_max_width(frame: Any, width: int | None) -> Any:
    if width and width > 0 and frame.shape[1] > width:
        scale = width / frame.shape[1]
        return cv2.resize(
            frame,
            (width, max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def load_vlm_video_sources(manifest_path: Path | None) -> dict[str, Path] | None:
    if manifest_path is None:
        return None
    rows = read_jsonl(manifest_path)
    sources: dict[str, Path] = {}
    for row in rows:
        segment_id = str(row.get("segment_id", ""))
        video_path = row.get("video_path")
        if segment_id and video_path:
            sources[segment_id] = Path(video_path)
    if not sources:
        raise RuntimeError(f"No video sources found in manifest: {manifest_path}")
    return sources


def read_record_frame(
    record: dict[str, Any],
    raw_video_by_segment: dict[str, Path] | None,
    captures: dict[str, cv2.VideoCapture],
) -> Any:
    if raw_video_by_segment is None:
        frame = cv2.imread(record["image_path"])
        if frame is None:
            raise RuntimeError(f"could not read frame: {record['image_path']}")
        return frame

    segment_id = str(record.get("segment_id", ""))
    video_path = raw_video_by_segment.get(segment_id)
    if video_path is None:
        raise RuntimeError(f"No raw video source for segment_id={segment_id!r}")

    capture = captures.get(segment_id)
    if capture is None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open raw video: {video_path}")
        captures[segment_id] = capture

    source_frame_idx = int(record.get("source_frame_idx", -1))
    if source_frame_idx < 0:
        raise RuntimeError(f"Invalid source_frame_idx on frame record: {record}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {source_frame_idx} from {video_path}")
    return frame


def overlay_frame(
    record: dict[str, Any],
    width: int | None = None,
    raw_video_by_segment: dict[str, Path] | None = None,
    target_height: int = 0,
    captures: dict[str, cv2.VideoCapture] | None = None,
) -> Any:
    frame = read_record_frame(record, raw_video_by_segment, captures if captures is not None else {})
    if raw_video_by_segment is not None:
        frame = resize_to_height(frame, target_height)
    frame = resize_to_max_width(frame, width)
    height, frame_width = frame.shape[:2]
    label = (
        f"original_t={float(record['global_time_s']):.1f}s  "
        f"seg{int(record['segment_idx']):04d}  "
        f"frame={int(record['global_frame_idx'])}"
    )
    cv2.rectangle(frame, (0, 0), (frame_width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def render_frames(
    records: list[dict[str, Any]],
    output_dir: Path,
    max_width: int,
    jpeg_quality: int,
    raw_video_by_segment: dict[str, Path] | None = None,
    target_height: int = 0,
) -> list[Path]:
    ensure_dir(output_dir)
    image_paths: list[Path] = []
    captures: dict[str, cv2.VideoCapture] = {}
    try:
        for out_idx, record in enumerate(records):
            frame = overlay_frame(
                record,
                width=max_width,
                raw_video_by_segment=raw_video_by_segment,
                target_height=target_height,
                captures=captures,
            )
            image_path = output_dir / f"frame_{out_idx:06d}.jpg"
            cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            image_paths.append(image_path)
    finally:
        for capture in captures.values():
            capture.release()
    return image_paths


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------


def evenly(pool: list[Any], k: int) -> list[Any]:
    if k <= 0 or not pool:
        return []
    if len(pool) <= k:
        return list(pool)
    step = len(pool) / k
    return [pool[min(len(pool) - 1, int(i * step))] for i in range(k)]


def segment_windows(
    t_min: float, t_max: float, window_s: float, overlap_s: float
) -> list[tuple[float, float]]:
    stride = max(1.0, window_s - overlap_s)
    windows: list[tuple[float, float]] = []
    start = t_min
    while True:
        end = start + window_s
        windows.append((start, end))
        if end >= t_max:
            break
        start += stride
    return windows


def sample_window_frames(
    in_window: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    max_images: int,
) -> list[dict[str, Any]]:
    """Sparse-sample a window: one frame per time slot, preferring active frames."""
    if len(in_window) <= max_images:
        return in_window
    step = (end_s - start_s) / max_images
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for i in range(max_images):
        lo = start_s + i * step
        hi = lo + step
        slot = [
            j
            for j, record in enumerate(in_window)
            if lo <= float(record["global_time_s"]) < hi and j not in used
        ]
        if not slot:
            continue
        active = [j for j in slot if in_window[j]["action"] != "NO_OP"]
        j = active[len(active) // 2] if active else slot[len(slot) // 2]
        picked.append(in_window[j])
        used.add(j)
    return picked


def select_naming_frames(
    frames: list[dict[str, Any]], max_images: int
) -> list[dict[str, Any]]:
    """Pick frames for pass B: always first/last, prefer active frames in between."""
    n = len(frames)
    if n <= max_images:
        return frames
    picks: set[int] = {0, n - 1}
    budget = max_images - len(picks)
    non_noop = [i for i in range(1, n - 1) if frames[i]["action"] != "NO_OP"]
    picks |= set(evenly(non_noop, budget))
    remaining = max_images - len(picks)
    if remaining > 0:
        rest = [i for i in range(n) if i not in picks]
        picks |= set(evenly(rest, remaining))
    return [frames[i] for i in sorted(picks)]


# ---------------------------------------------------------------------------
# Candidate segment merging
# ---------------------------------------------------------------------------


def should_merge(
    a: dict[str, Any],
    b: dict[str, Any],
    junction_gap_s: float,
    min_overlap_frac: float,
) -> bool:
    a_start, a_end = float(a["start_time_s"]), float(a["end_time_s"])
    b_start, b_end = float(b["start_time_s"]), float(b["end_time_s"])
    overlap = min(a_end, b_end) - max(a_start, b_start)
    shorter = min(a_end - a_start, b_end - b_start)
    if shorter > 0 and overlap / shorter >= min_overlap_frac:
        return True
    gap = b_start - a_end
    return (
        -1.0 <= gap <= junction_gap_s
        and (
            str(a.get("boundary", "")) in TRUNCATED_END
            or str(b.get("boundary", "")) in TRUNCATED_START
        )
    )


def merge_candidate_segments(
    segments: list[dict[str, Any]],
    junction_gap_s: float,
    min_overlap_frac: float,
) -> list[dict[str, Any]]:
    """Join window-truncated and duplicate (overlap-region) candidates."""
    ordered = sorted(
        segments,
        key=lambda s: (float(s["start_time_s"]), float(s["end_time_s"])),
    )
    merged: list[dict[str, Any]] = []
    for seg in ordered:
        if merged and should_merge(merged[-1], seg, junction_gap_s, min_overlap_frac):
            prev = merged[-1]
            prev["end_time_s"] = max(float(prev["end_time_s"]), float(seg["end_time_s"]))
            prev["start_time_s"] = min(float(prev["start_time_s"]), float(seg["start_time_s"]))
            if str(seg.get("label", "")) not in str(prev.get("label", "")):
                prev["label"] = f"{prev.get('label', '')} | {seg.get('label', '')}".strip(" |")
            prev["source_windows"] = sorted(
                set(prev.get("source_windows", [])) | set(seg.get("source_windows", []))
            )
        else:
            merged.append(dict(seg))
    return merged


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def build_segment_prompt(
    recording_id: str,
    window_idx: int,
    n_windows: int,
    window_start_s: float,
    window_end_s: float,
    n_frames: int,
    recording_span: tuple[float, float],
) -> str:
    return (
        "You are segmenting a screen recording of real desktop work into "
        "atomic, goal-directed tasks for SFT data creation.\n\n"
        "Input: an ordered list of frames SPARSELY sampled from one time "
        "window of the recording, denser around user activity. Frames are NOT "
        "consecutive; use the overlaid `original_t=...s` label on each frame "
        "for every time you report.\n\n"
        "Segmentation rules:\n"
        "- A segment is one atomic user task: a visible goal someone could "
        "state as a single instruction before it starts, e.g. 'Compare the "
        "train/loss curves of the selected runs' or 'Open the settings page "
        "and enable dark mode'.\n"
        "- Prefer 15-180 seconds per segment. Do not pad: idle time, aimless "
        "scrolling, or reading without a clear objective belongs to no "
        "segment.\n"
        "- Segments may touch but must not overlap.\n"
        "- If a task is clearly still in progress at a window edge, report "
        "the visible part anyway and set \"boundary\" accordingly; a merge "
        "step joins it with the neighboring window.\n"
        "- Report fewer, high-quality segments rather than covering the "
        "whole time range.\n\n"
        "Return ONLY valid JSON, no markdown fences or commentary:\n"
        "{\n"
        '  "segments": [\n'
        "    {\n"
        '      "start_time_s": number,\n'
        '      "end_time_s": number,\n'
        '      "label": "short description of the task",\n'
        '      "boundary": "complete" | "truncated_start" | "truncated_end" | "truncated_both"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Recording id: {recording_id}\n"
        f"Window {window_idx + 1} of {n_windows}: original_t "
        f"{window_start_s:.1f}s to {window_end_s:.1f}s\n"
        f"Frames attached: {n_frames}\n"
        f"Recording total span: {recording_span[0]:.1f}s to {recording_span[1]:.1f}s\n"
    )


def build_naming_prompt(
    label: str,
    start_s: float,
    end_s: float,
    n_frames: int,
) -> str:
    return (
        "You are writing the instruction for ONE candidate task segment cut "
        "from a desktop screen recording, for instruction-following SFT.\n\n"
        "Input: ordered frames sampled from the segment; the first and last "
        "frames of the segment are included. Use the overlaid "
        "`original_t=...s` labels for all times.\n\n"
        f"Candidate segment label: {label!r}\n"
        f"Candidate span: original_t {start_s:.1f}s to {end_s:.1f}s "
        f"({n_frames} frames attached)\n\n"
        "Produce:\n"
        "- \"instruction\": ONE imperative user request that this segment "
        "fulfills, phrased as if given before the first frame. Be specific: "
        "name the application, page, object, and goal visible in the frames, "
        "e.g. 'In the W&B run table, sort by masked_norm and open the best "
        "run'. Never write generic text like 'complete the task' and never "
        "describe what happened - request it.\n"
        "- \"completed\": \"yes\" if the final frames clearly show the goal "
        "state reached, \"partial\" if work toward it is visible but "
        "unfinished, \"no\" otherwise.\n"
        "- \"refined_start_time_s\" / \"refined_end_time_s\": tight bounds of "
        "the task, taken from frame labels, staying within the candidate "
        "span.\n"
        "- \"reason\": one short sentence of visual evidence.\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "{\n"
        '  "instruction": "string",\n'
        '  "completed": "yes" | "partial" | "no",\n'
        '  "refined_start_time_s": number,\n'
        '  "refined_end_time_s": number,\n'
        '  "reason": "string"\n'
        "}\n"
    )


def build_verify_prompt(instruction: str, start_s: float, end_s: float, n_frames: int) -> str:
    return (
        "You are auditing a candidate training example cut from a screen "
        "recording. You see the ordered frames of ONE proposed task segment "
        "(first and last frames included; each has its true timestamp overlaid "
        "as `original_t=...s`) and the instruction a labeler wrote for it.\n\n"
        f"Proposed instruction: {instruction!r}\n"
        f"Segment span: original_t {start_s:.1f}s to {end_s:.1f}s ({n_frames} frames).\n\n"
        "Judge the segment against the frames ONLY. Answer each question with a "
        "strict boolean and one short sentence of visual evidence. Be skeptical: "
        "when the evidence is absent or ambiguous, answer false.\n"
        "- \"active\": Does the segment show goal-directed progress? true if the "
        "user takes an action (typing, clicking, running a command, submitting a "
        "request to an AI/coding agent, build, or tool) AND the screen changes "
        "toward the goal. This INCLUDES the user issuing a request and then "
        "waiting while an invoked agent/build/process visibly works or streams "
        "output toward the goal, and INCLUDES deliberately scrolling/reading "
        "through content to find or review information. false only when the "
        "screen is essentially static with no user action: idle desktop, black "
        "frames, or a frozen window.\n"
        "- \"action_visible\": Is the task's INITIATING action visible in these "
        "frames (request typed/submitted, command run, control clicked, content "
        "scrolled)? The action may be a single request that kicks off the work; "
        "you do NOT need to see continuous manual input. false only if no action "
        "is visible at all.\n"
        "- \"start_grounded\": Could this instruction be issued by someone seeing "
        "ONLY the first frame, with no knowledge of anything before it? false if "
        "it assumes context/apps/state not visible at the start.\n"
        "- \"end_reached\": Is the goal state the instruction describes visibly "
        "present in the LAST frame? For an agent/build task, true if the invoked "
        "work has visibly finished or produced its result by the last frame; "
        "false if it is still running, only partially done, or the result "
        "happens off-screen / after the span.\n"
        "- \"atomic\": Is this exactly ONE task, not several bundled together and "
        "not a fragment of a larger one?\n\n"
        "Return ONLY valid JSON, no markdown:\n"
        "{\n"
        '  "active": true|false,\n'
        '  "action_visible": true|false,\n'
        '  "start_grounded": true|false,\n'
        '  "end_reached": true|false,\n'
        '  "atomic": true|false,\n'
        '  "evidence": "string"\n'
        "}\n"
    )


# ---------------------------------------------------------------------------
# VLM calls
# ---------------------------------------------------------------------------


def call_vlm_frames(
    prompt: str,
    image_paths: list[Path],
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})

    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # sglang + Qwen3: keep reasoning off (verification supersedes it).
        "extra_body": {"chat_template_kwargs": {"enable_thinking": config.DEFAULT_ENABLE_THINKING}},
    }
    response = client.chat.completions.create(**kwargs)
    elapsed_s = time.perf_counter() - started
    message = response.choices[0].message
    text = message.content or ""
    if not text.strip():
        # If reasoning is ever enabled, the answer can land in the reasoning
        # channel with empty content; salvage it rather than failing the call.
        text = getattr(message, "reasoning_content", None) or ""
    usage = response.usage.model_dump() if response.usage else None
    return text, {"elapsed_s": elapsed_s, "usage": usage, "model": model, "base_url": base_url}


def call_with_retry_and_reuse(
    args: argparse.Namespace,
    prompt: str,
    image_paths: list[Path],
    response_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Call the VLM, persisting raw responses so reruns reuse paid output."""
    if response_path.exists() and not args.no_reuse_responses:
        try:
            return extract_json_object(response_path.read_text()), {"reused": True}
        except Exception:  # noqa: BLE001 - unparseable cached response; re-call
            pass

    last_error: str | None = None
    for attempt in range(args.vlm_retries + 1):
        try:
            raw_text, meta = call_vlm_frames(
                prompt=prompt,
                image_paths=image_paths,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout_s=args.timeout_s,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            response_path.write_text(raw_text)
            return extract_json_object(raw_text), meta
        except Exception as exc:  # noqa: BLE001 - log and retry
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"  attempt {attempt + 1} failed for {response_path.name}: {last_error}")
    return None, {"error": last_error}


# ---------------------------------------------------------------------------
# Explicit dry-run heuristic (plumbing only; stage 03 refuses these without --allow-heuristic)
# ---------------------------------------------------------------------------


def heuristic_trajectories(frame_records: list[dict[str, Any]]) -> dict[str, Any]:
    non_noop = [record for record in frame_records if record["action"] != "NO_OP"]
    if not non_noop:
        return {
            "recording_id": frame_records[0]["recording_id"] if frame_records else "",
            "trajectories": [],
            "annotation_source": "heuristic_no_non_noop",
        }

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in non_noop:
        if current and float(record["global_time_s"]) - float(current[-1]["global_time_s"]) > 20.0:
            groups.append(current)
            current = []
        current.append(record)
    if current:
        groups.append(current)

    trajectories: list[dict[str, Any]] = []
    for group in groups:
        start = max(0.0, float(group[0]["global_time_s"]) - 2.0)
        end = float(group[-1]["global_time_s"]) + 2.0
        while end - start > 180.0:
            trajectories.append(
                {
                    "start_time_s": round(start, 1),
                    "end_time_s": round(start + 150.0, 1),
                    "instruction": "PLACEHOLDER_HEURISTIC",
                    "confidence": 0.0,
                    "reason": "Heuristic active-action split; plumbing test only.",
                }
            )
            start += 150.0
        if end - start >= 8.0:
            trajectories.append(
                {
                    "start_time_s": round(start, 1),
                    "end_time_s": round(end, 1),
                    "instruction": "PLACEHOLDER_HEURISTIC",
                    "confidence": 0.0,
                    "reason": "Heuristic active-action group; plumbing test only.",
                }
            )
    return {
        "recording_id": frame_records[0]["recording_id"],
        "trajectories": trajectories,
        "annotation_source": "heuristic",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-records", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Stage 00 manifest. When provided, VLM frames are rendered from "
            "the original MP4s at --vlm-frame-height instead of reusing the "
            "stage 01 training JPEGs."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=config.vlm_base_url())
    parser.add_argument("--api-key", default=config.vlm_api_key())
    parser.add_argument(
        "--model",
        default=config.vlm_model(),
        help=(
            "OpenAI/Azure model or Azure deployment name. Defaults to "
            f"{config.DEFAULT_VLM_MODEL!r}; override with JUERGEN_ANNOTATION_VLM_MODEL if the "
            "Azure deployment has a different name."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    # Pass A: segmentation over sparse wide windows.
    parser.add_argument("--segment-window-s", type=float, default=config.DEFAULT_SEGMENT_WINDOW_S)
    parser.add_argument("--segment-overlap-s", type=float, default=config.DEFAULT_SEGMENT_OVERLAP_S)
    parser.add_argument(
        "--segment-image-max",
        type=int,
        default=config.DEFAULT_FRAME_IMAGE_MAX,
        help="Max images per pass-A request (Azure caps at 50).",
    )
    parser.add_argument("--segment-image-width", type=int, default=config.DEFAULT_SEGMENT_IMAGE_WIDTH)
    parser.add_argument("--min-segment-s", type=float, default=8.0)
    parser.add_argument("--junction-gap-s", type=float, default=10.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.5)
    # Pass B: instruction naming per candidate segment.
    parser.add_argument("--name-image-max", type=int, default=config.DEFAULT_NAME_IMAGE_MAX)
    parser.add_argument("--name-image-width", type=int, default=config.DEFAULT_NAME_IMAGE_WIDTH)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep trajectories whose completion verdict is 'partial'.",
    )
    # Shared VLM call settings.
    parser.add_argument(
        "--vlm-frame-height",
        type=int,
        default=config.DEFAULT_VLM_FRAME_HEIGHT,
        help="Height for Stage 02 VLM renders from the raw MP4s. Use <=0 for native size.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.environ.get("JUERGEN_ANNOTATION_VLM_TIMEOUT_S", "900")),
        help="Per-call timeout; override via JUERGEN_ANNOTATION_VLM_TIMEOUT_S (lower for local servers so hung requests fail fast).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("JUERGEN_ANNOTATION_VLM_MAX_TOKENS", "4096")),
        help="Completion token cap; override via JUERGEN_ANNOTATION_VLM_MAX_TOKENS (e.g. for thinking modes).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--vlm-retries", type=int, default=2)
    parser.add_argument(
        "--no-reuse-responses",
        action="store_true",
        help="Re-call the VLM even when a parseable response file already exists.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("JUERGEN_ANNOTATION_MAX_CONCURRENCY", "8")),
        help="In-flight VLM requests per pass, to keep all sglang DP replicas busy.",
    )
    return parser.parse_args()


def run_pass_a(
    args: argparse.Namespace,
    frame_records: list[dict[str, Any]],
    output_dir: Path,
    vlm_video_by_segment: dict[str, Path] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recording_id = str(frame_records[0]["recording_id"])
    t_min = float(frame_records[0]["global_time_s"])
    t_max = float(frame_records[-1]["global_time_s"])
    windows = segment_windows(t_min, t_max, args.segment_window_s, args.segment_overlap_s)

    # Windows with no keylog activity are skipped before any VLM call (cheap
    # pre-gate). The rest run concurrently to keep all sglang DP replicas busy.
    skipped: dict[int, dict[str, Any]] = {}
    tasks: list[tuple[int, float, float, list[dict[str, Any]], int]] = []
    for window_idx, (window_start, window_end) in enumerate(windows):
        in_window = [
            record
            for record in frame_records
            if window_start <= float(record["global_time_s"]) < window_end
        ]
        n_active = sum(1 for record in in_window if record["action"] != "NO_OP")
        if not in_window or n_active == 0:
            skipped[window_idx] = {
                "window_idx": window_idx, "skipped": "no_activity", "n_frames": len(in_window)
            }
        else:
            tasks.append((window_idx, window_start, window_end, in_window, n_active))

    def process(task: tuple[int, float, float, list[dict[str, Any]], int]):
        window_idx, window_start, window_end, in_window, n_active = task
        sampled = sample_window_frames(in_window, window_start, window_end, args.segment_image_max)
        image_paths = render_frames(
            sampled,
            output_dir / "pass_a_frames" / f"window_{window_idx:04d}",
            max_width=args.segment_image_width,
            jpeg_quality=args.jpeg_quality,
            raw_video_by_segment=vlm_video_by_segment,
            target_height=args.vlm_frame_height,
        )
        prompt = build_segment_prompt(
            recording_id=recording_id,
            window_idx=window_idx,
            n_windows=len(windows),
            window_start_s=window_start,
            window_end_s=window_end,
            n_frames=len(sampled),
            recording_span=(t_min, t_max),
        )
        parsed, meta = call_with_retry_and_reuse(
            args, prompt, image_paths,
            output_dir / f"pass_a_response_window_{window_idx:04d}.txt",
        )
        cands: list[dict[str, Any]] = []
        for seg in (parsed or {}).get("segments", []):
            try:
                start_s = float(seg["start_time_s"])
                end_s = float(seg["end_time_s"])
            except (KeyError, TypeError, ValueError):
                continue
            # Clamp hallucinated times to the window the VLM actually saw.
            start_s = max(window_start, min(start_s, window_end))
            end_s = max(window_start, min(end_s, window_end))
            if end_s - start_s < args.min_segment_s:
                continue
            cands.append({
                "start_time_s": round(start_s, 1),
                "end_time_s": round(end_s, 1),
                "label": str(seg.get("label", "")).strip(),
                "boundary": str(seg.get("boundary", "complete")),
                "source_windows": [window_idx],
            })
        meta = {
            "window_idx": window_idx,
            "span_s": [round(window_start, 1), round(window_end, 1)],
            "n_frames_sent": len(sampled),
            "n_active_in_window": n_active,
            **meta,
        }
        return window_idx, meta, cands

    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as pool:
        processed = {r[0]: (r[1], r[2]) for r in pool.map(process, tasks)}

    candidates: list[dict[str, Any]] = []
    window_metas: list[dict[str, Any]] = []
    for window_idx in range(len(windows)):
        if window_idx in skipped:
            window_metas.append(skipped[window_idx])
        else:
            meta, cands = processed[window_idx]
            window_metas.append(meta)
            candidates.extend(cands)
    return candidates, window_metas


def run_pass_b(
    args: argparse.Namespace,
    frame_records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    output_dir: Path,
    vlm_video_by_segment: dict[str, Path] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_verdicts = {"yes", "partial"} if args.allow_partial else {"yes"}

    def process(item: tuple[int, dict[str, Any]]):
        """Name (pass B) then verify (pass C) one merged segment. Returns
        (seg_idx, naming_meta|None, trajectory|None, rejected|None)."""
        seg_idx, seg = item
        start_s = float(seg["start_time_s"])
        end_s = float(seg["end_time_s"])
        in_span = [
            record
            for record in frame_records
            if start_s - 0.5 <= float(record["global_time_s"]) <= end_s + 0.5
        ]
        if len(in_span) < 2:
            return seg_idx, None, None, {"segment": seg, "reason": "too_few_frames_in_span"}

        naming_frames = select_naming_frames(in_span, args.name_image_max)
        image_paths = render_frames(
            naming_frames,
            output_dir / "pass_b_frames" / f"segment_{seg_idx:04d}",
            max_width=args.name_image_width,
            jpeg_quality=args.jpeg_quality,
            raw_video_by_segment=vlm_video_by_segment,
            target_height=args.vlm_frame_height,
        )
        parsed, meta = call_with_retry_and_reuse(
            args,
            build_naming_prompt(str(seg.get("label", "")), start_s, end_s, len(naming_frames)),
            image_paths,
            output_dir / f"pass_b_response_segment_{seg_idx:04d}.txt",
        )
        naming_meta = {"segment_idx": seg_idx, **meta}
        if parsed is None:
            return seg_idx, naming_meta, None, {"segment": seg, "reason": "vlm_call_failed", "meta": meta}

        instruction = str(parsed.get("instruction", "")).strip()
        completed = str(parsed.get("completed", "")).strip().lower()
        if not instruction:
            return seg_idx, naming_meta, None, {"segment": seg, "reason": "empty_instruction", "vlm": parsed}
        if completed not in allowed_verdicts:
            return seg_idx, naming_meta, None, {"segment": seg, "reason": f"completed_{completed or 'missing'}", "vlm": parsed}

        # Refined bounds must stay within the candidate span.
        try:
            refined_start = float(parsed.get("refined_start_time_s", start_s))
            refined_end = float(parsed.get("refined_end_time_s", end_s))
        except (TypeError, ValueError):
            refined_start, refined_end = start_s, end_s
        refined_start = max(start_s, min(refined_start, end_s))
        refined_end = max(start_s, min(refined_end, end_s))
        if refined_end - refined_start < args.min_segment_s:
            refined_start, refined_end = start_s, end_s

        # Pass C: verify the named trajectory on the same frames with discrete,
        # grounded yes/no checks. The accept rule (active AND action_visible AND
        # start_grounded AND end_reached) replaces the unreliable self-reported
        # confidence as stage 03's quality gate.
        verify_parsed, _ = call_with_retry_and_reuse(
            args,
            build_verify_prompt(instruction, refined_start, refined_end, len(naming_frames)),
            image_paths,
            output_dir / f"pass_c_verify_segment_{seg_idx:04d}.txt",
        )
        checks = verify_parsed or {}
        verified = bool(
            checks.get("active")
            and checks.get("action_visible")
            and checks.get("start_grounded")
            and checks.get("end_reached")
        )
        trajectory = {
            "start_time_s": round(refined_start, 1),
            "end_time_s": round(refined_end, 1),
            "instruction": instruction,
            "completed": completed,
            "reason": str(parsed.get("reason", "")),
            "segment_label": seg.get("label", ""),
            "segment_span_s": [seg["start_time_s"], seg["end_time_s"]],
            "source_windows": seg.get("source_windows", []),
            "verified": verified,
            "verify_checks": checks,
        }
        return seg_idx, naming_meta, trajectory, None

    with ThreadPoolExecutor(max_workers=max(1, args.max_concurrency)) as pool:
        results = sorted(pool.map(process, list(enumerate(segments))), key=lambda r: r[0])

    trajectories: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    naming_metas: list[dict[str, Any]] = []
    for _, naming_meta, trajectory, reject in results:
        if naming_meta is not None:
            naming_metas.append(naming_meta)
        if trajectory is not None:
            trajectories.append(trajectory)
        if reject is not None:
            rejected.append(reject)
    return trajectories, rejected, naming_metas


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frame_records = read_jsonl(args.frame_records)
    if not frame_records:
        raise RuntimeError(f"No frame records: {args.frame_records}")

    if args.dry_run:
        trajectories = heuristic_trajectories(frame_records)
        trajectories["dry_run"] = True
        trajectories["dry_run_reason"] = "explicit dry_run flag"
        write_json(output_dir / "trajectories_raw.json", trajectories)
        print(f"Wrote heuristic trajectories to {output_dir / 'trajectories_raw.json'}")
        return
    if not (args.base_url and args.api_key and args.model):
        raise RuntimeError("missing VLM endpoint/API key/model")

    if args.vlm_frame_height > 0 and args.manifest is None:
        raise RuntimeError(
            "Stage 02 needs --manifest to render VLM frames from raw MP4s at "
            "--vlm-frame-height. Use --vlm-frame-height 0 only if you intend "
            "to send the stage 01 training JPEGs."
        )

    vlm_video_by_segment = load_vlm_video_sources(args.manifest)
    vlm_frame_source = "raw_video_manifest" if vlm_video_by_segment is not None else "stage01_frames"

    candidates, window_metas = run_pass_a(args, frame_records, output_dir, vlm_video_by_segment)
    write_jsonl(output_dir / "pass_a_candidates.jsonl", candidates)

    merged = merge_candidate_segments(
        candidates,
        junction_gap_s=args.junction_gap_s,
        min_overlap_frac=args.min_overlap_frac,
    )
    write_jsonl(output_dir / "pass_a_merged_segments.jsonl", merged)

    trajectories, rejected, naming_metas = run_pass_b(
        args,
        frame_records,
        merged,
        output_dir,
        vlm_video_by_segment,
    )
    write_json(output_dir / "naming_rejected.json", rejected)

    result = {
        "recording_id": str(frame_records[0]["recording_id"]),
        "annotation_source": "vlm_two_pass",
        "media_used": "frames",
        "trajectories": sorted(
            trajectories,
            key=lambda item: (float(item["start_time_s"]), float(item["end_time_s"])),
        ),
        "response_meta": {
            "model": args.model,
            "base_url": args.base_url,
            "vlm_frame_source": vlm_frame_source,
            "vlm_frame_height": args.vlm_frame_height,
            "segment_image_width": args.segment_image_width,
            "name_image_width": args.name_image_width,
            "pass_a_windows": window_metas,
            "pass_b_segments": naming_metas,
        },
    }
    write_json(output_dir / "trajectories_raw.json", result)
    write_json(
        output_dir / "stage02_summary.json",
        {
            "n_frames": len(frame_records),
            "n_pass_a_windows": len(window_metas),
            "n_candidates": len(candidates),
            "n_merged_segments": len(merged),
            "n_trajectories": len(trajectories),
            "n_verified": sum(1 for t in trajectories if t.get("verified")),
            "n_naming_rejected": len(rejected),
            "segment_window_s": args.segment_window_s,
            "segment_overlap_s": args.segment_overlap_s,
            "vlm_frame_source": vlm_frame_source,
            "vlm_frame_height": args.vlm_frame_height,
            "segment_image_width": args.segment_image_width,
            "name_image_width": args.name_image_width,
            "allow_partial": bool(args.allow_partial),
        },
    )
    print(
        f"Pass A: {len(candidates)} candidates -> {len(merged)} merged segments; "
        f"Pass B: {len(trajectories)} named, {len(rejected)} rejected. "
        f"Wrote {output_dir / 'trajectories_raw.json'}"
    )


if __name__ == "__main__":
    main()
