#!/usr/bin/env python3
"""Extract PSAI MP4s onto the 2 fps, 540p BC frame grid."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
import tempfile

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2  # noqa: E402

BC_FPS = 2.0
BC_HEIGHT = 540
JPEG_Q = 90


def resize_h(img, target_h):
    h, w = img.shape[:2]
    if h <= target_h:
        return img
    new_w = int(round(w * target_h / h / 2) * 2)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def grid_times(duration, rate):
    n = max(int(duration * rate + 1e-6), 1)
    return [k / rate for k in range(n)]


def extract_video(video_path, out_task_dir, overwrite=False):
    meta_path = os.path.join(out_task_dir, "extract_meta.json")
    if os.path.exists(meta_path) and not overwrite:
        return "skip"
    if not os.path.exists(video_path):
        return "no_video"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unreadable_video"

    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_raw = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_raw = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(n_frames_raw) if math.isfinite(n_frames_raw) else 0
        width = int(width_raw) if math.isfinite(width_raw) else 0
        height = int(height_raw) if math.isfinite(height_raw) else 0

        duration = 0.0
        if n_frames > 0 and math.isfinite(native_fps) and native_fps > 0:
            duration = n_frames / native_fps
        if not (
            math.isfinite(native_fps)
            and native_fps > 0
            and n_frames > 0
            and math.isfinite(duration)
            and 0 < duration <= 7200
        ):
            return "bad_metadata"

        bc_ts = grid_times(duration, BC_FPS)
        bc_src = [
            min(int(round(t * native_fps)), n_frames - 1)
            for t in bc_ts
        ]
        want = {}
        for grid_idx, source_idx in enumerate(bc_src):
            want.setdefault(source_idx, []).append(grid_idx)

        bc_dir = os.path.join(out_task_dir, "bc_frames")
        os.makedirs(bc_dir, exist_ok=True)

        max_needed = max(want) if want else -1
        source_idx = 0
        written = 0
        last_ok = None
        while source_idx <= max_needed:
            ret, frame = cap.read()
            if not ret:
                break
            last_ok = frame
            if source_idx in want:
                bc_img = resize_h(frame, BC_HEIGHT)
                for grid_idx in want[source_idx]:
                    frame_path = os.path.join(
                        bc_dir, f"frame_{grid_idx:06d}.jpg"
                    )
                    cv2.imwrite(
                        frame_path,
                        bc_img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q],
                    )
                    written += 1
            source_idx += 1

        padded = 0
        if last_ok is not None:
            bc_img = resize_h(last_ok, BC_HEIGHT)
            for grid_idx in range(len(bc_ts)):
                frame_path = os.path.join(
                    bc_dir, f"frame_{grid_idx:06d}.jpg"
                )
                if not os.path.exists(frame_path):
                    cv2.imwrite(
                        frame_path,
                        bc_img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q],
                    )
                    padded += 1

        meta = {
            "native_fps": native_fps,
            "n_frames": n_frames,
            "width": width,
            "height": height,
            "duration": duration,
            "bc_fps": BC_FPS,
            "bc_height": BC_HEIGHT,
            "n_bc_frames": len(bc_ts),
            "bc_src_indices": bc_src,
            "written": {"bc": written},
            "padded": {"bc": padded},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return "ok"
    finally:
        cap.release()


def discover_tasks(videos_dir):
    tasks = []
    for basename in sorted(os.listdir(videos_dir)):
        video_path = os.path.join(videos_dir, basename)
        if os.path.isfile(video_path) and basename.endswith(".mp4"):
            tasks.append((os.path.splitext(basename)[0], video_path))
    return tasks


def write_summary(out_dir, shard_idx, stats):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(
        out_dir, f"extract_summary_shard_{shard_idx:03d}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, sort_keys=True)
    return summary_path


def run_shard(args):
    tasks = discover_tasks(args.videos_dir)
    tasks = [
        task
        for task_idx, task in enumerate(tasks)
        if task_idx % args.num_shards == args.shard_idx
    ]
    print(
        f"shard {args.shard_idx}/{args.num_shards}: {len(tasks)} tasks",
        flush=True,
    )

    stats = Counter()
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                extract_video,
                video_path,
                os.path.join(args.out_dir, unique_data_id),
                args.overwrite,
            ): unique_data_id
            for unique_data_id, video_path in tasks
        }
        done = 0
        for future in as_completed(futures):
            unique_data_id = futures[future]
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001
                status = "error"
                print(f"ERROR {unique_data_id}: {exc}", flush=True)
            stats[status] += 1
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(tasks)} {dict(stats)}", flush=True)

    summary_path = write_summary(args.out_dir, args.shard_idx, stats)
    print(f"DONE shard {args.shard_idx}: {dict(stats)}", flush=True)
    print(f"summary: {summary_path}", flush=True)


def self_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        videos_dir = os.path.join(temp_dir, "videos")
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(videos_dir)
        video_path = os.path.join(videos_dir, "tiny.mp4")
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 48),
        )
        assert writer.isOpened()
        black_path = os.path.join(temp_dir, "black.ppm")
        white_path = os.path.join(temp_dir, "white.ppm")
        ppm_header = b"P6\n64 48\n255\n"
        with open(black_path, "wb") as f:
            f.write(ppm_header + bytes(64 * 48 * 3))
        with open(white_path, "wb") as f:
            f.write(ppm_header + bytes([255]) * (64 * 48 * 3))
        test_frames = [cv2.imread(black_path), cv2.imread(white_path)]
        assert all(frame is not None for frame in test_frames)
        for frame_idx in range(20):
            writer.write(test_frames[frame_idx % 2])
        writer.release()

        task_out_dir = os.path.join(out_dir, "tiny")
        assert extract_video(video_path, task_out_dir) == "ok"
        with open(
            os.path.join(task_out_dir, "extract_meta.json"),
            encoding="utf-8",
        ) as f:
            meta = json.load(f)
        assert meta["n_bc_frames"] == 4
        for frame_idx in range(4):
            assert os.path.exists(
                os.path.join(
                    task_out_dir,
                    "bc_frames",
                    f"frame_{frame_idx:06d}.jpg",
                )
            )
    print("self_test: ok", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return args
    if not args.videos_dir or not args.out_dir:
        parser.error("--videos_dir and --out_dir are required")
    if args.num_shards <= 0:
        parser.error("--num_shards must be positive")
    if not 0 <= args.shard_idx < args.num_shards:
        parser.error("--shard_idx must satisfy 0 <= shard_idx < num_shards")
    if args.num_workers <= 0:
        parser.error("--num_workers must be positive")
    return args


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    run_shard(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
