"""Stage C: keyframe captioning from Stage B cleaned frames.

Reads Stage B's filtered chat_line.json files, samples every Nth frame as
a keyframe, and sends batched filmstrips to Kimi K2.6 (via Azure AI Foundry)
for per-keyframe captioning.

Input:  Stage B output dir (--source_path) containing <split>/<segment>/chat_line.json
Output: <output_dir>/keyframes_meta.json, captions.json
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from absl import app, flags
from openai import OpenAI

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Output directory for captions.", required=True)
flags.DEFINE_string("source_path", None, "Stage B output dir.", required=True)
flags.DEFINE_string("split", "train", "Split to process.")
flags.DEFINE_string(
    "recording_id", None,
    "Recording ID filter. If set, only process segments matching this ID.",
)
flags.DEFINE_integer(
    "keyframe_every_n", 75,
    "Take every Nth frame from the cleaned stream as a keyframe.",
)
flags.DEFINE_integer("target_height", 720, "Resize keyframes to this height for LLM input.")
flags.DEFINE_integer("batch_size", 15, "Keyframes per LLM captioning call.")
flags.DEFINE_string("model", "kimi-k2.6", "Azure AI Foundry model deployment name.")
flags.DEFINE_float("temperature", 0.3, "LLM temperature for captioning.")
flags.DEFINE_integer("max_tokens", 16384, "Max output tokens per captioning call.")
flags.DEFINE_boolean("dry_run", False, "Extract keyframes but skip LLM calls.")


def _extract_keyframes_from_stage_b(
    source_path: Path,
    split: str,
    recording_id: str | None,
    keyframe_every_n: int,
    target_height: int,
) -> list[dict]:
    """Sample keyframes from Stage B's cleaned frame stream."""
    import cv2

    split_dir = source_path / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"No split dir: {split_dir}")

    seg_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if recording_id:
        seg_dirs = [d for d in seg_dirs if recording_id in d.name]

    keyframes = []
    global_idx = 0

    for seg_dir in seg_dirs:
        chat_path = seg_dir / "chat_line.json"
        if not chat_path.exists():
            continue

        data = json.loads(chat_path.read_text())
        msgs = data["messages"]
        n_frames = len(msgs) // 2

        seg_name = seg_dir.name
        seg_idx_str = seg_name.rsplit("_seg", 1)[-1] if "_seg" in seg_name else "0"
        try:
            seg_idx = int(seg_idx_str)
        except ValueError:
            seg_idx = 0

        local_kept = 0
        for i in range(n_frames):
            user_msg = msgs[2 * i]
            img_path = user_msg["content"][0].get("image", "")
            if not img_path:
                global_idx += 1
                continue

            if local_kept % keyframe_every_n == 0:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    out_w = round(target_height * w / h)
                    resized = cv2.resize(img, (out_w, target_height))
                    keyframes.append({
                        "seg_idx": seg_idx,
                        "seg_name": seg_name,
                        "frame_idx_in_seg": i,
                        "global_frame_idx": global_idx,
                        "image": resized,
                        "source_path": img_path,
                    })

            local_kept += 1
            global_idx += 1

        n_kf_seg = sum(1 for kf in keyframes if kf["seg_idx"] == seg_idx)
        print(
            f"  {seg_name}: {n_frames} frames, {n_kf_seg} keyframes",
            flush=True,
        )

    return keyframes


def _save_keyframes(keyframes: list[dict], out_dir: Path) -> list[dict]:
    import cv2

    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, kf in enumerate(keyframes):
        fname = f"seg{kf['seg_idx']:02d}_kf{i:04d}.jpg"
        fpath = kf_dir / fname
        cv2.imwrite(str(fpath), kf["image"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        metadata.append({
            "idx": i,
            "path": str(fpath),
            "seg_idx": kf["seg_idx"],
            "seg_name": kf["seg_name"],
            "frame_idx_in_seg": kf["frame_idx_in_seg"],
            "global_frame_idx": kf["global_frame_idx"],
        })
    return metadata


def _img_to_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}"


def _build_caption_messages(batch: list[dict]) -> list[dict]:
    content_blocks = []
    content_blocks.append({
        "type": "text",
        "text": (
            f"These are {len(batch)} chronological screenshots from a screen recording "
            f"of a software engineer working. They are sampled roughly every 5 seconds.\n\n"
            f"For each screenshot (numbered 1-{len(batch)}):\n"
            f"1. Describe what application is visible and what content is on screen "
            f"(read text, code, terminal output, URLs, file names — be specific).\n"
            f"2. Describe what the user appears to be doing.\n"
            f"3. If the activity changed from the previous screenshot, note the transition.\n\n"
            f"Be concise — one or two sentences per screenshot. Use this format:\n"
            f"[1] <description>\n"
            f"[2] <description>\n"
            f"..."
        ),
    })

    for kf in batch:
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": _img_to_data_url(kf["path"])},
        })

    return [{"role": "user", "content": content_blocks}]


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _call_llm(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    content = msg.content or ""
    if not content:
        content = getattr(msg, "reasoning_content", "") or ""
    return _strip_think(content)


def _run_captioning(
    client: OpenAI,
    kf_metadata: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    batch_size: int,
) -> list[dict]:
    batches = [
        kf_metadata[i : i + batch_size]
        for i in range(0, len(kf_metadata), batch_size)
    ]
    all_captions = list(kf_metadata)

    for bi, batch in enumerate(batches):
        print(f"  captioning batch {bi + 1}/{len(batches)} ({len(batch)} keyframes)...", flush=True)
        t0 = time.time()
        messages = _build_caption_messages(batch)
        raw = _call_llm(client, messages, model, temperature, max_tokens)
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s, {len(raw)} chars", flush=True)

        caption_map = {}
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                try:
                    num = int(line[1:bracket_end])
                    caption_map[num] = line[bracket_end + 1:].strip()
                except ValueError:
                    continue

        for j, kf in enumerate(batch):
            kf_global_idx = kf["idx"]
            caption = caption_map.get(j + 1, f"(no caption for frame {j + 1})")
            all_captions[kf_global_idx]["caption"] = caption

    return all_captions


def main(_) -> None:
    output_dir = Path(FLAGS.output_dir)
    source_path = Path(FLAGS.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting keyframes from Stage B output at {source_path}...", flush=True)
    keyframes = _extract_keyframes_from_stage_b(
        source_path,
        split=FLAGS.split,
        recording_id=FLAGS.recording_id,
        keyframe_every_n=FLAGS.keyframe_every_n,
        target_height=FLAGS.target_height,
    )
    print(f"Extracted {len(keyframes)} keyframes", flush=True)

    if not keyframes:
        print("No keyframes extracted. Exiting.", flush=True)
        return

    print("Saving keyframes to disk...", flush=True)
    kf_metadata = _save_keyframes(keyframes, output_dir)
    del keyframes

    meta_path = output_dir / "keyframes_meta.json"
    meta_path.write_text(json.dumps(kf_metadata, indent=2))
    print(f"Saved {len(kf_metadata)} keyframe metadata to {meta_path}", flush=True)

    if FLAGS.dry_run:
        print("Dry run: skipping LLM calls.", flush=True)
        return

    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Set AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_API_KEY or use --dry_run"
        )
    client = OpenAI(base_url=endpoint, api_key=api_key)

    print(f"\n=== Captioning ({len(kf_metadata)} keyframes, batch_size={FLAGS.batch_size}) ===", flush=True)
    captions = _run_captioning(
        client, kf_metadata, FLAGS.model, FLAGS.temperature,
        FLAGS.max_tokens, FLAGS.batch_size,
    )
    captions_path = output_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2))
    print(f"Saved {len(captions)} captions to {captions_path}", flush=True)


if __name__ == "__main__":
    app.run(main)
