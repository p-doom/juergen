from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_ROOT = REPO_ROOT / "data_pipeline" / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import synthetic_clip  # noqa: E402
from test_pipeline_end_to_end import Chain  # noqa: E402

from pipeline.cua_gym.stage_01_image_store import build_store  # noqa: E402
from pipeline.cua_gym.stage_04_build_conversations import (  # noqa: E402
    ImageIndex,
    build_episode_records,
    render_contract,
)
from pipeline.lib.image_store import read_jpeg_bytes  # noqa: E402
from pipeline.lib.omegalax import (  # noqa: E402
    isolated_subprocess_environment,
    omegalax_python,
)

_CONSUMER_TEST = r"""
import json
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer

from omegalax.data import collator_qwen3, qwen3_encoding

omegalax_root = Path(sys.argv[1]).resolve()
snapshot = Path(sys.argv[2]).resolve()
examples = json.loads(Path(sys.argv[3]).read_text())
assert Path(qwen3_encoding.__file__).resolve().is_relative_to(omegalax_root)
assert Path(collator_qwen3.__file__).resolve().is_relative_to(omegalax_root)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
for source, messages in examples.items():
    measure = qwen3_encoding.make_message_length_fn(tokenizer, processor)
    lengths = [measure(message)["length"] for message in messages]
    measured_length = sum(lengths)
    encoded = qwen3_encoding.encode_qwen_messages(
        messages, tokenizer=tokenizer, image_processor=processor, include_pixels=True
    )
    assert measured_length == len(encoded["input_ids"])
    assert int(encoded["loss_mask"].sum()) > 0
    assert len(encoded["image_grid_thw"]) > 0
    assert len(encoded["pixel_values"]) > 0
    offset = 0
    for message, length in zip(messages, lengths, strict=True):
        mask = encoded["loss_mask"][offset:offset + length]
        if message["role"] == "assistant" and message.get("loss", True):
            assert int(mask.sum()) > 0
        else:
            assert not mask.any()
        offset += length
    collator = collator_qwen3.VLMSFTCollator(tokenizer, measured_length + 16, processor)
    batch = collator([{"messages": messages}])
    np.testing.assert_array_equal(batch["token_ids_BT"][0, :measured_length], encoded["input_ids"])
    np.testing.assert_array_equal(batch["loss_mask_BT"][0, :measured_length], encoded["loss_mask"])
    assert batch["token_ids_BT"].shape == (1, measured_length + 16)
    assert np.all(batch["token_ids_BT"][0, measured_length:] == tokenizer.pad_token_id)
    assert not batch["loss_mask_BT"][0, measured_length:].any()
    assert np.all(batch["attention_mask_BT"][0, :measured_length] == 1)
    assert not batch["attention_mask_BT"][0, measured_length:].any()
    np.testing.assert_array_equal(batch["image_grid_thw"], encoded["image_grid_thw"])
    assert batch["pixel_values"].dtype == ml_dtypes.bfloat16
    np.testing.assert_array_equal(batch["pixel_values"], encoded["pixel_values"].astype(ml_dtypes.bfloat16))
print(json.dumps({name: len(messages) for name, messages in examples.items()}))
""".strip()


def _crowdcast_messages(chat: Path) -> list[dict]:
    candidates = []
    with chat.open(encoding="utf-8") as source:
        for line in source:
            messages = json.loads(line)["messages"]
            assistants = [message for message in messages if message["role"] == "assistant"]
            if len(assistants) > 1:
                candidates.append(messages)
    messages = min(candidates, key=lambda value: len(json.dumps(value)))
    assert all("loss" not in message for message in messages if message["role"] == "assistant")
    assert "TERMINATE" not in messages[-1]["content"][0]["text"]
    image_uri = next(
        part["image"]
        for message in messages
        for part in message["content"]
        if part["type"] == "image"
    )
    with Image.open(io.BytesIO(read_jpeg_bytes(image_uri))) as image:
        assert image.format == "JPEG"
        assert image.size == (synthetic_clip.FRAME_W, synthetic_clip.FRAME_H)
    return messages


def _cua_messages(tmp_path: Path) -> list[dict]:
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    shard = screenshots / "screenshots-0000.tar"
    with tarfile.open(shard, "w") as archive:
        for index, color in enumerate(((10, 20, 30), (30, 20, 10))):
            image = io.BytesIO()
            Image.new("RGB", (1920, 1080), color).save(image, format="PNG")
            payload = image.getvalue()
            info = tarfile.TarInfo(f"task/step_{index:03d}.png")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    image_store = tmp_path / "image-store"
    build_store(screenshots, image_store, workers=1)
    record = {
        "task_id": "runtime-cua",
        "instruction": "Complete the task and stop.",
        "app": "writer",
        "screen": [1920, 1080],
        "steps": [
            {
                "step": 0,
                "shard": shard.name,
                "member": "task/step_000.png",
                "reasoning": "The application is still loading.",
                "action": {"primitives": [], "no_op": True, "terminate": None},
            },
            {
                "step": 1,
                "shard": shard.name,
                "member": "task/step_001.png",
                "reasoning": "The requested result is complete.",
                "action": {"primitives": [], "no_op": True, "terminate": "success"},
            },
        ],
    }
    rows = build_episode_records(record, ImageIndex(image_store), render_contract(), Counter())
    messages = rows[-1]["messages"]
    assert messages[-1]["content"][0]["text"].endswith("NO_OP\nTERMINATE: success")
    assert any(
        message["role"] == "assistant" and message.get("loss") is False for message in messages
    )
    return messages


def test_stage04_examples_match_measurement_encoder_and_training_collator(
    tmp_path: Path, monkeypatch
) -> None:
    omegalax_root = Path(os.environ["OMEGALAX_REPO"]).resolve()
    snapshot = Path(os.environ["PROCESSOR_SNAPSHOT"]).resolve()
    monkeypatch.setattr(synthetic_clip, "FRAME_W", 1280)
    monkeypatch.setattr(synthetic_clip, "FRAME_H", 720)
    chain = Chain(tmp_path / "crowdcast")
    examples = {
        "crowd-cast": _crowdcast_messages(chain.conversations / "chat.jsonl"),
        "cua": _cua_messages(tmp_path),
    }
    examples_path = tmp_path / "examples.json"
    examples_path.write_text(json.dumps(examples), encoding="utf-8")
    result = subprocess.run(
        omegalax_python(
            omegalax_root,
            "-c",
            _CONSUMER_TEST,
            str(omegalax_root),
            str(snapshot),
            str(examples_path),
        ),
        cwd=omegalax_root,
        env=isolated_subprocess_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed["crowd-cast"] > 3
    assert observed["cua"] == 5
