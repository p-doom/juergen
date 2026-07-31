#!/usr/bin/env python3
"""Build the matched coalesced-vs-per-key synthetic typing factorial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import string
import sys
import uuid
from pathlib import Path
from typing import Any

COUNTS = {"train": 2000, "val": 200}
SEEDS = {"train": 2026073111, "val": 2026073112}
FORMATS = ("coalesced", "perkey")
PROSE = "I will type the requested text exactly."
BASE_SYSTEM = (
    "You operate a desktop computer. The text field shown is already focused. "
    "Reply with one short reasoning sentence, then exactly one action on the final line. "
    "Use raw-pixel relative deltatype syntax with a zero mouse delta. "
)
FORMAT_SYSTEM = {
    "coalesced": BASE_SYSTEM + 'Emit the requested printable text as one type("...") element.',
    "perkey": BASE_SYSTEM + "Emit every character as ordered key press and release events; do not use type(...).",
}


class BuildError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_parser(parser_dir: Path):
    expected = (parser_dir / "action_parser.py").resolve()
    if not expected.is_file():
        raise BuildError(f"production action parser missing: {expected}")
    sys.path.insert(0, str(parser_dir.resolve()))
    import action_parser  # type: ignore
    if Path(action_parser.__file__).resolve() != expected:
        raise BuildError(f"loaded wrong parser: {action_parser.__file__}")
    return action_parser


def key_tokens(text: str) -> list[str]:
    result: list[str] = []
    simple = {" ": "Space", "-": "Minus", ".": "Dot"}
    for character in text:
        if character in string.ascii_lowercase:
            key, shifted = f"Key{character.upper()}", False
        elif character in string.digits:
            key, shifted = f"Num{character}", False
        elif character in simple:
            key, shifted = simple[character], False
        else:
            raise BuildError(f"unsupported generated typing character: {character!r}")
        if shifted:
            result.extend(("+ShiftLeft", f"+{key}", f"-{key}", "-ShiftLeft"))
        else:
            result.extend((f"+{key}", f"-{key}"))
    return result


def decode_elements(elements: tuple) -> str:
    inverse = {"Space": " ", "Minus": "-", "Dot": "."}
    output: list[str] = []
    shift = False
    for kind, value in elements:
        if kind == "type":
            output.append(value)
            continue
        if value.what == "ShiftLeft":
            shift = value.kind == "press"
            continue
        if value.kind != "press":
            continue
        key = value.what
        if key.startswith("Key") and len(key) == 4:
            char = key[-1].lower()
        elif key.startswith("Num") and len(key) == 4:
            char = key[-1]
        elif key in inverse:
            char = inverse[key]
        else:
            raise BuildError(f"gold action contains undecodable key: {key}")
        output.append(char.upper() if shift else char)
    return "".join(output)


def action_line(fmt: str, text: str) -> str:
    if fmt == "coalesced":
        return "0 0 0 ; type(" + json.dumps(text, ensure_ascii=False) + ")"
    return "0 0 0 ; " + " ".join(key_tokens(text))


def render(path: Path, sample_number: int) -> None:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1000, 700), (242, 244, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 110, 880, 590), fill=(255, 255, 255), outline=(58, 68, 82), width=3)
    draw.text((160, 155), "Text entry task", fill=(24, 30, 40))
    draw.text((160, 210), "The input field below is focused.", fill=(60, 68, 80))
    draw.rectangle((160, 285, 840, 365), fill=(255, 255, 255), outline=(35, 105, 220), width=5)
    draw.line((184, 305, 184, 345), fill=(20, 20, 20), width=3)
    draw.text((160, 455), f"Task card {sample_number:04d}", fill=(105, 112, 124))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def reference_hashes(roots: list[Path]) -> tuple[set[str], set[str]]:
    images: set[str] = set()
    sample_ids: set[str] = set()
    for root in roots:
        if not root.exists():
            raise BuildError(f"reference root missing: {root}")
        for path in root.rglob("*.png"):
            images.add(sha(path))
        for path in root.rglob("*.jsonl"):
            if path.stat().st_size > 100_000_000:
                continue
            for line in path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("sample_id"):
                    sample_ids.add(str(row["sample_id"]))
    return images, sample_ids


def build(out: Path, parser_dir: Path, references: list[Path]) -> dict[str, Any]:
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise BuildError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    parser = load_parser(parser_dir.resolve())
    ref_images, ref_ids = reference_hashes([path.resolve() for path in references])
    stage = out / f".building_{os.getpid()}_{uuid.uuid4().hex}"
    stage.mkdir()
    all_texts: set[str] = set()
    split_texts: dict[str, set[str]] = {}
    split_images: dict[str, set[str]] = {}
    pair_hashes: dict[str, str] = {}
    try:
        for split, count in COUNTS.items():
            rng = random.Random(SEEDS[split])
            texts: list[str] = []
            while len(texts) < count:
                length = rng.randrange(6, 25)
                alphabet = string.ascii_lowercase + string.digits + " -."
                value = "".join(rng.choice(alphabet) for _ in range(length)).strip(" .-")
                if len(value) >= 4 and value not in all_texts:
                    texts.append(value)
                    all_texts.add(value)
            split_texts[split] = set(texts)
            image_hashes: set[str] = set()
            rows = {fmt: [] for fmt in FORMATS}
            pair_projection = []
            for index, text in enumerate(texts):
                sid = f"typing_{split}_{index:04d}"
                if sid in ref_ids:
                    raise BuildError(f"heldout sample-id collision: {sid}")
                image = stage / "images" / split / f"{sid}.png"
                visual_nonce = (0 if split == "train" else COUNTS["train"]) + index
                render(image, visual_nonce)
                image_hash = sha(image)
                if image_hash in ref_images or image_hash in image_hashes:
                    raise BuildError(f"heldout/within-split image collision: {sid}")
                image_hashes.add(image_hash)
                final_image = out / "images" / split / image.name
                user = f"Type this exact text into the focused field: {json.dumps(text)}"
                if any(token in (user + PROSE) for token in ("/fast/", "run_", "source_model")):
                    raise BuildError(f"provenance leak in visible text: {sid}")
                pair_projection.append({
                    "sample_id": sid, "target_text": text, "image_sha256": image_hash,
                    "user": user, "prose": PROSE,
                })
                for fmt in FORMATS:
                    line = action_line(fmt, text)
                    parsed = parser.parse_deltatype(line)
                    if parser.format_deltatype(parsed) != line:
                        raise BuildError(f"non-canonical {fmt} round trip: {sid}")
                    decoded = decode_elements(parsed.elements)
                    if decoded != text or parsed.dx or parsed.dy or parsed.scroll:
                        raise BuildError(f"typed-string execution mismatch: {fmt}/{sid}")
                    rows[fmt].append({
                        "sample_id": sid, "recording_id": sid, "scene_id": sid,
                        "target_text": text, "format": fmt,
                        "messages": [
                            {"role": "system", "content": [{"type": "text", "text": FORMAT_SYSTEM[fmt]}]},
                            {"role": "user", "content": [
                                {"type": "image", "image": str(final_image)},
                                {"type": "text", "text": user},
                            ]},
                            {"role": "assistant", "content": [{"type": "text", "text": PROSE + "\n" + line}]},
                        ],
                    })
            split_images[split] = image_hashes
            canonical = json.dumps(pair_projection, sort_keys=True, separators=(",", ":")).encode()
            pair_hashes[split] = hashlib.sha256(canonical).hexdigest()
            for fmt in FORMATS:
                write_jsonl(stage / fmt / "_normalized" / split / "chat.jsonl", rows[fmt])
            write_jsonl(stage / f"pairs_{split}.jsonl", pair_projection)
        if split_texts["train"] & split_texts["val"]:
            raise BuildError("train/validation target-text leak")
        if split_images["train"] & split_images["val"]:
            raise BuildError("train/validation image leak")
        report = {
            "status": "pass", "counts": COUNTS, "seeds": SEEDS,
            "formats": list(FORMATS), "identical_pair_projection_sha256": pair_hashes,
            "pair_fields": ["sample_id", "target_text", "image_sha256", "user", "prose"],
            "roundtrip_parse_format_execute": {"passing": 2 * sum(COUNTS.values()), "total": 2 * sum(COUNTS.values())},
            "typed_string_exact": {"passing": 2 * sum(COUNTS.values()), "total": 2 * sum(COUNTS.values())},
            "train_val_target_overlap": 0, "train_val_image_overlap": 0,
            "reference_image_overlap": 0, "reference_sample_id_overlap": 0,
            "reference_roots": [str(path.resolve()) for path in references],
            "action_parser_sha256": sha(parser_dir / "action_parser.py"),
            "prose_sha256": hashlib.sha256(PROSE.encode()).hexdigest(),
        }
        (stage / "typing_pairing_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        source = {
            "artifact_type": "synthetic_typing_factorial_source", "schema_version": 1,
            "status": "complete", "formats": list(FORMATS), "train_records_per_format": 2000,
            "validation_records_per_format": 200, "seeds": SEEDS,
            "pairing_report": "typing_pairing_report.json",
        }
        (stage / "typing_source_manifest.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
        for child in stage.iterdir():
            os.replace(child, out / child.name)
        stage.rmdir()
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parser-dir", required=True, type=Path)
    parser.add_argument("--reference", action="append", default=[], type=Path)
    args = parser.parse_args()
    try:
        result = build(args.out, args.parser_dir, args.reference)
    except BuildError as exc:
        print(f"FATAL typing factorial build: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
