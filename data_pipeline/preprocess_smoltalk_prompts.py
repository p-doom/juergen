"""Smoltalk2 chat.jsonl → prompts-only JSONL for slime OPD rollouts.

Consumes the output of ``prep_smoltalk2.py`` (or the on-policy variant): a
JSONL of ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
"_source": "..."}`` per line. Emits a JSONL of ``{"prompt": <user_text>,
"_source": ...}`` — drops the assistant turn entirely (OPD generates fresh
on-policy responses during training; the recorded assistant turn is unused).

Filters:
  - records with no user message are dropped
  - records whose templated user prompt exceeds ``--max_prompt_tokens`` under
    the target model's tokenizer are dropped (so we don't waste rollout budget
    on prompt-only OOL samples)

Mirrors slime's ``--prompt-data --input-key prompt --apply-chat-template``
contract: the output is a flat JSONL with one prompt per line, and slime
applies its own chat template at rollout time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        type=Path,
        required=True,
        help="Input chat.jsonl (with messages list per record)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory; writes train/prompts.jsonl + manifest.json",
    )
    ap.add_argument(
        "--model_id",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HF model_id whose tokenizer is used for length-filtering",
    )
    ap.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=2048,
        help="Drop records whose templated user prompt exceeds this many tokens",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = ap.parse_args()

    train_dir = args.out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "prompts.jsonl"

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    n_in = n_out = n_no_user = n_too_long = 0
    with args.in_path.open() as fin, out_path.open("w") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages") or []
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            if not user_msgs:
                n_no_user += 1
                continue
            prompt = "\n\n".join(m.get("content", "") for m in user_msgs).strip()
            if not prompt:
                n_no_user += 1
                continue
            tpl = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = tok(tpl, add_special_tokens=False)["input_ids"]
            if len(ids) > args.max_prompt_tokens:
                n_too_long += 1
                continue
            fout.write(
                json.dumps(
                    {"prompt": prompt, "_source": rec.get("_source", "")},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_out += 1
            if args.limit and n_out >= args.limit:
                break

    manifest = {
        "schema_version": 1,
        "in_path": str(args.in_path),
        "model_id": args.model_id,
        "max_prompt_tokens": args.max_prompt_tokens,
        "counts": {
            "read": n_in,
            "kept": n_out,
            "dropped_no_user_message": n_no_user,
            "dropped_too_long": n_too_long,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
