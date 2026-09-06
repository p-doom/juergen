# CUA action-parity micro-evaluation

This package runs one fixed 18-task suite against
`ordered_events_v3_relative_1000_grid_v1`. Each task has exactly four attempts
with seeds 41000–41003. The evaluator owns one local SGLang server and four
Desktop QEMU sessions, records Desktop's raw 1920×1080 q92 JPEG observations,
and uses the same four-completed-turn/160-character history renderer as CUA-Gym
training.

The Desktop dependency is pinned to
`1db6ae2499afc16d87dee15453a57042dff13f64`. Every checked-out session must
restore the `cua_micro_xcursor_v1` checkpoint; its setup repairs and verifies the
guest Xcursor assignment before the checkpoint is created.

The locked SGLang runtime requires FFmpeg 6 shared libraries on the host.

```bash
uv sync --locked
uv run --locked python -c \
  'import torchcodec; assert torchcodec.ffmpeg_major_version == 6'
uv run --locked python cua_micro_eval.py \
  --model-path /path/to/checkpoint \
  --desktop-image /path/to/osworld.qcow2 \
  --output-dir /path/to/new/output
```

`WANDB_PROJECT` enables W&B reporting. Provider, model-attestation, Desktop,
process, and configured W&B failures abort the run. A `completed.json` file is
written only after all 72 attempts succeed operationally.

Unit tests never launch a model or VM:

```bash
uv run --locked pytest -q
```
