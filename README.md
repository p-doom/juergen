# Juergen

Juergen contains two SFT data streams and one evaluation suite:

- the canonical Crowd-Cast pipeline, annotated with `describe_extract` and
  trained with `deltatype_v2`;
- the CUA-Gym action-format parity pipeline, trained with the normalized
  relative `ordered_events_v3` grammar;
- the standalone CUA micro-eval under `eval/`.

Training is owned by Omegalax. Desktop/QEMU lifecycle and Slurm deployment are
separate repositories.

Run the root and pipeline tests with:

```bash
uv run --locked --extra dev pytest -q
uv run --project data_pipeline --locked pytest -q data_pipeline/tests
```

Run the micro-eval tests and evaluator from its independent lock:

```bash
uv run --project eval --locked pytest -q eval
uv run --project eval --locked python eval/cua_micro_eval.py \
  --model_path <hf-checkout> --output_dir <dir> --qemu_image <qcow2>
```

See `data_pipeline/README.md` and `eval/README.md` for the artifact contracts
and runtime options.
