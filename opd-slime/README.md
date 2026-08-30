# opd-slime

This directory builds and launches the pinned Slime stack used for on-policy
distillation jobs.

## Contents

- `scripts/build_slime_venv.sh` clones the pinned external repositories,
  applies the checked-in patches, builds the environment, and writes its
  manifest and ready marker.
- `scripts/run_opd.sh` launches one training job from explicit environment,
  student, teacher, prompt, output, and resource arguments.
- `opd_plugins/null_rm.py` provides the zero-reward function used by pure OPD.
- `patches/apex.patch` is the local Apex build patch.

The external repositories are not vendored. Their revisions and patch inputs
are defined by `build_slime_venv.sh` and recorded in the generated manifest.

## Build

```bash
bash scripts/build_slime_venv.sh --out_dir=/absolute/environment-root
```

This is intentionally not a uv project; the environment includes patched
CUDA extensions built from external repositories. `run_opd.sh` is invoked by
the dispatch recipe with all required arguments.

## Checks

```bash
bash -n scripts/build_slime_venv.sh scripts/run_opd.sh
ruff check opd_plugins
```
