# Crowd-Cast RFT toolkit

This package contains the reusable, dependency-light control plane for rejection
fine-tuning experiments. It owns sample/score/record-building/train/evaluate
orchestration plus the guards around action grammars, arm parity, label leakage,
provenance, reward semantics, and evaluation parsing. Model training and serving
remain in OmegaLAX and the existing evaluation harnesses.

The labctl entry points are in `labctl/recipes/`. Run the CLI and tests from the
repository root with:

```bash
uv run --project rft -- python -m rft.cli --help
uv run --project rft --extra dev -- pytest -q rft/tests
```

The core package uses only the Python standard library. HTTP serving and offline
Weights & Biases access are optional extras and are imported lazily.
