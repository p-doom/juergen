# `datasets/` — dataset construction from rollouts

One file: `convert.py`, the rollout → training-record converter, parameterized by
target grammar (`--codec <name>` → `grammars/<name>/` via `grammars.load`), with
the encoding delegated to `Codec.format`.

```bash
uv run python datasets/convert.py \
    --codec deltatype_v2 \
    --rollouts_dir /path/to/collect_v1,/path/to/collect_v2 \
    --out_dir /path/to/dataset \
    --min_task_success 1.0 --exclude_slugs eval_leak.txt
```

Output is `_normalized/{train,val}/chat.jsonl` + `convert_manifest.json`, which
is what omegalax's `scripts/build_sft_records_from_chat.py` consumes.

There is no `--coord_space`: the coordinate convention is a property of the
grammar you pick, resolved inside its codec, never named here.

## The seam

`convert.py` resolves each recorded step into the absolute-Operation vocabulary
documented in `grammars/_support.py` (`move_to`, `glide_to`, `mouse_down/up`,
`scroll`, `key_down/up`, `coalesced_type`, `wait`; every coordinate an absolute,
clamped screen pixel). The codec then lifts Operations into its own Action:

```python
def action_from_operations(self, operations, *, geometry, cursor, terminate=None) -> Action
```

the inverse of the `Codec.compile_action` the eval and RL path uses. It belongs
to the codec because the codec owns both the coordinate convention and the
terminal spelling — a flag on the Action for `deltatype_v2` / `diffabs` /
`ordered_events_v3`, a `terminate` call inside it for `native_absolute` /
`move_rel`. A grammar that has not implemented it gets a `NotImplementedError`
naming the method, not a guess.

## Two source vocabularies, one declared

`source_reader` keys on a declared field, never on the shape of the payload:

* our own harness records the grammar it ran, so `result.json` carries a `codec`
  field and each row carries the absolute Operation stream the harness
  dispatched plus the `control` verdict it resolved. That stream IS the seam, so
  it is read back rather than re-derived, and no source codec is loaded.
* the external absolute-teacher collections carry no `codec` field, and their
  rows carry the teacher's `computer_use` arguments plus `intended_target`, the
  post-scale post-clip pixel the VM acted on. The argument's own `coordinate` is
  on a normalized grid (`coord_grid = 1000`) and is **not** a pixel.

Sniffing rather than reading the declaration is a silent corruption either way
round: in one direction a source parse error on every step of our own rollouts,
in the other a 0..999 grid read as pixels. Nothing here asks a grammar to emit
`computer_use` — the estate keeps one action vocabulary, and it is Operations.

## Prose

Prose is kept by default; `--no_keep_prose` builds a tool-call-only arm.
`convert.py` records coverage in the manifest (`n_turns`, `n_turns_with_prose`,
`prose_turn_frac`, `n_turns_dropped_by_codec`) and refuses to write a
zero-coverage dataset. Because a zero floor cannot see `100% vs 5%`, it also
compares this arm's coverage against sibling arms in sibling `--out_dir`s that
record the same rollout selection, over the codec-invariant pre-codec turn
population, and warns above `--prose_divergence_tol` (default 0.05).
`--prose_divergence` selects `warn` (default), `abort` or `off`;
`--no_keep_prose` forces `off`, and naming both is an error. An absent or
mid-build sibling is skipped, so the first arm of a sweep is compared against
nothing.

## Naming hazard

This directory is not a Python package: there is no `__init__.py` and there must
never be one. `data_pipeline` depends on HuggingFace `datasets`, and a regular
package named `datasets` at the repo root would shadow it for anything that puts
the repo root on `sys.path` (which every `pipeline/stage_*.py` does). Run
`convert.py` as a script (`python datasets/convert.py`), never as
`python -m datasets.convert`.
