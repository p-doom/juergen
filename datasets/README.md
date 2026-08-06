# `datasets/` — dataset construction from rollouts

One file: `convert.py`, the single ABSOLUTE-teacher-rollout → training-record
converter, parameterized by target grammar (`--codec <name>` → `grammars/<name>/`
via `grammars.load`). It replaces
`convert_abs_to_{absolute,relative,moverel,diffabs,deltatype}.py` (1,389 LOC) with
the shared machinery written once and the encoding delegated to `Codec.format`.

```bash
uv run python datasets/convert.py \
    --codec deltatype_v2 \
    --rollouts_dir /path/to/collect_v1,/path/to/collect_v2 \
    --out_dir /path/to/dataset \
    --min_task_success 1.0 --exclude_slugs eval_leak.txt
```

Output is `_normalized/{train,val}/chat.jsonl` + `convert_manifest.json`, i.e.
byte-compatible with what omegalax's `build_sft_records_from_chat.py` consumes —
unchanged from the five originals.

There is no `--coord_space`: the coordinate convention is a property of the grammar
you pick, resolved inside its codec, never named here.

## The seam

`convert.py` lowers each teacher step into the **absolute-Operation vocabulary**
documented in `grammars/_support.py` (`move_to`, `glide_to`, `mouse_down/up`,
`scroll`, `key_down/up`, `coalesced_type`, `wait`; every coordinate an absolute,
clamped screen pixel). The codec then lifts Operations into its own Action:

```python
def action_from_operations(self, operations, *, geometry, cursor, terminate=None) -> Action
```

which is the inverse of the `Codec.compile_action` the eval/RL path already uses.
It belongs to the codec because the codec owns both the coordinate convention and
the terminal spelling (a flag on the Action for `deltatype_v2` / `diffabs` /
`ordered_events_v3`, a `terminate` call inside it for `native_absolute` /
`move_rel`). `convert.py` raises `NotImplementedError` naming this method if a
grammar has not implemented it, rather than guessing.

## Prose

**Prose is kept by default.** `--no_keep_prose` exists but must be a deliberate
choice: the previous generation's two bare-token converters had no prose flag at
all, so the relative arms trained with zero reasoning supervision while the
absolute control had full reasoning supervision. Measured over the three arms
built from `rollouts/teacher_8b_osworld_train_v1` (2026-08-05): absolute
10721/10721 assistant turns prose-bearing, `deltatype_raw` 0/11337, `diffabs`
0/11102.

`convert.py` records prose coverage in the manifest (`n_turns`,
`n_turns_with_prose`, `prose_turn_frac`, `n_turns_dropped_by_codec`) and refuses
to write a zero-coverage dataset. Because a zero floor cannot see `100% vs 5%`,
it *also* compares this arm's coverage against sibling arms in sibling
`--out_dir`s that record the same rollout selection, over the codec-invariant
pre-codec turn population, and warns above `--prose_divergence_tol` (default
0.05). `--prose_divergence` picks what a divergence does — `warn` (default),
`abort`, or `off` — in one flag, so `abort` cannot be silently ignored by a
second flag that turned the comparison off. `--no_keep_prose` forces `off`, and
naming both is an error. The comparison is recorded in the manifest under
`prose_divergence`.
An absent or mid-build sibling is skipped, never a failure — so the first arm of
a sweep is compared against nothing.

## Naming hazard

This directory is deliberately **not** a Python package — there is no
`__init__.py` and there must never be one. `data_pipeline` depends on
HuggingFace `datasets`, and a regular package named `datasets` at the repo root
would shadow it for anything that puts the repo root on `sys.path` (which every
`pipeline/stage_*.py` does). Without `__init__.py` this directory can only ever be
a namespace portion, and a real installed `datasets` package always wins the
import. Run `convert.py` as a script (`python datasets/convert.py`), never as
`python -m datasets.convert`.
