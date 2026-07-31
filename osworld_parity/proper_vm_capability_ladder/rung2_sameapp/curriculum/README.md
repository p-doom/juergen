# Same-app semantic curriculum scaffold

This is a small train/development-only bridge between isolated Phase-B mouse
skills and eventual full-VM parity. It stays on newly authored deterministic
Writer, Calc, Files, Chrome, and VS Code fixtures; it is not an OSWorld task
ingestion or offline-training pipeline.

## Contract

`semantic_task.schema.json` defines task identity independently of action
format. Each task pins a parameter seed, deterministic asset recipe/hash,
snapshot/reset strategy, 2–4 ordered semantic steps, semantic cursor milestones,
hidden final-state verifier, reset/near-miss/gold requirements, and a fixture
seal. `program.py` is a runtime bridge to the existing rung-2 symbolic compiler;
native absolute and compact raw-relative actions are never stored in task
identity.

The initial materialized matrix contains one train and one development seed for
each family:

| App | Family | Steps | Phase-B coverage | Explicit edge/thin labels |
|---|---|---:|---|---|
| Writer | replace, bold, save | 3 | click, type, hotkey | Ctrl+S |
| Calc | select, formula, confirm, save | 4 | click, type, hotkey | Ctrl+S |
| Files | select, drag, rename | 3 | click, drag, type, hotkey | file-drag (thin) |
| Chrome | navigate, signed scroll, toggle | 3 | click, vscroll | each sign (thin) |
| VS Code | focus, Unicode replace, save | 3 | click, type, hotkey | Unicode (thin), Ctrl+S |

Train Chrome scrolls down and development Chrome scrolls up, so the complete
materialized scaffold covers both signs. Horizontal scroll and timing-sensitive
double-click are frozen exclusions until their transport contracts are proven.
Unicode, real file drag, Ctrl+S, and every single-family thin case remain
visible in `coverage`; they are not silently promoted into broad evidence.

Every fixture must reject exact reset, reject its deterministic near miss,
accept gold, reproduce the same reset signature, run its final oracle in a new
process, and finish with zero held inputs. Writer/Calc/Files/Chrome delegate to
the existing same-app state oracles; VS Code delegates to the existing UTF-8
file oracle.

## Split boundary

Only `manifests/train.json` and `manifests/development.json` exist. The family
registry commits to a future sealed count per family, but assigns no sealed
seeds or inputs. `load_manifest("sealed_eval")` rejects before any file access.
No official task IDs, benchmark assets, GPU recipes, or training stages are in
this directory.

Run the local contract suite with:

```bash
python3 -m pytest -q \
  osworld_parity/proper_vm_capability_ladder/rung2_sameapp/curriculum/tests \
  osworld_parity/proper_vm_capability_ladder/rung2_sameapp/tests
```
