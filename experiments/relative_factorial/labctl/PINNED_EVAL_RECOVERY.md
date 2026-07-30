# Pinned eval recovery

The four `*_pinned_019fb44a.toml` recipes bypass the terminal-failed
pipeline children in `pipeline_019fb44ae67d76e1b022d529a5a01123`. Each
recipe resolves one successfully registered absolute-export artifact by
alias, so it can be submitted as a standalone run after evaluator hardening
is complete. Do not reuse the failed child run IDs.

Validate all four recipes before submission:

```bash
for recipe in experiments/relative_factorial/labctl/recipes/*_pinned_019fb44a.toml; do
  labctl validate "$recipe"
done
```

For a relative cell, wait until its train run is `succeeded` and `labctl show`
lists a `model` output. Copy the matching stage-based eval recipe, give the
copy a unique pinned recipe/output name, and replace only its model input:

```toml
[inputs.model]
type = "artifact"
artifact = "<registered model alias>"
```

The expected aliases for the active factorial pipeline are:

| Cell | Expected model alias |
| --- | --- |
| relraw_act | `relative_factorial_relraw_act_r32_s750_v1_run_019fb44ae67d76e1b022d493c752f8d1` |
| relraw_pre | `relative_factorial_relraw_pre_r32_s750_v1_run_019fb44ae67d76e1b022d4bde293f353` |
| reltool_act | `relative_factorial_reltool_act_r32_s750_v1_run_019fb44ae67d76e1b022d4dbea28c4dd` |
| reltool_pre | `relative_factorial_reltool_pre_r32_s750_v1_run_019fb44ae67d76e1b022d4f39937e0d9` |

Keep the matching original eval settings: `deltatype_raw` or `move_rel`,
`preamble=false` for `act`, `preamble=true` for `pre`,
and `model_path = "{inputs.model.path}/hf"`. Validate each pinned copy before
submitting it. Artifact aliases are not considered ready merely because they
are predictable; confirm registration with `labctl show <train-run-id>`.
