# Eval checkout patches

Patches for **vendored** code that lives outside this repo (the OSWorld checkout
at `$OSWORLD_ROOT`), kept here so the fix is reviewable and versioned without
forking upstream.

## `qwen3vl_agent_sampling.patch` — kill the dead-flag trap

**Bug (inference-params audit, agent a398f7b).** The vendored OpenAI backend in
`mm_agents/qwen3vl_agent.py` comments out the sampling params it is handed:

```python
# qwen3vl_agent.py:646-648
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=self.max_tokens,
    # temperature=self.temperature,   <-- DEAD
    # top_p=self.top_p,               <-- DEAD
)
```

So the `--temperature` / `--top_p` flags on `osworld_one_task_runner.py` and
`osworld_fullbench_runner.py` (the "native" OSWorld runners, which route through
this agent) are **silently ignored** — the model always samples at the
checkpoint's `generation_config` defaults, and `top_k` / `repetition_penalty` /
`presence_penalty` are never sent at all.

**What the patch does.** Uncomments `temperature` / `top_p`, adds
`top_k` / `repetition_penalty` / `presence_penalty` constructor params (default
`None` = unset), and forwards the latter three via `extra_body` (the stock
OpenAI schema rejects them at the top level; sglang forwards `extra_body` to the
sampler). This makes the runners' Qwen-recommended tuple (see
`eval/sampling.py`) actually reach the sampler.

**Apply it** against your OSWorld checkout:

```bash
cd "$OSWORLD_ROOT"
git apply -p1 /path/to/juergen/eval/patches/qwen3vl_agent_sampling.patch
#   ... or, in a non-git checkout:
patch -p1 < /path/to/juergen/eval/patches/qwen3vl_agent_sampling.patch
```

Until it is applied, the juergen native runners **feature-detect** the
unpatched constructor and log a loud warning
(`sampling._UNPATCHED_AGENT_WARNING`) rather than crashing — they still pass the
params the stock constructor accepts, but those remain dead on the create call
until you patch, so the warning means "this run sampled at the gen-config
default regardless of your flags."

The patch is validated to apply cleanly (`git apply -p1 --check`) against the
checkout that was current when it was written; if upstream drifts and the hunks
no longer match, re-generate it against your checkout.

## Out of this repo: `grounding_reach_eval.py` (RL repo)

The same audit flagged `grounding_reach_eval.py:79` (the grounding-reach RL eval)
hardcoding `temperature=1.0` — the *Thinking* temperature on a non-thinking
Instruct model, which should be `0.7`. That file lives in the
`p-doom/reinforcement-learning` repo, not here, so it is **not** fixed by this
PR; the fix is the one-line `1.0 -> 0.7` change (or, better, import the shared
`sampling.qwen_sampling("instruct").temperature`). Tracked as a cross-repo
follow-up.
