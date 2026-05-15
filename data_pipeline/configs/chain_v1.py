"""Full 4-stage chain: prepare → run_length_cap → grain_payload → chunk_index.

Launching this fires stage A; the dispatcher auto-fires B on A's completion,
C on B's, D on C's — via cfg.children with trigger="on_complete".

Each stage is a regular pipeline_task; launching the chain is just stage A
with stage B nested as on_complete child, etc.
"""

from configs import (
    stage_a_v1_5fps_360p,
    stage_b_v1_run_length_cap_k0p4,
    stage_c_v1_grain_payload,
    stage_d_v1_chunk_index_len4096,
)


def _as_child(child_cfg, trigger: str = "on_complete") -> dict:
    d = child_cfg.to_dict()
    d["trigger"] = trigger
    return d


def get_config():
    # Build inside-out: D nested in C, C in B, B in A.
    stage_d = stage_d_v1_chunk_index_len4096.get_config()

    stage_c = stage_c_v1_grain_payload.get_config()
    stage_c.children = [_as_child(stage_d)]

    stage_b = stage_b_v1_run_length_cap_k0p4.get_config()
    stage_b.children = [_as_child(stage_c)]

    stage_a = stage_a_v1_5fps_360p.get_config()
    stage_a.children = [_as_child(stage_b)]

    return stage_a
