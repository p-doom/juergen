"""Smoke chain: same as chain_v1 but with max_segments=10 for fast testing.

Launches stage A on 10 segments only. The chain still fires through B, C, D
via on_complete. Useful for end-to-end pmanager validation without burning
~17h of cluster time on a full rebuild.
"""

from configs import (
    stage_a_v1_5fps_360p,
    stage_b_v1_run_length_cap_k0p4,
    stage_c_v1_grain_payload,
    stage_d_v1_chunk_index_len4096,
)

VERSION_TAG = "smoke_v0"


def _as_child(child_cfg, trigger: str = "on_complete") -> dict:
    d = child_cfg.to_dict()
    d["trigger"] = trigger
    return d


def _smoke_a():
    cfg = stage_a_v1_5fps_360p.get_config()
    cfg.name = f"{VERSION_TAG}_raw"
    cfg.output.dataset_version = f"{VERSION_TAG}_raw"
    cfg.entrypoint.args.max_segments = 10
    cfg.entrypoint.args.num_workers = 8
    cfg.resources.cpus = 8
    cfg.resources.mem = "32GB"
    cfg.resources.time = "0:30:00"
    return cfg


def _smoke_b():
    cfg = stage_b_v1_run_length_cap_k0p4.get_config()
    cfg.name = f"{VERSION_TAG}_capped"
    cfg.output.dataset_version = f"{VERSION_TAG}_capped"
    cfg.inputs.source = {"kind": "dataset", "version": f"{VERSION_TAG}_raw"}
    cfg.entrypoint.args.num_workers = 4
    cfg.resources.cpus = 4
    cfg.resources.mem = "16GB"
    cfg.resources.time = "0:15:00"
    return cfg


def _smoke_c():
    cfg = stage_c_v1_grain_payload.get_config()
    cfg.name = f"{VERSION_TAG}_payload"
    cfg.output.dataset_version = f"{VERSION_TAG}_payload"
    cfg.inputs.source = {"kind": "dataset", "version": f"{VERSION_TAG}_capped"}
    cfg.resources.cpus = 4
    cfg.resources.mem = "16GB"
    cfg.resources.time = "0:30:00"
    return cfg


def _smoke_d():
    cfg = stage_d_v1_chunk_index_len4096.get_config()
    cfg.name = f"{VERSION_TAG}_index"
    cfg.output.dataset_version = f"{VERSION_TAG}_index"
    cfg.inputs.payload = {"kind": "dataset", "version": f"{VERSION_TAG}_payload"}
    cfg.entrypoint.args.num_workers = 2  # min allowed by build_sft_chunk_index
    cfg.resources.cpus = 4
    cfg.resources.mem = "32GB"
    cfg.resources.time = "0:30:00"
    return cfg


def get_config():
    d = _smoke_d()
    c = _smoke_c()
    c.children = [_as_child(d)]
    b = _smoke_b()
    b.children = [_as_child(c)]
    a = _smoke_a()
    a.children = [_as_child(b)]
    return a
