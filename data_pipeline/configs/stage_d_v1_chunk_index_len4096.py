"""Stage D — build offline chunk index for the grain payload. Standalone.

Reads ``v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09``.
Output dataset version:
``v1_chunk_index_qwen3vl2b_len4096_msgs128_k0p4_5fps_360p_2026_04_09``.
"""

from pmanager.configs.schema import pipeline_task

PROJECT_REPO = "/fast/home/franz.srambical/data_pipeline"
# build_sft_chunk_index.py's --num_workers flag is a main-only feature
# (PR #24, merged 4ec13a9). Pin this stage at the main worktree so the
# feature branch's older signature doesn't break the run.
OMEGALAX_REPO = "/fast/home/franz.srambical/omegalax-main"
SOURCE_VERSION = "v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09"
DATASET_VERSION = "v1_chunk_index_qwen3vl2b_len4096_msgs128_k0p4_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "8:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 16

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "stage_d_chunk_index.py"
    cfg.entrypoint.args.omegalax_repo = OMEGALAX_REPO
    cfg.entrypoint.args.model_id = "Qwen/Qwen3-VL-2B-Instruct"
    cfg.entrypoint.args.processor = "Qwen/Qwen3-VL-2B-Instruct"
    cfg.entrypoint.args.max_length = 4096
    cfg.entrypoint.args.records_per_shard = 100_000
    cfg.entrypoint.args.num_workers = 16

    cfg.inputs.payload = {"kind": "dataset", "version": SOURCE_VERSION}

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""
    return cfg
