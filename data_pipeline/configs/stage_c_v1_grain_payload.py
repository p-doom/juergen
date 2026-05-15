"""Stage C — compile chat.jsonl into Grain payload shards. Standalone.

Reads ``v1_run_length_capped_k0p4_5fps_360p_2026_04_09``.
Output dataset version: ``v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09``.
"""

from pmanager.configs.schema import pipeline_task

PROJECT_REPO = "/fast/home/franz.srambical/data_pipeline"
OMEGALAX_REPO = "/fast/home/franz.srambical/omegalax"
SOURCE_VERSION = "v1_run_length_capped_k0p4_5fps_360p_2026_04_09"
DATASET_VERSION = "v1_grain_payload_msgs128_k0p4_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "2:00:00"
    cfg.resources.mem = "64GB"
    cfg.resources.cpus = 8

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "stage_c_grain_payload.py"
    cfg.entrypoint.args.omegalax_repo = OMEGALAX_REPO
    cfg.entrypoint.args.messages_per_record = 128
    cfg.entrypoint.args.records_per_shard = 10_000

    cfg.inputs.source = {"kind": "dataset", "version": SOURCE_VERSION}

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""
    return cfg
