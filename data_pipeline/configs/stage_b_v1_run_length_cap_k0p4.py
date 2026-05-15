"""Stage B — run-length-cap NO_OPs. Standalone-launchable.

Reads ``v1_raw_event_stream_5fps_360p_2026_04_09``.
Output dataset version: ``v1_run_length_capped_k0p4_5fps_360p_2026_04_09``.
"""

from pmanager.configs.schema import pipeline_task

PROJECT_REPO = "/fast/home/franz.srambical/data_pipeline"
SOURCE_VERSION = "v1_raw_event_stream_5fps_360p_2026_04_09"
DATASET_VERSION = "v1_run_length_capped_k0p4_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "1:00:00"
    cfg.resources.mem = "64GB"
    cfg.resources.cpus = 32

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "stage_b_run_length_cap.py"
    cfg.entrypoint.args.k_seconds = 0.4
    cfg.entrypoint.args.num_workers = 32

    cfg.inputs.source = {"kind": "dataset", "version": SOURCE_VERSION}

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""
    return cfg
