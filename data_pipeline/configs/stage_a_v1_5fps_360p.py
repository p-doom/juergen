"""Stage A — extract frames + actions from S3 sync. Standalone-launchable.

Output dataset version: ``v1_raw_event_stream_5fps_360p_2026_04_09``
"""

from pmanager.configs.schema import pipeline_task

PROJECT_REPO = "/fast/home/franz.srambical/data_pipeline"
DATASET_VERSION = "v1_raw_event_stream_5fps_360p_2026_04_09"


def get_config():
    cfg = pipeline_task()
    cfg.name = DATASET_VERSION

    cfg.resources.n_gpus = 0
    cfg.resources.time = "6:00:00"
    cfg.resources.mem = "200GB"
    cfg.resources.cpus = 32

    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "stage_a_prepare.py"
    cfg.entrypoint.args.target_fps = 5
    cfg.entrypoint.args.target_height = 360
    cfg.entrypoint.args.jpeg_quality = 85
    cfg.entrypoint.args.train_ratio = 0.8
    cfg.entrypoint.args.val_ratio = 0.1
    cfg.entrypoint.args.seed = 0
    cfg.entrypoint.args.num_workers = 32
    cfg.entrypoint.args.max_segments = 0

    cfg.inputs.source = {
        "kind": "path",
        "path_per_cluster": {
            "berlin": "/fast/project/HFMI_SynergyUnit/p-doom/crowd-cast/crowd-cast-2026-04-09/uploads",
        },
    }

    cfg.output.dataset_version = DATASET_VERSION
    cfg.dataset = ""  # pipelines don't reference an input dataset version
    return cfg
