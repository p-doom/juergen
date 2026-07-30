"""Realigned-pipeline training chain STUB: 03 filter -> 04 conversations @x fps
-> 05 measure -> 06 records.

Follows the chain_v1 pattern (each stage nested as its parent's on_complete
child). The filter is shared with chain_annotate (same master + realigned
manifest family); the training fps X is independent of the annotation fps.
Goal-conditioned training: point ``GOALS_DIR`` at a finished chain_annotate
artifact (leave None for the goal-free path).

STUB status — adjust before launching:
  * PROJECT_REPO / MASTER_DIR / CLIPS_MANIFEST / OMEGALAX_REPO are placeholders
    for the ccast0618d family + your omegalax checkout (stage 05 subprocess-
    wraps its measure script).
  * pmanager injects ``--output_dir`` (and ``--source_path`` for 05/06); the
    03/04 input dirs are derived statically from the datasets root + version
    (the argparse stages accept ``--foo_bar=value`` via
    ``common.normalize_dashed_argv``).
"""

from pmanager.configs.schema import pipeline_task

PROJECT_REPO = "/fast/project/HFMI_SynergyUnit/yll/juergen/data_pipeline"
DATASETS_ROOT = "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/yll.kryeziu"
MASTER_DIR = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/alfred.nguyen/"
    "ccast0618d_dataset_full_master_fps_15_sharded"
)
CLIPS_MANIFEST = (
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/labctl/datasets/alfred.nguyen/"
    "ccast0618d_dataset_full_v1_stage_02_realign_manifest/clips_manifest.jsonl"
)
OMEGALAX_REPO = "/fast/project/HFMI_SynergyUnit/yll/omegalax"  # needs the measure script

TAG = "ccast0618d_v2"
TRAIN_FPS = 1.0
GOALS_DIR = None  # e.g. f"{DATASETS_ROOT}/{TAG}_stage_03b_goals_describe_extract_fps_0.5"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

FILTER_VERSION = f"{TAG}_stage_03_filter"
CONV_VERSION = f"{TAG}_stage_04_conversations_fps_{TRAIN_FPS}" + ("_goals" if GOALS_DIR else "")
MEASURE_VERSION = f"{TAG}_stage_05_measure_fps_{TRAIN_FPS}"
RECORDS_VERSION = f"{TAG}_stage_06_records_fps_{TRAIN_FPS}"


def _as_child(child_cfg, trigger: str = "on_complete") -> dict:
    d = child_cfg.to_dict()
    d["trigger"] = trigger
    return d


def stage_03_filter():
    # Shared with chain_annotate; imported lazily so each config stays
    # standalone-launchable.
    from configs.chain_annotate import stage_03_filter as shared  # noqa: PLC0415

    return shared()


def stage_04_conversations():
    cfg = pipeline_task()
    cfg.name = CONV_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "4:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "realigned_pipeline/stage_04_conversations.py"
    cfg.entrypoint.args.mode = "action"
    cfg.entrypoint.args.filter_dir = f"{DATASETS_ROOT}/{FILTER_VERSION}"
    cfg.entrypoint.args.clips_manifest = CLIPS_MANIFEST
    cfg.entrypoint.args.day_index_cache = f"{DATASETS_ROOT}/{CONV_VERSION}_day_index_cache.json"
    cfg.entrypoint.args.fps = TRAIN_FPS
    cfg.entrypoint.args.action_format = "canonical"
    cfg.entrypoint.args.num_workers = 32
    if GOALS_DIR:
        cfg.entrypoint.args.goals_dir = GOALS_DIR
        cfg.entrypoint.args.use_plans = True
        cfg.entrypoint.args.include_variants = True
        cfg.entrypoint.args.terminal_token = "<terminate>"
    cfg.inputs.source = {"kind": "dataset", "version": FILTER_VERSION}
    cfg.output.dataset_version = CONV_VERSION
    cfg.dataset = ""
    return cfg


def stage_05_measure():
    cfg = pipeline_task()
    cfg.name = MEASURE_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "6:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "realigned_pipeline/stage_05_measure_lengths.py"
    cfg.entrypoint.args.source_path = f"{DATASETS_ROOT}/{CONV_VERSION}"
    cfg.entrypoint.args.omegalax_repo = OMEGALAX_REPO
    cfg.entrypoint.args.model_id = MODEL_ID
    cfg.entrypoint.args.processor = MODEL_ID
    cfg.entrypoint.args.num_workers = 32
    cfg.inputs.source = {"kind": "dataset", "version": CONV_VERSION}
    cfg.output.dataset_version = MEASURE_VERSION
    cfg.dataset = ""
    return cfg


def stage_06_records():
    cfg = pipeline_task()
    cfg.name = RECORDS_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "6:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "realigned_pipeline/stage_06_training_records.py"
    cfg.entrypoint.args.source_path = f"{DATASETS_ROOT}/{CONV_VERSION}"
    cfg.entrypoint.args.message_lengths_path = f"{DATASETS_ROOT}/{MEASURE_VERSION}"
    cfg.entrypoint.args.omegalax_repo = OMEGALAX_REPO
    cfg.entrypoint.args.model_id = MODEL_ID
    cfg.entrypoint.args.processor = MODEL_ID
    cfg.entrypoint.args.max_length = 32768
    cfg.entrypoint.args.records_per_shard = 1024
    cfg.entrypoint.args.num_workers = 32
    cfg.entrypoint.args.val_fraction = 0.02
    cfg.inputs.source = {"kind": "dataset", "version": MEASURE_VERSION}
    cfg.output.dataset_version = RECORDS_VERSION
    cfg.dataset = ""
    return cfg


def get_config():
    # Build inside-out: 06 nested in 05, 05 in 04, 04 in 03.
    stage_06 = stage_06_records()
    stage_05 = stage_05_measure()
    stage_05.children = [_as_child(stage_06)]
    stage_04 = stage_04_conversations()
    stage_04.children = [_as_child(stage_05)]
    stage_03 = stage_03_filter()
    stage_03.children = [_as_child(stage_04)]
    return stage_03
