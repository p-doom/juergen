"""Realigned-pipeline annotation chain STUB: 03 filter -> 03b annotate @k fps.

Follows the chain_v1 pattern (stage B nested as stage A's on_complete child).
Both stages read the SAME master store + realigned clips manifest family; the
annotation fps K is independent of any training fps (bounded only by the
master fps, integer stride).

STUB status — adjust before launching:
  * PROJECT_REPO / MASTER_DIR / CLIPS_MANIFEST point at the ccast0618d
    artifacts; swap for your dataset family.
  * pmanager injects ``--output_dir``; the downstream stage's ``--filter_dir``
    is derived statically from the datasets root + version here (the stages
    accept pmanager's ``--foo_bar=value`` arg form via
    ``common.normalize_dashed_argv``).
  * stage 03b spends LABELER tokens: it needs AZURE_OPENAI_ENDPOINT /
    AZURE_OPENAI_API_KEY in the job env and honest --target-tpm/--max-workers.
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

TAG = "ccast0618d_v2"
ANNOTATE_FPS = 0.5
FILTER_VERSION = f"{TAG}_stage_03_filter"
GOALS_VERSION = f"{TAG}_stage_03b_goals_describe_extract_fps_{ANNOTATE_FPS}"


def _as_child(child_cfg, trigger: str = "on_complete") -> dict:
    d = child_cfg.to_dict()
    d["trigger"] = trigger
    return d


def stage_03_filter():
    cfg = pipeline_task()
    cfg.name = FILTER_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "4:00:00"
    cfg.resources.mem = "128GB"
    cfg.resources.cpus = 32
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "realigned_pipeline/stage_03_filter.py"
    cfg.entrypoint.args.frames_master_dir = MASTER_DIR
    cfg.entrypoint.args.clips_manifest = CLIPS_MANIFEST
    cfg.entrypoint.args.num_workers = 32
    # Idle knobs (seconds; identical semantics on any master fps). These are
    # the stage defaults, restated for auditability: the legacy rounded NO_OP
    # predicate per 2 s bin, runs > 4 s thinned keeping 2 s ends — byte-
    # mirrors the pre-rewrite sampler's default (noop head/tail 1/1 @ 0.5 fps).
    cfg.entrypoint.args.idle_activity = "rounded"
    cfg.entrypoint.args.idle_judgment_bin_s = 2.0
    cfg.entrypoint.args.idle_min_duration_s = 4.0
    cfg.entrypoint.args.idle_keep_head_s = 2.0
    cfg.entrypoint.args.idle_keep_tail_s = 2.0
    cfg.inputs.source = {"kind": "path", "path_per_cluster": {"berlin": MASTER_DIR}}
    cfg.output.dataset_version = FILTER_VERSION
    cfg.dataset = ""
    return cfg


def stage_03b_annotate():
    cfg = pipeline_task()
    cfg.name = GOALS_VERSION
    cfg.resources.n_gpus = 0
    cfg.resources.time = "24:00:00"
    cfg.resources.mem = "64GB"
    cfg.resources.cpus = 16
    cfg.entrypoint.repo_paths = {"berlin": PROJECT_REPO}
    cfg.entrypoint.path = "realigned_pipeline/annotation/stage_annotate.py"
    cfg.entrypoint.args.filter_dir = f"{DATASETS_ROOT}/{FILTER_VERSION}"
    cfg.entrypoint.args.fps = ANNOTATE_FPS
    cfg.entrypoint.args.method = "describe_extract"
    cfg.entrypoint.args.models = "Kimi-K2.6"
    cfg.entrypoint.args.target_tpm = 1_800_000
    cfg.entrypoint.args.max_workers = 64
    cfg.inputs.source = {"kind": "dataset", "version": FILTER_VERSION}
    cfg.output.dataset_version = GOALS_VERSION
    cfg.dataset = ""
    return cfg


def get_config():
    stage_b = stage_03b_annotate()
    stage_a = stage_03_filter()
    stage_a.children = [_as_child(stage_b)]
    return stage_a
