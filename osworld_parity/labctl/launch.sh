#!/bin/bash
# ============================================================================
# labctl-native OSWorld-parity SFT -- launch guide (VERIFIED on this cluster).
#
# RUN FROM THE LOGIN NODE (hai-login2) where the registry is reachable:
#   ssh hai-login2   (the PG registry is a host-local unix socket on the login
#   node; from a compute node `labctl doctor` shows registry connect = FAIL).
#
# CORRECTED DATASET MODEL (an earlier draft was wrong):
#   * There is NO "external" artifact kind on this cluster. Valid kinds are only
#     dataset | checkpoint | eval_result | environment (cluster.filesystem.
#     artifact_roots). `register-external --kind external` ERRORS.
#   * register-external CANONICALIZES its --path, so it can only register a real
#     directory that physically lives UNDER an artifact_root. A symlink from the
#     datasets root to an outside dir is resolved to its target and REJECTED
#     ("not under artifact root"). => you cannot "alias in" the converted/ dirs.
#   * THEREFORE: **no dataset registration is needed OR possible for these.**
#     A training recipe's  [inputs.dataset] type = "external"  resolves the raw
#     absolute path at `labctl run` time with artifact_id=null -- VERIFIED: the
#     real fmt_sft_8b_osw_diffabs_v1 recipe dispatched, resolved
#     converted/osworld_train_diffabs, and ran end-to-end. This is exactly how
#     the team's videocua_diffabs_v1 / videocua_moverel suites already work
#     (unregistered external paths).
#
#   * To have a format dataset TRACKED as a first-class artifact (optional,
#     provenance only -- NOT required to dispatch), it must be PRODUCED under the
#     datasets root by a recipe whose [outputs.dataset] type = "dataset". That is
#     what the build recipe osw_tokenize_fmt_records_deltatype_raw_v1 does; clone
#     it per format to promote the others. register-external is NOT the tool here.
#
# Everything below is additive; it does not touch the in-flight raw runs.
# ============================================================================
set -uo pipefail
RECIPES=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill/labctl/recipes

echo "### 0. sanity (must be run on hai-login2):"
echo "labctl doctor | grep 'registry connect'      # expect [OK]"
echo
echo "### 1. dataset registration: NONE NEEDED. type=external inputs resolve raw"
echo "###    paths at run time. (Skip straight to dispatch.)"
echo
echo "### 2. dispatch a format-SFT run -- reproduces the raw fmt_sft_8b.sbatch,"
echo "###    checkpoints now labctl-owned/tracked. Guard: do NOT launch"
echo "###    deltatype_norm while the raw job 134193 is still running."
for f in diffabs moverel absolute diffabsnorm deltatype_norm; do
  echo "labctl run $RECIPES/training/fmt_sft_8b_osw_${f}_v1.toml"
done
echo
echo "### 3. deltatype_raw only -- build its tokenized dataset NATIVELY first"
echo "###    (produces a tracked kind=dataset artifact under the datasets root),"
echo "###    then train from it:"
echo "labctl run $RECIPES/data_pipeline/osw_tokenize_fmt_records_deltatype_raw_v1.toml"
echo "labctl run $RECIPES/training/fmt_sft_8b_osw_deltatype_raw_v1.toml"
echo
echo "### 4. per-checkpoint HF export -- activate the policy (input is type=checkpoint,"
echo "###    so it is policy-driven, not a manual 'labctl run'):"
echo "cp $RECIPES/../policies_templates/export_per_ckpt_fmt_sft_8b_osw_diffabs_v1.toml \\"
echo "   /fast/home/franz.srambical/slurm/dev/franz/berlin/crowd-cast-bc/labctl/policies/"
echo "labctl evald once"
echo
echo "### 5. track:  labctl status   |   labctl show <run_id>"
