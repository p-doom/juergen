#!/usr/bin/env bash
#SBATCH --job-name=probe_move_dispatch
#SBATCH --partition=standard
#SBATCH --qos=low
#SBATCH --cpus-per-task=10
#SBATCH --mem=24GB
#SBATCH --time=01:30:00
#SBATCH --output=/fast/project/HFMI_SynergyUnit/yll/logs/probe_move_dispatch_%j.log
#SBATCH --error=/fast/project/HFMI_SynergyUnit/yll/logs/probe_move_dispatch_%j.log

set -uo pipefail

WT=/fast/project/HFMI_SynergyUnit/yll/.worktrees/juergen-oev3-cuagym
PY=$WT/.venv/bin/python
OUT=${PROBE_OUT:-/fast/project/HFMI_SynergyUnit/yll/osworld_probe/move_dispatch}

mkdir -p "$OUT"
cd "$OUT"

export PYTHONUNBUFFERED=1
export OSWORLD_ROOT=/fast/project/HFMI_SynergyUnit/yll/osworld-pinned
export OSWORLD_QCOW2=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/osworld_vm/Ubuntu.qcow2
export OSWORLD_QEMU_BIN=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/qemu/bin/qemu-system-x86_64-wrapped
export OSWORLD_VM_LOG_DIR=$OUT
export OSWORLD_VM_SMP=8
export OSWORLD_VM_MEM=8G

echo "node=$(hostname) kvm=$(ls -l /dev/kvm 2>&1)"

"$PY" "$WT/eval/probe_move_dispatch.py" --out "$OUT"
RC=$?
echo "probe exit=$RC"
exit $RC
