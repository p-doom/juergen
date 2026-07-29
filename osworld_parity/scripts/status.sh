#!/bin/bash
# One-shot status probe for the on-policy DISTILLATION stream. Run at each
# re-engagement to see exactly where the afterok chain is + the metrics to act on.
# Usage: bash status.sh
ROOT=/fast/project/HFMI_SynergyUnit/p-doom_shared/franz/onpolicy_distill
CH=$ROOT/logs/chain_jobids.txt

echo "############ ON-POLICY DISTILLATION STREAM STATUS ############"
if [ -f "$CH" ]; then
  read -r COLLECT FILTER CONV ANNEAL EXPORT EVAL < "$CH"
  echo "chain jobids: collect=$COLLECT filter=$FILTER conv=$CONV anneal=$ANNEAL export=$EXPORT eval=$EVAL"
  for j in $COLLECT $FILTER $CONV $ANNEAL $EXPORT $EVAL; do
    st=$(sacct -j "$j" -n --format=State 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$st" ] && st=$(squeue -j "$j" -h -o "%T" 2>/dev/null)
    echo "  job $j : ${st:-UNKNOWN}"
  done
else
  echo "no chain_jobids.txt yet"
fi

echo; echo "===== COLLECT ====="
for idx in "$ROOT"/rollouts/*/index.json; do
  [ -f "$idx" ] || continue
  python3 - "$idx" <<'PY'
import json,sys,collections
d=json.load(open(sys.argv[1]))
runs=d.get("runs",[])
sr=collections.Counter(r.get("stop_reason","ERR") for r in runs)
print(f"  {sys.argv[1].split('rollouts/')[-1]}: n_runs={len(runs)} coord_grid={d.get('coord_grid')} stop_reasons={dict(sr)}")
PY
done

echo; echo "===== FILTER (keep-rate) ====="
for fr in "$ROOT"/filtered/*/filter_report.json; do
  [ -f "$fr" ] || continue
  python3 - "$fr" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
scores=[r.get("judge_score") for r in d.get("runs",[]) if r.get("judge_score") is not None]
avg=sum(scores)/len(scores) if scores else 0
print(f"  {sys.argv[1].split('filtered/')[-1]}: kept={d['n_kept']}/{d['n_total']} prefilter_pass={d['n_prefilter_pass']} min_score={d['min_score']} best_of_n={d['best_of_n']} avg_judge={avg:.1f}")
PY
done

echo; echo "===== CONVERT (dataset size) ====="
for cm in "$ROOT"/converted/*/convert_manifest.json; do
  [ -f "$cm" ] || continue
  python3 -c "import json,sys;d=json.load(open('$cm'));print('  '+'$cm'.split('converted/')[-1]+f\": n_train={d['n_train']} n_val={d['n_val']} kept={d['n_kept']} unusable={d['n_unusable']}\")"
done

echo; echo "===== ANNEAL (orbax steps written) ====="
for sd in "$ROOT"/checkpoints/anneal_*; do
  [ -d "$sd" ] || continue
  steps=$(ls -d "$sd"/[0-9]*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
  echo "  $(basename "$sd"): steps=[$steps]"
done

echo; echo "===== EVAL (reach vs ~1% baseline) ====="
for rj in "$ROOT"/eval/*/result.json "$ROOT"/eval/*/*/summary.json; do
  [ -f "$rj" ] || continue
  echo "  $rj:"; python3 -c "import json;d=json.load(open('$rj'));print('   ', {k:d[k] for k in list(d)[:12]})" 2>/dev/null || head -c 400 "$rj"
done
echo "  GIFs for MANUAL inspection: $ROOT/eval/*/**/rollout.gif  (does the cursor servo to targets now?)"
echo "##############################################################"
