cd "$(git rev-parse --show-toplevel)"
# Qwen3.8-27B at the current default (fieldnorms2, budget 5) on all 100 claims (2026-11-06), after its
# server came back. Same run directory, subclaims and judge as run_fn2_b5_full.sh. Concurrency is kept
# low (4 shards x 2 workers): 18 requests in flight wedged the server on 2026-11-05.
# --merge and --add-passes make it safe to rerun after an interruption. Subclaims seeded from dataset50run.
R=results/dataset50run_fn2_b5
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
ALL=($(ls results/dataset50run_fn/eval/ground_truth | sed 's/problem_//'))
mkdir -p $R/logs $R/eval/ground_truth
for p in "${ALL[@]}"; do [ -d $R/eval/ground_truth/problem_$p ] || cp -R results/dataset50run_fn/eval/ground_truth/problem_$p $R/eval/ground_truth/; done
seed() {  # seed <dir>: every claim's stored subclaims, no arms (never overwrites)
  .venv/bin/python - "$1" <<'PY'
import json, glob, os, sys
for f in glob.glob("results/dataset50run/proposals/problem_*/scify_proposer.json"):
    out = f.replace("results/dataset50run/proposals", sys.argv[1])
    if os.path.exists(out): continue
    d = json.load(open(f)); d["arms"] = {}
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(d, open(out, "w"), indent=2)
PY
}
# Each group writes its OWN proposals dir (read-modify-write per claim).
propose() {  # propose <models> <shards> <workers> <tag> <out>
  local MS=$1 N=$2 W=$3 T=$4 OUT=$5 i j sub
  seed $OUT
  for ((i=0; i<N; i++)); do
    sub=(); for ((j=i; j<${#ALL[@]}; j+=N)); do sub+=("${ALL[$j]}"); done
    .venv/bin/python -u -m norm_cards.eval.propose --problems "$(IFS=,; echo "${sub[*]}")" --models $MS --arms nocard,card,rag \
      --prompt fieldnorms2 --cards results/dataset50run/cards --budget 5 --workers $W --merge --out $OUT \
      > $R/logs/propose_full_${T}_$i.log 2>&1 &
  done
  wait
}
judge() {  # judge <models> <shards> <source>
  local MS=$1 N=$2 SRC=$3 i j sub
  local A=$(for m in ${MS//,/ }; do printf "%s+nocard,%s+card,%s+rag," $m $m $m; done)
  for ((i=0; i<N; i++)); do
    sub=(); for ((j=i; j<${#ALL[@]}; j+=N)); do sub+=("${ALL[$j]}"); done
    .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(IFS=,; echo "${sub[*]}")" --arms "${A%,}" --source $SRC \
      --judge-runs 3 --add-passes --workers 3 --prompt v2 --model gpt-6-luna > $R/logs/judge_full_${MS//,/_}_$i.log 2>&1 &
  done
  wait
}
propose qwen3.8-27b 4 2 qwen $R/proposals_qwen
echo "[$(date +%H:%M)] qwen proposals done"
judge qwen3.8-27b 10 $R/proposals_qwen
echo "[$(date +%H:%M)] qwen judged: $(ls $R/eval/judgments/problem_*/qwen3.8-27b+*.json | grep -v trace | wc -l | tr -d ' ') judgments"
