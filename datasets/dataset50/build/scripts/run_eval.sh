cd "$(git rev-parse --show-toplevel)"
set -e
# dataset50: bootstrap references, assemble per-claim cards, then propose and judge in parallel shards.
# Proposers gpt-6-sol / gpt-6.1-sol / gpt-6.1-terra (propose.DEFAULT_MODELS), decomposer and judge gpt-6-luna.
R=results/dataset50run
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
PROPOSE_SHARDS=${PROPOSE_SHARDS:-8}; JUDGE_SHARDS=${JUDGE_SHARDS:-20}
mkdir -p $R/logs
.venv/bin/python -m norm_cards.dataset bootstrap --dataset dataset50 --run $R > $R/logs/bootstrap.log 2>&1
.venv/bin/python -m norm_cards.dataset assemble --dataset dataset50 --run $R > $R/logs/assemble.log 2>&1
IDS=($(ls $R/eval/ground_truth | sed 's/problem_//'))
echo "[$(date +%H:%M)] ${#IDS[@]} claims"

shard() {  # shard <i> <n>: every n-th claim starting at i, comma-joined
  local i=$1 n=$2 out=() j
  for ((j=i; j<${#IDS[@]}; j+=n)); do out+=("${IDS[$j]}"); done
  (IFS=,; echo "${out[*]}")
}

# 1. proposals: each shard runs its claims' 9 (model, condition) arms at once; skip-if-done makes reruns resume
for ((i=0; i<PROPOSE_SHARDS; i++)); do
  .venv/bin/python -u -m norm_cards.eval.propose --problems "$(shard $i $PROPOSE_SHARDS)" \
    --arms nocard,card,rag --cards $R/cards --budget 9 --workers 9 --merge --out $R/proposals \
    > $R/logs/propose_$i.log 2>&1 &
done
wait
n=$(.venv/bin/python -c "
import json,glob; print(sum(len(json.load(open(p))['arms'])==9 for p in glob.glob('$R/proposals/problem_*/scify_proposer.json')))")
echo "[$(date +%H:%M)] proposals complete for $n/${#IDS[@]} claims"
[ "$n" = "${#IDS[@]}" ] || { echo "incomplete proposals — rerun this script to resume"; exit 1; }

# 2. judging: gpt-6-luna, prompt v2, 3 runs majority-voted; each shard judges its claims' 9 arms
A=$(for m in gpt-6-sol gpt-6.1-sol gpt-6.1-terra; do printf "%s+nocard,%s+card,%s+rag," $m $m $m; done)
for ((i=0; i<JUDGE_SHARDS; i++)); do
  .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(shard $i $JUDGE_SHARDS)" --arms "${A%,}" \
    --source $R/proposals --judge-runs 3 --workers 3 --prompt v2 --model gpt-6-luna \
    > $R/logs/judge_$i.log 2>&1 &
done
wait
n=$(ls $R/eval/judgments/problem_*/*.json 2>/dev/null | grep -v trace | wc -l | tr -d ' ')
echo "[$(date +%H:%M)] judged $n/$(( ${#IDS[@]} * 9 )) arms"
