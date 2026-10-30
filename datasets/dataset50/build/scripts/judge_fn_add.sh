cd "$(git rev-parse --show-toplevel)"
# Bring field-norms judgments up to k=3: keeps each arm's existing pass(es), runs only the missing
# ones, majority-votes all; arms never judged get all 3. usage: judge_fn_add.sh <models> <shards> [source]
R=results/dataset50run_fn; MS=$1; N=${2:-10}; SRC=${3:-$R/proposals}
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
IDS=($(ls $R/eval/ground_truth | sed 's/problem_//'))
A=$(for m in ${MS//,/ }; do printf "%s+nocard,%s+card,%s+rag," $m $m $m; done)
for ((i=0; i<N; i++)); do
  sub=(); for ((j=i; j<${#IDS[@]}; j+=N)); do sub+=("${IDS[$j]}"); done
  .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(IFS=,; echo "${sub[*]}")" --arms "${A%,}" --source $SRC \
    --judge-runs 3 --add-passes --workers 3 --prompt v2 --model gpt-6-luna > $R/logs/judge_k3_${MS//,/_}_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] k=3 pass for $MS done"
