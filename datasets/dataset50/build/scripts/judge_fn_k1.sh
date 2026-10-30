cd "$(git rev-parse --show-toplevel)"
# Single-vote screening judge for field-norms runs (compare against run 0 of the k=3 judgments).
# usage: judge_fn_k1.sh <models comma-separated> <claims file or "all">
R=results/dataset50run_fn; MS=$1; SRC=${4:-$R/proposals}
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
if [ "$2" = all ]; then IDS=($(ls $R/eval/ground_truth | sed 's/problem_//')); else IFS=, read -r -a IDS < <(cat $2; echo); fi
A=$(for m in ${MS//,/ }; do printf "%s+nocard,%s+card,%s+rag," $m $m $m; done)
N=${3:-10}
for ((i=0; i<N; i++)); do
  sub=(); for ((j=i; j<${#IDS[@]}; j+=N)); do sub+=("${IDS[$j]}"); done
  .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(IFS=,; echo "${sub[*]}")" --arms "${A%,}" --source $SRC \
    --judge-runs 1 --workers 1 --prompt v2 --model gpt-6-luna > $R/logs/judge_k1_${MS//,/_}_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] judged $(ls $R/eval/judgments/problem_*/*.json 2>/dev/null | grep -v trace | wc -l | tr -d ' ') arms in $R"
