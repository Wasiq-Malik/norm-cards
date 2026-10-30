cd "$(git rev-parse --show-toplevel)"
# (X) k=3 for the finished proposers; (Y) extend gpt-5.6-luna to 100 claims (user: +0.019 counts as a pass),
# then k=3 for both 5.6 models. Y's judging waits for its proposals: proposals_g56 is rewritten while proposing.
R=results/dataset50run_fn
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
ALL=($(ls $R/eval/ground_truth | sed 's/problem_//'))
X() {
  bash datasets/dataset50/build/scripts/judge_fn_add.sh gpt-6-sol,gpt-6.1-sol,gpt-6.1-terra 16 &
  bash datasets/dataset50/build/scripts/judge_fn_add.sh qwen3.8-27b 6 $R/proposals_qwen &
  wait
}
Y() {
  N=8
  for ((i=0; i<N; i++)); do
    sub=(); for ((j=i; j<${#ALL[@]}; j+=N)); do sub+=("${ALL[$j]}"); done
    .venv/bin/python -u -m norm_cards.eval.propose --problems "$(IFS=,; echo "${sub[*]}")" --models gpt-5.6-luna --arms nocard,card,rag \
      --prompt fieldnorms --cards results/dataset50run/cards --budget 9 --workers 9 --merge --out $R/proposals_g56 \
      > $R/logs/propose_g56l_$i.log 2>&1 &
  done
  wait; echo "[$(date +%H:%M)] luna extension proposed"
  bash datasets/dataset50/build/scripts/judge_fn_add.sh gpt-5.6-terra,gpt-5.6-luna 8 $R/proposals_g56
}
X & Y & wait
echo "[$(date +%H:%M)] all done"
