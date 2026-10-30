cd "$(git rev-parse --show-toplevel)"
# gpt-5.6-terra passed the subset rule under Field Norms (card - nocard +0.022 >= +0.02 on 30 claims):
# extend to all 100. --merge into proposals_g56 skips the 30 finished claims; the judge skips judged arms.
R=results/dataset50run_fn
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
ALL=($(ls $R/eval/ground_truth | sed 's/problem_//'))
N=8
for ((i=0; i<N; i++)); do
  sub=(); for ((j=i; j<${#ALL[@]}; j+=N)); do sub+=("${ALL[$j]}"); done
  .venv/bin/python -u -m norm_cards.eval.propose --problems "$(IFS=,; echo "${sub[*]}")" --models gpt-5.6-terra --arms nocard,card,rag \
    --prompt fieldnorms --cards results/dataset50run/cards --budget 9 --workers 9 --merge --out $R/proposals_g56 \
    > $R/logs/propose_g56x_$i.log 2>&1 &
done
wait; echo "[$(date +%H:%M)] terra extension proposed"
bash datasets/dataset50/build/scripts/judge_fn_k1.sh gpt-5.6-terra all 12 $R/proposals_g56
echo "[$(date +%H:%M)] terra extension judged"
