cd "$(git rev-parse --show-toplevel)"
# Field Norms prompt roll-out (single-vote judge). Group A on OpenAI, group B (Qwen) on Zeblok, in
# parallel. --merge into the same proposals files: claims a model already finished are skipped, and
# the judge skips proposals it has already judged, so nothing finished is recomputed.
R=results/dataset50run_fn
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
ALL=($(ls $R/eval/ground_truth | sed 's/problem_//'))
IFS=, read -r -a SUB < <(cat datasets/dataset50/build/subset30.txt; echo)
# Each group writes its OWN proposals dir: groups merging into the same files at once would
# overwrite each other's arms (read-modify-write per claim).
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
seed $R/proposals_qwen; seed $R/proposals_g56
propose() {  # propose <models> <shards> <workers> <tag> <out> <ids...>
  local MS=$1 N=$2 W=$3 T=$4 OUT=$5; shift 5; local IDS=("$@") i j sub
  for ((i=0; i<N; i++)); do
    sub=(); for ((j=i; j<${#IDS[@]}; j+=N)); do sub+=("${IDS[$j]}"); done
    .venv/bin/python -u -m norm_cards.eval.propose --problems "$(IFS=,; echo "${sub[*]}")" --models $MS --arms nocard,card,rag \
      --prompt fieldnorms --cards results/dataset50run/cards --budget 9 --workers $W --merge --out $OUT \
      > $R/logs/propose_${T}_$i.log 2>&1 &
  done
  wait
}
groupA() {
  propose gpt-6-sol,gpt-6.1-sol,gpt-6.1-terra 8 9 gpt6 $R/proposals "${ALL[@]}" &
  propose gpt-5.6-terra,gpt-5.6-luna 6 6 g56 $R/proposals_g56 "${SUB[@]}" &
  wait; echo "[$(date +%H:%M)] A: proposals done"
  bash datasets/dataset50/build/scripts/judge_fn_k1.sh gpt-6-sol,gpt-6.1-sol,gpt-6.1-terra all 16 &
  bash datasets/dataset50/build/scripts/judge_fn_k1.sh gpt-5.6-terra,gpt-5.6-luna datasets/dataset50/build/subset30.txt 6 $R/proposals_g56 &
  wait; echo "[$(date +%H:%M)] A: judged"
}
groupB() {
  propose qwen3.8-27b 8 3 qwen $R/proposals_qwen "${ALL[@]}"
  echo "[$(date +%H:%M)] B: qwen proposals done"
  bash datasets/dataset50/build/scripts/judge_fn_k1.sh qwen3.8-27b all 10 $R/proposals_qwen
  echo "[$(date +%H:%M)] B: qwen judged"
}
groupA & groupB & wait
echo "[$(date +%H:%M)] all done: $(ls $R/eval/judgments/problem_*/*.json | grep -v trace | wc -l | tr -d ' ') judgments"
