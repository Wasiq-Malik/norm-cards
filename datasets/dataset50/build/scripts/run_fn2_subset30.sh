cd "$(git rev-parse --show-toplevel)"
# fieldnorms2 prompt trial on subset30 (2026-11-05): empty fields say None, the card arm's block is
# Field Norms and the rag arm's is Relevant Literature, each arm's system prompt explains only its
# own key. All six proposers, k=3 judge. Subclaims are seeded from dataset50run, the same ones the
# fieldnorms run used, so the two prompts compare claim by claim.
R=results/dataset50run_fn2
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
IFS=, read -r -a SUB < <(cat datasets/dataset50/build/subset30.txt; echo)
mkdir -p $R/logs $R/eval/ground_truth
for p in "${SUB[@]}"; do cp -R results/dataset50run_fn/eval/ground_truth/problem_$p $R/eval/ground_truth/; done
seed() {  # seed <dir>: subset30's stored subclaims, no arms (never overwrites)
  .venv/bin/python - "$1" datasets/dataset50/build/subset30.txt <<'PY'
import json, os, sys
for p in open(sys.argv[2]).read().strip().split(","):
    f = f"results/dataset50run/proposals/problem_{p}/scify_proposer.json"
    out = f.replace("results/dataset50run/proposals", sys.argv[1])
    if os.path.exists(out): continue
    d = json.load(open(f)); d["arms"] = {}
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(d, open(out, "w"), indent=2)
PY
}
# Each group writes its OWN proposals dir: groups merging into the same files at once would
# overwrite each other's arms (read-modify-write per claim).
propose() {  # propose <models> <shards> <workers> <tag> <out>
  local MS=$1 N=$2 W=$3 T=$4 OUT=$5 i j sub
  seed $OUT
  for ((i=0; i<N; i++)); do
    sub=(); for ((j=i; j<${#SUB[@]}; j+=N)); do sub+=("${SUB[$j]}"); done
    .venv/bin/python -u -m norm_cards.eval.propose --problems "$(IFS=,; echo "${sub[*]}")" --models $MS --arms nocard,card,rag \
      --prompt fieldnorms2 --cards results/dataset50run/cards --budget 9 --workers $W --merge --out $OUT \
      > $R/logs/propose_${T}_$i.log 2>&1 &
  done
  wait
}
judge() {  # judge <models> <shards> <source>
  local MS=$1 N=$2 SRC=$3 i j sub
  local A=$(for m in ${MS//,/ }; do printf "%s+nocard,%s+card,%s+rag," $m $m $m; done)
  for ((i=0; i<N; i++)); do
    sub=(); for ((j=i; j<${#SUB[@]}; j+=N)); do sub+=("${SUB[$j]}"); done
    .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(IFS=,; echo "${sub[*]}")" --arms "${A%,}" --source $SRC \
      --judge-runs 3 --add-passes --workers 3 --prompt v2 --model gpt-6-luna > $R/logs/judge_${MS//,/_}_$i.log 2>&1 &
  done
  wait
}
A() {
  propose gpt-6-sol,gpt-6.1-sol,gpt-6.1-terra 6 9 gpt6 $R/proposals
  echo "[$(date +%H:%M)] A: gpt-6 proposals done"
  judge gpt-6-sol,gpt-6.1-sol,gpt-6.1-terra 15 $R/proposals
  echo "[$(date +%H:%M)] A: gpt-6 judged"
}
C() {
  propose gpt-5.6-terra,gpt-5.6-luna 6 6 g56 $R/proposals_g56
  echo "[$(date +%H:%M)] C: gpt-5.6 proposals done"
  judge gpt-5.6-terra,gpt-5.6-luna 10 $R/proposals_g56
  echo "[$(date +%H:%M)] C: gpt-5.6 judged"
}
B() {
  propose qwen3.8-27b 6 3 qwen $R/proposals_qwen
  echo "[$(date +%H:%M)] B: qwen proposals done"
  judge qwen3.8-27b 10 $R/proposals_qwen
  echo "[$(date +%H:%M)] B: qwen judged"
}
A & B & C & wait
echo "[$(date +%H:%M)] all done: $(ls $R/eval/judgments/problem_*/*.json | grep -v trace | wc -l | tr -d ' ') judgments"
