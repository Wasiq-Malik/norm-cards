cd "$(git rev-parse --show-toplevel)"
set -e
# Older OpenAI proposers (gpt-5.6-terra, gpt-5.6-luna) on the 30-claim subset (6 per field).
# proposals_older is seeded with every claim's stored subclaims, so extending to all 100 later
# is the same propose command over the other 70 claims with --merge.
R=results/dataset50run; P=$R/proposals_older; MS=gpt-5.6-terra,gpt-5.6-luna
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
mkdir -p $R/logs
.venv/bin/python - <<'PY'
import json, glob, os
for f in glob.glob("results/dataset50run/proposals/problem_*/scify_proposer.json"):
    out = f.replace("/proposals/", "/proposals_older/")
    if os.path.exists(out): continue          # never clobber arms already proposed
    d = json.load(open(f)); d["arms"] = {}
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(d, open(out, "w"), indent=2)
PY
IFS=, read -r -a IDS < <(cat datasets/dataset50/build/subset30.txt; echo)
echo "${#IDS[@]} claims"
for i in 0 1 2 3 4 5; do
  sub=(); for ((j=i; j<${#IDS[@]}; j+=6)); do sub+=("${IDS[$j]}"); done
  .venv/bin/python -u -m norm_cards.eval.propose --problems "$(IFS=,; echo "${sub[*]}")" --models $MS --arms nocard,card,rag \
    --cards $R/cards --budget 9 --workers 6 --merge --out $P > $R/logs/propose_older_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] proposals done"
A=$(for m in ${MS//,/ }; do printf "%s+nocard,%s+card,%s+rag," $m $m $m; done)
for i in 0 1 2 3 4 5; do
  sub=(); for ((j=i; j<${#IDS[@]}; j+=6)); do sub+=("${IDS[$j]}"); done
  .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(IFS=,; echo "${sub[*]}")" --arms "${A%,}" --source $P \
    --judge-runs 3 --workers 3 --prompt v2 --model gpt-6-luna > $R/logs/judge_older_$i.log 2>&1 &
done
wait
n=0; for p in "${IDS[@]}"; do n=$((n + $(ls $R/eval/judgments/problem_$p/ 2>/dev/null | grep -cE '^gpt-5\.6-(terra|luna)\+(nocard|card|rag)\.json$'))); done
echo "[$(date +%H:%M)] judged $n/180"
