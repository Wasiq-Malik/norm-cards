cd "$(git rev-parse --show-toplevel)"
set -e
# Ablation: the card built from the ORIGINAL template-query search, everything else identical
# (same claims, same stored subclaims, same 3 proposers, budget 9). Answers whether the
# LLM-query + relevance-screen search made better cards.
R=results/dataset50run; P=$R/proposals_tmpl
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
mkdir -p $P $R/logs
.venv/bin/python - <<'PY'
import json, glob, os
for f in glob.glob("results/dataset50run/proposals/problem_*/scify_proposer.json"):
    d = json.load(open(f)); d["arms"] = {}
    out = f.replace("/proposals/", "/proposals_tmpl/"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(d, open(out, "w"), indent=2)
PY
IDS=($(ls $R/eval/ground_truth | sed 's/problem_//'))
shard() { local i=$1 n=$2 out=() j; for ((j=i; j<${#IDS[@]}; j+=n)); do out+=("${IDS[$j]}"); done; (IFS=,; echo "${out[*]}"); }
for ((i=0; i<8; i++)); do
  .venv/bin/python -u -m norm_cards.eval.propose --problems "$(shard $i 8)" --arms card --card_label template \
    --cards results/dataset50run_tmpl/cards --budget 9 --workers 3 --merge --out $P > $R/logs/propose_tmpl_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] template-card proposals done"
A=$(for m in gpt-6-sol gpt-6.1-sol gpt-6.1-terra; do printf "%s+card:template," $m; done)
for ((i=0; i<10; i++)); do
  .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(shard $i 10)" --arms "${A%,}" --source $P \
    --judge-runs 3 --workers 3 --prompt v2 --model gpt-6-luna > $R/logs/judge_tmpl_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] judged $(ls $R/eval/judgments/problem_*/*card-template.json | wc -l | tr -d ' ')/300 template-card arms"
