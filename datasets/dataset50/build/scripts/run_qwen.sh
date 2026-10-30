cd "$(git rev-parse --show-toplevel)"
set -e
# Qwen3.8-27B (open weights, Zeblok vLLM) as a fourth proposer: same claims, same stored
# subclaims, same cards and rag passages, budget 9 (capped), judged like the GPT arms.
R=results/dataset50run; P=$R/proposals_qwen; M=qwen3.8-27b
export PYTHONPATH=. NORM_CARDS_EVAL_ROOT=$R/eval NORM_CARDS_CLAIMS=norm_cards/data/dataset50-claims-v1.jsonl
mkdir -p $P $R/logs
.venv/bin/python - <<'PY'
import json, glob, os
for f in glob.glob("results/dataset50run/proposals/problem_*/scify_proposer.json"):
    d = json.load(open(f)); d["arms"] = {}
    out = f.replace("/proposals/", "/proposals_qwen/"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(d, open(out, "w"), indent=2)
PY
IDS=($(ls $R/eval/ground_truth | sed 's/problem_//'))
shard() { local i=$1 n=$2 out=() j; for ((j=i; j<${#IDS[@]}; j+=n)); do out+=("${IDS[$j]}"); done; (IFS=,; echo "${out[*]}"); }
for ((i=0; i<8; i++)); do
  .venv/bin/python -u -m norm_cards.eval.propose --problems "$(shard $i 8)" --models $M --arms nocard,card,rag \
    --cards $R/cards --budget 9 --workers 3 --merge --out $P > $R/logs/propose_qwen_$i.log 2>&1 &
done
wait
n=$(.venv/bin/python -c "
import json,glob; print(sum(sum(1 for v in json.load(open(p))['arms'].values() if v.get('experiments')) for p in glob.glob('$P/problem_*/scify_proposer.json')))")
echo "[$(date +%H:%M)] qwen proposals with experiments: $n/300"
for ((i=0; i<10; i++)); do
  .venv/bin/python -u -m norm_cards.eval.evaluate --problems "$(shard $i 10)" --arms "$M+nocard,$M+card,$M+rag" \
    --source $P --judge-runs 3 --workers 3 --prompt v2 --model gpt-6-luna > $R/logs/judge_qwen_$i.log 2>&1 &
done
wait
echo "[$(date +%H:%M)] judged $(ls $R/eval/judgments/problem_*/$M+*.json 2>/dev/null | grep -v trace | wc -l | tr -d ' ')/300 qwen arms"
