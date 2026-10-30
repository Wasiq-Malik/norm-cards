cd "$(git rev-parse --show-toplevel)"
set -e
D=datasets/dataset50; B=results/cards50; G=results/gather50
export PYTHONPATH=.
# 1. let the first gather finish, then re-run it: it resumes and retries only the subfields that failed
while pgrep -f "subfield_gather --subfields $D" >/dev/null; do sleep 30; done
for i in 1 2 3; do
  n=$(wc -l < $D/subfield_bundles.jsonl | tr -d ' ')
  [ "$n" = 20 ] && break
  echo "[$(date +%H:%M)] $n/20 bundles; gather pass $((i+1))"
  .venv/bin/python -u -m norm_cards.subfield_gather --subfields $D/subfields.json --out $D --require_fulltext >> $G/gather.log 2>&1
done
n=$(wc -l < $D/subfield_bundles.jsonl | tr -d ' ')
[ "$n" = 20 ] || { echo "only $n/20 bundles after 4 passes — stopping"; exit 1; }
# 2. no card may be built from one of the dataset's own source papers
.venv/bin/python datasets/dataset50/build/scripts/leakage.py $D/subfield_bundles.jsonl --write
# 3. cards: format 2, gpt-6-luna map and reduce (normcard defaults), HTML first
mkdir -p $B
.venv/bin/python - <<'PY'
import json, os
L = [l for l in open("datasets/dataset50/subfield_bundles.jsonl") if l.strip()]
for i in range(4):
    os.makedirs(f"results/cards50/s{i}", exist_ok=True)
    open(f"results/cards50/s{i}/paper_bundles.jsonl", "w").writelines(L[i::4])
PY
for i in 0 1 2 3; do
  .venv/bin/python -u -m norm_cards.eval.build_cards --run $B/s$i --format 2 --no_recipes > $B/s$i/build.log 2>&1 &
done
wait
n=$(ls $B/s[0-3]/problem_subfield_*/norm_card.json | wc -l | tr -d ' ')
[ "$n" = 20 ] || { echo "only $n/20 cards built — stopping"; exit 1; }
mkdir -p $D/cards
for c in $B/s[0-3]/problem_subfield_*; do mkdir -p $D/cards/$(basename $c); cp $c/norm_card.json $D/cards/$(basename $c)/; done
echo "[$(date +%H:%M)] BUILT 20/20 cards -> $D/cards"
