"""Screen harvested candidates: accepted main-track venue, then an LLM read of the abstract."""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/Users/wasiqmalik/Public/norm-cards")
from norm_cards import llm
H = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(H, "candidates.json")))
have = {x["id"] for x in json.load(open("/Users/wasiqmalik/Public/norm-cards/results/dataset20/all_papers.json"))}
sprint = [json.loads(l)["claim"][:400] for l in open("/Users/wasiqmalik/Public/norm-cards/norm_cards/data/sprint2-continuous-release-problems-v1.jsonl")]
BAD = re.compile(r"under review|submitted to|submission|workshop|findings|tiny paper|position|blog|demo track|student", re.I)
GOOD = re.compile(r"accept|to appear|camera|published|proceedings|main (track|conference)|oral|spotlight|poster|^\s*(neurips|icml|iclr|cvpr|eccv|acl|emnlp|naacl|corl|rss|icra|aaai) 20", re.I)
def venue_ok(r):
    c = r["comment"]
    return bool(GOOD.search(c)) and not BAD.search(c)
P = """You are screening a paper to become a benchmark item. From the paper we will write
1-3 falsifiable claims of the form "method/system X does Y under conditions Z", and the
list of experiments the authors ran to test each claim becomes the ground truth.

Field we are filling: {field}
Reference claims this benchmark is modelled on (style and topic):
{sprint}

PAPER: {title}
ABSTRACT: {abstract}

Score 1-5 each:
 claimable: the paper makes a concrete empirical claim tested by several distinct
   experiments (main result + controls/ablations/robustness). 1 = survey, position,
   pure theory, dataset/benchmark release with no claim, or a single-number leaderboard paper.
 field_fit: it squarely belongs to the field above.
 sprint_fit: its topic is close to the reference claims' topics.
Return JSON {{"claimable":n,"field_fit":n,"sprint_fit":n,"subfield":"2-5 words","why":"one line"}}"""
FIELD = {"robotics": "robotics / embodied agents / RL control",
         "interpretability": "interpretability: mechanistic interpretability, knowledge editing, activation steering",
         "vision": "computer vision robustness: backdoors, adversarial attacks, poisoning, deepfake/forensics, OOD detection",
         "nlp": "NLP / LLMs: constrained decoding, structured output, instruction following, jailbreaks and LLM safety",
         "nlp_cd": "NLP / LLMs: constrained decoding, structured output, instruction following, jailbreaks and LLM safety",
         "sci_math": "scientific and mathematical ML: theorem proving, autoformalization, molecules, proteins, materials",
         "ml_theory": "ML theory and optimization: optimizers, learning-rate and batch-size schedules, training dynamics, implicit bias, generalization, scaling laws (papers with empirical tests, not science applications)"}
SP = "\n".join(f"- {s}" for s in sprint)
ONLY = os.environ.get("FIELD_ONLY")  # e.g. ml_theory, to screen one field's new rows
todo = [r for r in rows.values() if r["id"] not in have and venue_ok(r) and "s" not in r
        and (not ONLY or r["field"] == ONLY)]
print(len(rows), "harvested;", sum(venue_ok(r) for r in rows.values()), "accepted-venue;", len(todo), "to screen", flush=True)
def one(r):
    try:
        d = llm.complete_json(P.format(field=FIELD[r["field"]], sprint=SP, title=r["title"],
                                       abstract=r["abstract"][:2500]), model="gpt-6-luna")
        r["s"] = d
    except Exception as e:
        r["s_err"] = str(e)[:200]
with ThreadPoolExecutor(12) as ex:
    list(ex.map(one, todo))
json.dump(rows, open(os.path.join(H, "candidates.json"), "w"), indent=1)
print("screened", sum("s" in r for r in rows.values()), "errors", sum("s_err" in r for r in rows.values()))
