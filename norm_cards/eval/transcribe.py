"""Draft a reference recipe by TRANSCRIBING a source paper's own experiments.

    python -m norm_cards.eval.transcribe --problem icml12345 --arxiv 2510.20487
    python -m norm_cards.eval.transcribe --problem icml12345 --pdf ./paper.pdf

Writes `ground_truth.draft.json` next to where the reference belongs, plus a
`.review.md` to read alongside the paper. It deliberately does NOT write
`ground_truth.json`: `--accept` does that, after a person has read the draft.

WHY THIS IS NOT THE THING THE HARNESS REFUSES TO DO
---------------------------------------------------
`reference.py` will not generate a reference, and that stands. The refusal is
specific: a reference must not be an LLM *designing* experiments for the claim,
because then the standard is just another system's output, and if it were good
enough to define correctness you would ship it as the proposer instead.

Transcription is a different operation on a different input. The experiments
already exist — a competent team ran them, a venue reviewed them, and they are
printed in the paper. The model is reading a source of truth and restating it,
which is extraction, and its output is checked against that source by a person
before it counts. The distinction is enforced structurally, not by intention:

  * the paper text is required; there is no path that runs from the claim alone,
  * the prompt forbids adding, improving or completing the authors' design,
  * every drafted experiment must cite the section it came from,
  * the draft is written to a different filename and carries
    `provenance.kind = "draft transcription, NOT reviewed"`, which
    `reference.py --check` rejects and the judge would display as weak,
  * only `--accept` promotes it, and only after the reviewer has seen the diff.

This is what makes 20 claims feasible where hand-transcription made 3 expensive.
It is not a licence to skip the reading.
"""

import argparse
import json
import os
import re

from . import gt_dir, gt_path, load_claims
from . import reference, schemas
from .. import llm, fulltext

DRAFT_KIND = "draft transcription, NOT reviewed"

_PROMPT = """Below is the full text of a paper. Transcribe the experiments the authors
ACTUALLY RAN into a list of prose paragraphs.

You are a scribe, not a designer. This distinction is the whole point of the task:

- Report only experiments the paper reports. If the authors did not run a control,
  the list has no control in it. An absent experiment is a finding about the paper.
- Do not improve, complete, tidy or generalize their design. Do not add the
  experiment you think they should have run.
- Do not editorialize. No emphasis, no "crucially", no "this is what makes the
  result non-trivial". Write what was done and what number decided it. The
  reference is scored against by a judge, and any framing you add is framing the
  proposer was never asked to produce, so it costs the proposal points for
  nothing.
- Keep the authors' own numbers, thresholds, model names, dataset names and
  hyperparameters. These are what make the decision rule unambiguous.

- NEVER point at a figure, table, panel or plotted line. "shown by the orange and
  green lines in the robustness panel", "see Figure 3", "reported in Table 2" are
  useless here: the reader of this reference is a system that cannot see the
  paper. Every decision must be stated in words and numbers on the page. If the
  result lives only in a plot, read the plot and write down what it shows —
  "clean accuracy rises from 84.1% to 86.7% while robust accuracy is unchanged
  within 0.3 points" — or, if you cannot read a value off it, describe the
  comparison and say the direction, never the figure.

FORM. One paragraph per experiment, at the granularity a proposer writes: what is
run, on what data and model, what is measured, and what result decides the
question. Open with a short imperative saying what the experiment is for ("Build
the evaluation set the analysis depends on.", "Rule out that any perturbation of
this size would do it."). Order them as the argument runs, not as the paper's
sections happen to fall.

ROLE. Label each with what it is FOR:
  apparatus  build or validate something the headline test presupposes
  headline   the direct test of the claim's assertion
  mechanism  show the asserted cause is the operative one
  control    rule out that the result is an artifact of intervening at all
  confound   kill a rival explanation under which the result would look the same
  external   show the finding generalises beyond the setup that produced it

CITE. For each, name the section, figure or table it is transcribed from, so a
reviewer can check it in one lookup.

EXAMPLES. Below are experiments from references that were transcribed by hand and
reviewed against their papers. Match their granularity, their plainness, and the
way each opens by saying what the experiment is FOR. Do not copy their subject
matter — they are from other fields.
{examples}

CLAIM this reference will be scored against (context only — transcribe the
paper's experiments, NOT experiments for this claim):
{claim}

PAPER:
{text}

Return JSON:
{{"experiments":[{{"text":"the prose paragraph",
                  "role":"apparatus|headline|mechanism|control|confound|external",
                  "source":"Section 4.2 / Table 3"}}],
  "not_run":["experiments a reader might expect that this paper does NOT run"]}}"""


# The three references authored and reviewed by hand before this tool existed.
# Style is easier to show than to specify, and these are the only examples that
# were checked against their source papers line by line.
EXAMPLE_REFS = os.path.join("results", "eval_icml2026_v3_notools", "ground_truth")


def load_examples(root: str = "", per_ref: int = 2, max_chars: int = 5000) -> str:
    """A few reviewed experiments, with the role each plays, as exemplars."""
    root = root or EXAMPLE_REFS
    if not os.path.isdir(root):
        return "(no reviewed references available as examples)"
    out, used = [], 0
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d, "ground_truth.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            ref = json.load(f)
        roles = ref.get("roles") or []
        for i, e in list(enumerate(ref.get("experiments") or []))[:per_ref]:
            role = roles[i] if i < len(roles) else "?"
            block = f"--- example ({role}) ---\n{e.strip()}"
            if used + len(block) > max_chars:
                break
            out.append(block)
            used += len(block)
    return "\n\n".join(out) or "(no reviewed references available as examples)"


def draft(claim: str, text: str, model: str = "gpt-5.6-sol", examples: str = "") -> dict:
    if not (text or "").strip():
        raise ValueError("no paper text — transcription has no source to work from")
    return llm.complete_json(
        _PROMPT.format(claim=claim[:1200], text=text[:200000],
                       examples=examples or load_examples()), model=model)


FIGREF = re.compile(r"\b(orange|green|blue|red|purple|dashed|solid)\s+(line|curve|bar)s?\b"
                    r"|\bFig(?:ure)?\.?\s*\d|\bTable\s*\d|\bpanel\b|\bsubplot\b"
                    r"|\bleft (?:panel|plot)\b|\bas shown in the (?:figure|plot|chart)\b", re.I)


def figure_references(exps: list) -> list:
    """Experiments that point at something the reader cannot see.

    A reference is read by a system with no access to the paper, so "shown by the
    orange and green lines" states no decision at all. Caught here rather than
    left for a reviewer, because it reads as fluent prose and is easy to skim past."""
    return [(i, FIGREF.search(e).group(0)) for i, e in enumerate(exps) if FIGREF.search(e)]


def to_reference(problem: dict, drafted: dict, source: str) -> dict:
    exps = drafted.get("experiments") or []
    return {
        "type": "ground_truth", "format_version": "3.0",
        "problem_id": problem["problem_id"], "claim": problem["claim"],
        "provenance": {
            "kind": DRAFT_KIND,
            "sources": [source],
            "notes": "Drafted by transcription from the source paper. Every "
                     "experiment must be checked against the cited section before "
                     "this is promoted with --accept.",
        },
        "experiments": [e["text"] for e in exps if e.get("text")],
        "roles": [e.get("role", "headline") for e in exps if e.get("text")],
        "_transcription_sources": [e.get("source", "") for e in exps if e.get("text")],
        "_not_run": drafted.get("not_run") or [],
    }


def review_markdown(ref: dict) -> str:
    L = [f"# Transcription draft — problem {ref['problem_id']}", "",
         "**Not a reference yet.** Read each experiment against the section it cites. "
         "Fix or delete anything the paper does not actually say, then promote with "
         "`--accept`.", "",
         f"**Claim.** {ref['claim']}", ""]
    srcs = ref.get("_transcription_sources") or []
    for i, e in enumerate(ref["experiments"]):
        L += [f"### {i} · `{ref['roles'][i]}` · transcribed from {srcs[i] if i < len(srcs) else '?'}",
              "", e, ""]
    if ref.get("_not_run"):
        L += ["## The paper does NOT run these", "",
              "Listed so a missing control is recorded as a fact about the paper "
              "rather than mistaken for a transcription slip.", ""]
        L += [f"- {x}" for x in ref["_not_run"]] + [""]
    return "\n".join(L)


def _draft_path(pid: str) -> str:
    return os.path.join(gt_dir(pid), "ground_truth.draft.json")


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problem", required=True, help="problem id from the claims file")
    ap.add_argument("--arxiv", default="", help="arXiv id of the source paper")
    ap.add_argument("--pdf", default="", help="local PDF path, instead of --arxiv")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--cache_dir", default=".pdfcache")
    ap.add_argument("--accept", action="store_true",
                    help="promote an existing draft to ground_truth.json. Only after "
                         "you have read it against the paper — this is the step the "
                         "whole design hangs on.")
    ap.add_argument("--author", default="",
                    help="who reviewed it, recorded in provenance on --accept")
    args = ap.parse_args()

    pid = args.problem
    claims = load_claims()
    if pid not in claims:
        raise SystemExit(f"{pid} is not in the claims file")
    os.makedirs(gt_dir(pid), exist_ok=True)

    if args.accept:
        path = _draft_path(pid)
        if not os.path.exists(path):
            raise SystemExit(f"no draft at {path} — run without --accept first")
        if not args.author:
            raise SystemExit("--accept needs --author: a reference records who "
                             "checked it, and the judge is shown that line")
        with open(path, encoding="utf-8") as f:
            ref = json.load(f)
        ref["provenance"]["kind"] = "transcribed from the source paper, human-reviewed"
        ref["provenance"]["author"] = args.author
        for k in ("_transcription_sources", "_not_run"):
            ref.pop(k, None)
        err = schemas.validate_reference(ref)
        if err:
            raise SystemExit(f"draft does not validate, fix it first: {err}")
        with open(gt_path(pid), "w", encoding="utf-8") as f:
            json.dump(ref, f, indent=2)
        with open(os.path.join(gt_dir(pid), "ground_truth.md"), "w",
                  encoding="utf-8") as f:
            f.write(reference.to_markdown(ref))
        print(f"promoted -> {gt_path(pid)}  ({len(ref['experiments'])} experiments)")
        return

    if not (args.arxiv or args.pdf):
        raise SystemExit("transcription needs a source: --arxiv or --pdf. There is "
                         "deliberately no path that drafts from the claim alone.")
    if args.pdf:
        text = fulltext.extract_text(open(args.pdf, "rb").read())
        source = os.path.basename(args.pdf)
    else:
        text, _ = fulltext.fetch_fulltext({"arxiv_id": args.arxiv, "title": pid},
                                          args.cache_dir)
        source = f"arXiv:{args.arxiv}"
    if not text:
        raise SystemExit(f"could not get full text for {source}")
    print(f"{len(text)} chars from {source}; drafting with {args.model}")

    ref = to_reference(claims[pid], draft(claims[pid]["claim"], text, args.model), source)
    with open(_draft_path(pid), "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)
    md = os.path.join(gt_dir(pid), "ground_truth.draft.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(review_markdown(ref))
    figs = figure_references(ref["experiments"])
    if figs:
        print(f"  WARNING: {len(figs)} experiment(s) point at a figure or table the reader "
              f"cannot see — rewrite these before accepting:")
        for i, m in figs:
            print(f"    experiment {i}: ...{m}...")
    print(f"drafted {len(ref['experiments'])} experiments "
          f"{dict((r, ref['roles'].count(r)) for r in sorted(set(ref['roles'])))}")
    print(f"  -> {_draft_path(pid)}\n  -> {md}   <- read this against the paper")
    print(f"then: python -m norm_cards.eval.transcribe --problem {pid} --accept "
          f"--author '<your name>'")


if __name__ == "__main__":
    _main()
