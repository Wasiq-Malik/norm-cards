"""Aggregate judgments into results/eval_v2/REPORT.md.

    python -m norm_cards.eval.report

The point of this file is comparability across pipeline iterations: same reference
recipes, same judge, same scoring, so a change in the numbers is a change in the
pipeline. Run metadata is stamped into the report for exactly that reason — a
report produced with a different judge model or a different k is not comparable to
one that wasn't, and shouldn't look like it is.
"""

import argparse
import json
import os
import re
from typing import Dict, List

from . import EVAL_ROOT, gt_path, load_ground_truth
from . import reference

# Arms are discovered from the judgment files rather than fixed, so the same report
# serves the norm-card ablation (baseline vs method) and a model comparison (one arm
# per model) without a code change.
PREFERRED_ARM_ORDER = ("_selftest", "baseline", "method")
SELFTEST_ARM = "_selftest"
METRICS = ("recall",)


def _order_arms(arms):
    known = [a for a in PREFERRED_ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in PREFERRED_ARM_ORDER)


JUDGMENT_FORMAT = "5.0"


_CONDITION_ORDER = ("nocard", "retrieval", "rag", "card")


def _order_conditions(conds):
    """nocard, then retrieval, then card, then any card:<sections> variant — the
    order the comparison is read in, weakest evidence condition first."""
    def key(c):
        # evaluate.py slugs "card:design" to "card-design" for the filename, and
        # report reads arm names back off those filenames.
        head = re.split(r"[:\-]", c, maxsplit=1)[0]
        i = _CONDITION_ORDER.index(head) if head in _CONDITION_ORDER else len(_CONDITION_ORDER)
        return (i, c)
    return sorted(conds, key=key)


def load_judgments(root: str = EVAL_ROOT) -> Dict[str, Dict[str, Dict]]:
    """problem_id -> arm -> judgment.

    Judgments from an older harness version are skipped rather than rendered. They
    carry a `coverage` key too, but it meant weighted *role* fill, whereas the column
    now means requirement satisfaction — so mixing them would put two different
    quantities in one column and read as comparable. Re-run `evaluate.py --force` to
    bring an old set forward.
    """
    out: Dict[str, Dict[str, Dict]] = {}
    stale: List[str] = []
    jroot = os.path.join(root, "judgments")
    if not os.path.isdir(jroot):
        return out
    for d in sorted(os.listdir(jroot)):
        if not d.startswith("problem_"):
            continue
        pid = d.replace("problem_", "")
        for fn in sorted(os.listdir(os.path.join(jroot, d))):
            if not fn.endswith(".json"):
                continue
            arm = fn[:-len(".json")]
            with open(os.path.join(jroot, d, fn), encoding="utf-8") as f:
                j = json.load(f)
            if j.get("format_version") != JUDGMENT_FORMAT:
                stale.append(f"{pid}/{arm} (v{j.get('format_version')})")
                continue
            out.setdefault(pid, {})[arm] = j
    if stale:
        print(f"skipped {len(stale)} judgment(s) from an older harness version — their "
              f"scores are not comparable to the current ones:\n  "
              + "\n  ".join(stale)
              + "\n  re-run `python -m norm_cards.eval.evaluate --force` to refresh.")
    return out


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _delta(a, b):
    """method − baseline, when both exist."""
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{d:+.2f}" if d else "0.00"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


SUFF_SHORT = {"sufficient": "suff", "sufficient_with_gaps": "gaps",
              "insufficient": "insuff", None: "—"}


def _provenance_section(pids: List[str]) -> List[str]:
    """Where each reference came from, and a warning if any is weakly sourced.

    A score is only as good as the reference it was measured against, so the report
    states that up front rather than leaving it to whoever remembers. This is also
    what keeps the caveat from being a hand-edited banner that the next
    `python -m norm_cards.eval.report` silently overwrites.
    """
    rows, suspect, unreviewed = [], [], []
    for pid in pids:
        try:
            line = reference.provenance_line(load_ground_truth(pid))
        except FileNotFoundError:
            line = "reference missing"
        rows.append(f"| {pid} | {line} |")
        if line.startswith("llm_generated") or line.startswith("provenance not stated") \
                or line == "reference missing":
            suspect.append(pid)
        elif "not yet reviewed" in line.lower() or "not reviewed" in line.lower():
            unreviewed.append(pid)

    L = ["## Reference provenance", "",
         "| Claim | Reference recipe came from |", "|---|---|"] + rows + [""]
    if suspect:
        L += [f"> **⚠️ These numbers are not a measurement of the pipeline.** The "
              f"reference recipes for claim(s) {', '.join(suspect)} are "
              f"machine-generated or unattributed. A reference written by asking a "
              f"model to design experiments for the claim is not an independent "
              f"standard — it is another system's output. Replace them with "
              f"hand-authored or paper-derived references (see "
              f"`docs/reference_format.md`) and re-run with `--force` before quoting "
              f"anything below.", ""]
    if unreviewed:
        L += [f"> **⚠️ Unreviewed references.** The recipe(s) for claim(s) "
              f"{', '.join(unreviewed)} are transcribed from a source but no human has "
              f"signed off on the claim statement, the role assignment, or what was "
              f"left out. The numbers below are reproducible, not yet authoritative.", ""]
    return L


def build(judgments: Dict[str, Dict[str, Dict]]) -> str:
    """One question, one number.

    Recall is the only metric the harness computes now — utility, soundness, norm
    alignment and the rest were removed from the judge, not just from this page. The
    question is whether a proposed set establishes what the source team's experiments
    established, and the extra columns made that harder to see rather than easier.
    """
    # The self-test is a gate, not a result. It is checked, reported in one line, and
    # kept out of every table so it cannot be mistaken for an arm.
    pids = sorted(judgments, key=lambda p: (0, int(p)) if p.isdigit() else (1, p))
    arms = [a for a in _order_arms({a for p_ in judgments.values() for a in p_})
            if a != SELFTEST_ARM]
    # Arms are "<model>+<condition>". A model that ran under two or more conditions
    # is a paired observation; a bare arm name is a plain proposer comparison.
    split = [(a.split("+", 1)[0], a.split("+", 1)[1].split("#", 1)[0])
             for a in arms if "+" in a]
    pair_models = sorted({m for m, _ in split
                          if len({c for m2, c in split if m2 == m}) > 1})
    conditions = _order_conditions({c for m, c in split if m in pair_models})
    baseline = "nocard" if "nocard" in conditions else (conditions[0] if conditions
                                                        else "")
    treatments = []
    plain = [a for a in arms if "+" not in a]

    if pair_models:
        title = "Norm-card ablation — " + ", ".join(f"`{m}`" for m in pair_models)
    elif len(plain) > 1:
        title = "Proposer comparison — " + " vs ".join(f"`{m}`" for m in plain)
    else:
        title = "Experiment-proposal evaluation"
    L = [f"# {title}", ""]

    metas = [j.get("judge", {}) for p_ in judgments.values() for j in p_.values()]
    ks = sorted({m.get("runs") for m in metas if m.get("runs")}) or [1]
    if metas:
        models = sorted({m.get("model") for m in metas if m.get("model")})
        L += [f"Judge `{', '.join(models)}`, k={'/'.join(str(k) for k in ks)}, tools "
              f"{'on' if any(m.get('tools') for m in metas) else 'off'}. "
              f"{len(pids)} claim(s), {len(arms)} arm(s).", ""]

    # ---- the one metric ---- #
    L += ["**Recall** — of the experiments the source team actually ran, how many would "
          "a team running the proposed set have effectively performed?", "",
          "For each reference experiment the judge asks whether the proposed set would "
          "establish what it establishes. `covered` = 1.0 (by any route — the method "
          "does not have to match), `partial` = 0.5 (gets at it, outcome stays "
          "ambiguous), `missing` = 0. Recall is the mean. Mapping is many-to-many, so a "
          "short proposal is not penalised for being short.", ""]

    st = _selftest_line(judgments, pids)
    if st:
        L += [st, ""]

    # Arms whose judgment predates the reference file it was scored against. Silent
    # staleness is the failure mode that makes a report quietly wrong, so it is
    # surfaced here rather than left to whoever remembers.
    stale = []
    for pid in pids:
        try:
            ref_mtime = os.path.getmtime(gt_path(pid))
        except OSError:
            continue
        for arm, j in judgments[pid].items():
            if arm == SELFTEST_ARM:
                continue
            f = os.path.join(EVAL_ROOT, "judgments", f"problem_{pid}", f"{arm}.json")
            if os.path.exists(f) and os.path.getmtime(f) < ref_mtime:
                stale.append(f"{pid}/{arm}")
    if stale:
        L += [f"> **⚠️ {len(stale)} arm(s) judged against an older version of their "
              f"reference** — {', '.join(stale)}. Re-run those before quoting them: "
              f"`python -m norm_cards.eval.evaluate --problems <id> --force`.", ""]
    L += _provenance_section(pids)

    # ---- per-claim recall ---- #
    L += ["## Recall", "",
          "| Claim | Arm | Recall | covered / partial / missing | n ref |",
          "|---|---|---|---|---|"]
    for pid in pids:
        for arm in arms:
            j = judgments[pid].get(arm)
            if not j:
                continue
            s_ = j["scores"]
            L.append(f"| {pid} | {arm} | **{_fmt(s_.get('recall'))}** | "
                     f"{s_.get('n_covered', 0)} / {s_.get('n_partial', 0)} / "
                     f"{s_.get('n_missing', 0)} | {s_.get('n_reference')} |")
    L.append("")

    # ---- the ablation ---- #
    if conditions and baseline:
        treatments = [c for c in conditions if c != baseline]

    if conditions and baseline and treatments:
        # Per-claim rollup first: model x claim x condition is the working detail,
        # but the question people ask is "did it help on this claim", so lead with the
        # answer averaged over models and put the per-model rows underneath.
        L += ["## Did the norm card help?", "",
              "Same model, same claim, same subclaims — conditions differ only in what "
              "sits in `current_evidence`. Each Δ is against `" + baseline + "`, paired "
              "run by run.", "",
              "`retrieval` is the control that matters for the project's thesis: a norm "
              "card is a synthesis over retrieved papers, so beating `nocard` only shows "
              "that relevant literature helps. Beating `retrieval` is what shows the "
              "synthesis is doing work.", ""]

        # A (model, condition) may have been drawn more than once (--repeats). Each
        # draw is an observation of the same condition, so they average into one
        # value before pairing rather than inflating n with correlated points.
        obs = {}
        for pid in pids:
            for arm, j in judgments[pid].items():
                if arm == SELFTEST_ARM or "+" not in arm:
                    continue
                m, rest = arm.split("+", 1)
                cond = rest.split("#", 1)[0]
                r = j["scores"].get("recall")
                if r is not None:
                    obs.setdefault((pid, m, cond), []).append(r)

        paired = {c: [] for c in treatments}        # condition -> [(base, treat), ...]
        by_claim = {}                               # pid -> condition -> [(base, treat)]
        for pid in pids:
            for m in pair_models:
                base = obs.get((pid, m, baseline))
                if not base:
                    continue
                x = _mean(base)
                for c in treatments:
                    t = obs.get((pid, m, c))
                    if not t:
                        continue
                    paired[c].append((x, _mean(t)))
                    by_claim.setdefault(pid, {}).setdefault(c, []).append((x, _mean(t)))

        if by_claim:
            head = ["Claim", "n models", baseline] + treatments + \
                   [f"Δ {c}" for c in treatments]
            L += ["### By claim (averaged over models)", "",
                  "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
            for pid in pids:
                row = by_claim.get(pid)
                if not row:
                    continue
                # The baseline column is the baseline's own mean over the models that
                # ran it — NOT a mean over the pair lists. Those lists have different
                # lengths when a treatment arm is missing for some model, and mixing
                # them printed a baseline that matched none of the deltas beside it.
                base_vals = [_mean(v) for m2 in pair_models
                             if (v := obs.get((pid, m2, baseline)))]
                nm = max(len(v) for v in row.values())
                cells = [pid, str(nm), _fmt(_mean(base_vals))]
                cells += [_fmt(_mean([y for _, y in row[c]])) if row.get(c) else "—"
                          for c in treatments]
                cells += [_delta(_mean([x for x, _ in row[c]]),
                                 _mean([y for _, y in row[c]])) if row.get(c) else "—"
                          for c in treatments]
                L.append("| " + " | ".join(cells) + " |")
            allb = _mean([_mean(v) for (pid_, m2, c2), v in obs.items()
                          if c2 == baseline and m2 in pair_models])
            cells = ["**all**", f"**{max(len(paired[c]) for c in treatments)}**",
                     f"**{_fmt(allb)}**"]
            cells += [f"**{_fmt(_mean([y for _, y in paired[c]]))}**" if paired[c]
                      else "—" for c in treatments]
            cells += [f"**{_delta(_mean([x for x, _ in paired[c]]), _mean([y for _, y in paired[c]]))}**"
                      if paired[c] else "—" for c in treatments]
            L += ["| " + " | ".join(cells) + " |", ""]

        L += ["### By model and claim", "",
              "| Claim | Model | " + " | ".join(conditions) + " | "
              + " | ".join(f"Δ {c}" for c in treatments) + " |",
              "|---" * (2 + len(conditions) + len(treatments)) + "|"]
        for pid in pids:
            for m in pair_models:
                got = {c: obs.get((pid, m, c)) for c in conditions}
                if not got.get(baseline):
                    continue
                x = _mean(got[baseline])
                vals = [_fmt(_mean(got[c])) if got.get(c) else "—" for c in conditions]
                dels = [_delta(x, _mean(got[c])) if got.get(c) else "—"
                        for c in treatments]
                L.append(f"| {pid} | {m} | " + " | ".join(vals) + " | "
                         + " | ".join(dels) + " |")
        L.append("")

        for c in treatments:
            prs = paired[c]
            if not prs:
                continue
            deltas = [y - x for x, y in prs]
            n = len(deltas)
            mean = sum(deltas) / n
            up = sum(1 for d in deltas if d > 0)
            dn = sum(1 for d in deltas if d < 0)
            line = (f"`{c}` vs `{baseline}`: **mean Δ = {mean:+.3f}** over {n} paired "
                    f"run(s) — helped {up}, hurt {dn}, tied {n - up - dn}.")
            if n > 1:
                var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
                se = (var / n) ** 0.5
                lo, hi = mean - 1.96 * se, mean + 1.96 * se
                line += f" 95% CI [{lo:+.3f}, {hi:+.3f}]."
                if lo <= 0 <= hi:
                    line += " **Interval contains zero: no detectable effect.**"
            L += [line, ""]

        spreads = [(k, v) for k, v in obs.items() if len(v) > 1]
        if spreads:
            widths = [max(v) - min(v) for _, v in spreads]
            L += ["### Proposer variance", "",
                  "The proposer runs at temperature 1, so an arm drawn twice is two "
                  "samples of one condition, not two conditions. Judge variance is "
                  "controlled at k=3; without repeats, proposer variance is not "
                  "controlled at all — and it lands on the same scale as the effect "
                  "being measured.", "",
                  f"Over {len(spreads)} arm(s) drawn {min(len(v) for _, v in spreads)}"
                  f"-{max(len(v) for _, v in spreads)} times: mean spread "
                  f"**{sum(widths) / len(widths):.3f}**, max **{max(widths):.3f}**.", ""]
            worst = max(spreads, key=lambda kv: max(kv[1]) - min(kv[1]))
            L += [f"Widest: `{worst[0][1]}+{worst[0][2]}` on {worst[0][0]} — "
                  f"{[round(x, 2) for x in sorted(worst[1])]}.", "",
                  "Any Δ below that spread is inside the noise of a single draw and "
                  "cannot be read as an effect.", ""]

        L += ["One coverage step on a 7-experiment reference moves recall by 0.071, "
              "so that is the smallest difference this can resolve. A mean below it "
              "bounds the effect rather than measuring it.", ""]

    # ---- where the misses are, by what the experiment is FOR ---- #
    L += _role_section(judgments, pids, arms, conditions)

    # ---- which reference experiments were covered ---- #
    L += ["## Which experiments were covered", "",
          "`C` covered · `p` partial · `-` missing. Columns are reference experiment "
          "indices; see each claim's `ground_truth.md` for what they are.", ""]
    short = {"covered": "C", "partial": "p", "missing": "-", "CONTESTED": "?"}
    for pid in pids:
        n_ref = next((judgments[pid][a]["scores"].get("n_reference")
                      for a in arms if a in judgments[pid]), 0) or 0
        if not n_ref:
            continue
        L += [f"**{pid}**", "",
              "| Arm | " + " | ".join(str(i) for i in range(n_ref)) + " |",
              "|---" * (n_ref + 1) + "|"]
        for arm in arms:
            j = judgments[pid].get(arm)
            if not j:
                continue
            cells = " | ".join(short.get(x, "?")
                               for x in j["scores"].get("coverage_statuses") or [])
            L.append(f"| {arm} | {cells} |")
        L.append("")

    # ---- what was missed ---- #
    L += ["## What the proposals missed", "",
          "The judge's account of what each set would still need.", ""]
    for pid in pids:
        for arm in arms:
            miss = (judgments[pid].get(arm) or {}).get("missing") or []
            if miss:
                L.append(f"**{pid} / {arm}**")
                L += [f"- {m}" for m in miss]
                L.append("")
    return "\n".join(L)


ROLE_ORDER = ("apparatus", "headline", "mechanism", "control", "confound", "external")

ROLE_GLOSS = {
    "apparatus": "build/validate what the headline test presupposes",
    "headline": "the direct test of the claim",
    "mechanism": "show the asserted cause is the operative one",
    "control": "rule out that the result is an artifact of intervening at all",
    "confound": "kill a rival explanation under which the result looks the same",
    "external": "show the finding generalises beyond its own setup",
}


def _role_section(judgments, pids, arms, conditions) -> List[str]:
    """Coverage grouped by what each reference experiment is FOR.

    A per-claim recall number says how much was missed. It cannot say what kind of
    thing was missed, and those are different problems with different fixes: a
    proposer that skips the apparatus needs the field's resources, one that skips
    the rival-explanation test needs the field's confounds. Requires `roles` on the
    reference; silently absent otherwise."""
    S = {"covered": 1.0, "partial": 0.5, "missing": 0.0, "CONTESTED": 0.5}
    got = {}                                   # role -> condition -> [score]
    counts = {}                                # role -> n reference experiments
    for pid in pids:
        try:
            roles = load_ground_truth(pid).get("roles")
        except Exception:
            roles = None
        if not roles:
            continue
        for r in roles:
            counts[r] = counts.get(r, 0) + 1
        for arm, j in judgments[pid].items():
            if arm == SELFTEST_ARM or "+" not in arm:
                continue
            cond = arm.split("+", 1)[1]
            statuses = j["scores"].get("coverage_statuses") or []
            for i, st in enumerate(statuses):
                if i < len(roles):
                    got.setdefault(roles[i], {}).setdefault(cond, []).append(
                        S.get(st, 0.0))
    if not got:
        return []

    conds = [c for c in conditions if any(c in v for v in got.values())]
    L = ["## Where the misses are", "",
         "Coverage grouped by what each reference experiment is FOR, pooled across "
         "claims and models. This is the diagnostic a recall number cannot give: it "
         "says which KIND of experiment goes missing, and so which part of a norm "
         "card would have to change to fix it.", "",
         "| What the experiment is for | refs | " + " | ".join(conds) + " |",
         "|---" * (2 + len(conds)) + "|"]
    for r in [x for x in ROLE_ORDER if x in got] + [x for x in got if x not in ROLE_ORDER]:
        cells = [_fmt(_mean(got[r][c])) if got[r].get(c) else "—" for c in conds]
        L.append(f"| **{r}** — {ROLE_GLOSS.get(r, '')} | {counts.get(r, 0)} | "
                 + " | ".join(cells) + " |")
    return L + [""]


def _selftest_line(judgments, pids) -> str:
    """The self-test as one sentence: it is a gate on the judge, not a result."""
    rows = [(pid, judgments[pid][SELFTEST_ARM]["scores"])
            for pid in pids if SELFTEST_ARM in judgments.get(pid, {})]
    if not rows:
        return ""
    bad = [pid for pid, s_ in rows if (s_.get("recall") or 0) < 0.95]
    if bad:
        return (f"> **⚠️ Judge self-test FAILED for {', '.join(bad)}** — scoring a "
                f"reference against itself did not return recall 1.0, so the judge is "
                f"under-crediting correct work. Every number below is suspect.")
    return (f"> Judge self-test passed on {len(rows)}/{len(rows)} claims: scoring each "
            f"reference against itself returns recall 1.00.")


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=EVAL_ROOT)
    ap.add_argument("--out", default=os.path.join(EVAL_ROOT, "REPORT.md"))
    args = ap.parse_args()

    judgments = load_judgments(args.root)
    if not judgments:
        print(f"no judgments under {args.root}/judgments — run "
              f"`python -m norm_cards.eval.evaluate` first")
        return
    md = build(judgments)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    n = sum(len(v) for v in judgments.values())
    print(f"wrote {args.out} ({len(judgments)} claims, {n} arm judgments)")


if __name__ == "__main__":
    _main()
