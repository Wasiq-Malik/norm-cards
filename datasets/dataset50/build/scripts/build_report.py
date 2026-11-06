"""Put report_data.py's output into report_live.html: the per-view data, the per-view text the page
quotes ([data-v] fields) and the per-proposer summary. Rerun after report_data.py whenever a run
changes; the page's prose and charts are edited in the HTML itself.

    D50_RUN=results/dataset50run_fn2_b5 D50_PROPOSALS=proposals,proposals_g56 \
    D50_EXTRA="results/dataset50run_fn:qwen3.8-27b:proposals_qwen" \
      python datasets/dataset50/build/scripts/report_data.py /tmp/d50_views.json
    python datasets/dataset50/build/scripts/build_report.py /tmp/d50_views.json
"""
import json
import sys

PAGE = "results/dataset50/report_live.html"
R = json.load(open(sys.argv[1]))
MODELS, POOL = R["models"], R["pool"]
NAME = {"gpt-6-sol": "gpt&#8209;6&#8209;sol", "gpt-6.1-sol": "gpt&#8209;6.1&#8209;sol", "gpt-6.1-terra": "gpt&#8209;6.1&#8209;terra",
        "qwen3.8-27b": "Qwen3.8&#8209;27B", "gpt-5.6-terra": "gpt&#8209;5.6&#8209;terra", "gpt-5.6-luna": "gpt&#8209;5.6&#8209;luna"}
FIELDS = [("robotics", "Robotics"), ("interpretability", "Interpretability"), ("vision", "Vision"), ("nlp", "NLP"),
          ("math_mltheory", "Math &amp; ML theory")]
TYPES = [("headline", "headline"), ("threat", "threat"), ("analysis", "analysis"), ("setup", "setup"), ("other", "other")]


def fp(v): return ("+" if v >= 0 else "&minus;") + f"{abs(v):.3f}"
def ci(d): return f"{fp(d['d'])} [{fp(d['lo'])}, {fp(d['hi'])}]" if d else "&mdash;"
def sig(d): return bool(d) and (d["lo"] > 0 or d["hi"] < 0)
def b(d): return f"<b>{ci(d)}</b>" if sig(d) else ci(d)
def rng(xs): return f"{min(xs):.1f}" if len(xs) == 1 else f"{min(xs):.1f}&ndash;{max(xs):.1f}"
def earlier(m): return m in R["extra"]


def nm(k):
    if k == "all":
        return f"{len(POOL)} gpt proposers, pooled"
    return NAME[k] + (", earlier run" if earlier(k) else "")


pd = {m: R["views"][m]["summary"]["full"]["deltas"]["card_vs_nocard"] for m in POOL}
lo, hi = min(pd.values(), key=lambda d: d["d"]), max(pd.values(), key=lambda d: d["d"])
gainers = [NAME[m] for m in POOL if sig(pd[m]) and pd[m]["d"] > 0]
A0 = R["views"]["all"]["summary"]["full"]["deltas"]
COMMON = {   # the same in every view
    "bprange": f"{fp(lo['d'])} &hellip; {fp(hi['d'])}",
    "bptext": f"Card &minus; no card ranges from {fp(lo['d'])} to {fp(hi['d'])} across the {len(POOL)} gpt proposers; "
              + (", ".join(gainers) if gainers else "none") + (" clears" if len(gainers) == 1 else " clear") + " zero.",
    "a0rag": ci(A0["rag_vs_nocard"]), "a0card": b(A0["card_vs_nocard"]), "a0cr": b(A0["card_vs_rag"]),
    "a0n": str(A0["card_vs_nocard"]["n"]),
}

TXT = {}
for k in ["all"] + MODELS:
    v = R["views"][k]; F = v["summary"]["full"]["deltas"]; G = v["summary"]["threat"]["deltas"]["card_vs_nocard"]
    O = v["summary"]["scored"]["deltas"]["card_vs_nocard"]; bt = v["by_type"]; ud = v["up_down"]; pc = v["per_card"]
    ms = POOL if k == "all" else [k]
    exn = [R["shape"][m]["exp_nocard"] for m in ms]; exc = [R["shape"][m]["exp_card"] for m in ms]
    budget = R["budget"][ms[0]]
    TXT[k] = dict(COMMON, **{
        "label": nm(k),
        "hd": fp(F["card_vs_nocard"]["d"]), "hdci": f"[{fp(F['card_vs_nocard']['lo'])}, {fp(F['card_vs_nocard']['hi'])}]",
        "hdn": str(F["card_vs_nocard"]["n"]),
        "hdsig": "excludes zero" if sig(F["card_vs_nocard"]) else "not distinguishable from zero",
        "hg": fp(G["d"]), "hgci": f"[{fp(G['lo'])}, {fp(G['hi'])}]",
        "npairs": str(F["card_vs_nocard"]["n"]),
        "who": ("3 judge passes over " + str(sum(1 for c in v["claims"].values() if c["d"] is not None)) + " claims"
                + (f", from the earlier run at budget {budget} under the first Field Norms prompt" if earlier(k) else "")),
        "ragnc": ci(F["rag_vs_nocard"]), "recall": ci(v["metrics"]["recall"]["delta"]),
        "rig": b(G),
        "core": ", ".join(f"{t} {b(bt[t]['card_vs_nocard'])}" for t in ("headline", "analysis", "setup")),
        "exps": f"{rng(exc)} experiments per card proposal against {rng(exn)} without, at a budget of {budget}",
        "cnfnote": " This view&rsquo;s gain on threats clears zero." if sig(G) and G["d"] > 0 else "",
        "obl": b(O), "rign": str(G["n"]),
        "cnfcov": f"{bt['analysis']['nocard']:.2f} without a card and {bt['analysis']['card']:.2f} with one",
        "up": str(ud["up"]), "down": str(ud["down"]), "ties": str(ud["n"] - ud["up"] - ud["down"]), "ncl": str(ud["n"]),
        "max": f"{fp(v['extremes']['max'][1])} (<span class=\"mono\">{v['extremes']['max'][0].replace('arxiv', '')}</span>)",
        "min": f"{fp(v['extremes']['min'][1])} (<span class=\"mono\">{v['extremes']['min'][0].replace('arxiv', '')}</span>)",
        "pcpos": f"{sum(r['mean'] > 0 for r in pc)} of {len(pc)}",
        "pctop": f"<b>{pc[0]['name']}</b> ({fp(pc[0]['mean'])}, {pc[0]['n']} claims) and "
                 f"<b>{pc[1]['name']}</b> ({fp(pc[1]['mean'])}, {pc[1]['n']} claims)",
        "pcbot": f"{pc[-1]['name']} ({fp(pc[-1]['mean'])}, {pc[-1]['n']} claims)",
        "dotsub": "mean over " + (f"the {len(POOL)} gpt proposers" if k == "all"
                                  else NAME[k] + "&rsquo;s single proposal per condition"),
        "budget": str(budget),
        "fieldrows": "".join(f'<tr><td class="l">{lab}</td><td>{v["by_field"][f]["n"]}</td>'
                             + "".join(f'<td>{v["by_field"][f]["means"][a]:.3f}</td>' for a in ("nocard", "rag", "card"))
                             + f'<td>{b(v["by_field"][f]["card_vs_nocard"])}</td></tr>' for f, lab in FIELDS),
        "typerows": "".join(f'<tr><td class="l">{"<b>" + lab + "</b>" if t == "threat" else lab}</td><td>{bt[t]["n"]}</td>'
                            + "".join(f'<td>{bt[t].get(a, 0):.3f}</td>' for a in ("nocard", "rag", "card"))
                            + f'<td>{b(bt[t]["card_vs_nocard"])}</td></tr>' for t, lab in TYPES),
    })

per = {m: R["views"][m]["summary"]["full"] for m in MODELS}
PM = {"per_model": {m: per[m]["means"] for m in MODELS}, "deltas": {m: per[m]["deltas"]["card_vs_nocard"] for m in MODELS}}

h = open(PAGE).read()
i = h.index("<script>window.__V3=") + len("<script>window.__V3="); j = h.index("</script>", i)
D = json.loads(h[i:j].rstrip().rstrip(";"))
D.update(models=MODELS, pool=POOL, budget=R["budget"], extra=R["extra"], claims=R["claims"], mean_ref=R["mean_ref"],
         shape=R["shape"], views=R["views"], txt=TXT, PM=PM)
if "agreement" in R:
    D["judge_agreement"] = {"items": R["agreement"]["n_items"], "unanimous": R["agreement"]["unanimous"],
                            "contested": R["agreement"]["contested"]}
h = h[:i] + json.dumps(D, separators=(",", ":")) + ";" + h[j:]
open(PAGE, "w").write(h)
print("ok", len(h), "models", MODELS, "pool", POOL)
