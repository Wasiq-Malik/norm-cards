"""CLI for norm-card paper gathering. Mirrors codeagent.py's batch style.

Single claim:
    python -m norm_cards.run --claim "..." --model gpt-5-mini

Batch (resumable):
    python -m norm_cards.run \
        --claim_file dataset/sprint2/sprint2-continuous-release-problems-v1.jsonl \
        --output_folder results/norm_cards/sprint2_v1 \
        --model gpt-5-mini --per_query 8 --top_k 30
"""

import argparse
import json
import os
import time

from .taxonomy import Taxonomy
from .sources import get_sources
from . import pipeline, config


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            json.dump(r, f)
            f.write("\n")


def get_args():
    p = argparse.ArgumentParser(description="Subfield-aware paper gathering for norm cards")
    p.add_argument("--claim", type=str, default="", help="Single claim to gather papers for")
    p.add_argument("--claim_file", type=str, default="", help="JSONL of problem records")
    p.add_argument("--output_folder", type=str, default="", help="Output dir (batch mode)")
    p.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                   help="LLM for classification/query-gen (falls back to heuristics)")
    p.add_argument("--sources", type=str, default="",
                   help="Comma-separated subset: openalex,arxiv,semantic_scholar,"
                        "papers_with_code,google_scholar (default: all available)")
    p.add_argument("--per_query", type=int, default=8, help="Results per query per source")
    p.add_argument("--top_k", type=int, default=30, help="Papers kept per claim after ranking")
    return p.parse_args()


def main():
    args = get_args()
    taxonomy = Taxonomy.load()
    source_names = [s.strip() for s in args.sources.split(",") if s.strip()] or None
    sources = get_sources(names=source_names)
    print(f"Sources: {[s.name for s in sources]}")
    print(f"Taxonomy: {len(taxonomy.tasks)} PwC subfields; OpenAlex topics resolved at runtime")

    # Single-claim mode.
    if args.claim:
        bundle = pipeline.build_bundle({"claim": args.claim}, taxonomy, sources,
                                       model=args.model, per_query=args.per_query,
                                       top_k=args.top_k)
        print(json.dumps(bundle, indent=2)[:4000])
        return

    if not args.claim_file or not args.output_folder:
        raise SystemExit("Provide --claim OR (--claim_file AND --output_folder)")

    os.makedirs(args.output_folder, exist_ok=True)
    out_path = os.path.join(args.output_folder, "paper_bundles.jsonl")
    rows = load_jsonl(args.claim_file)
    done = load_jsonl(out_path) if os.path.exists(out_path) else []
    start = len(done)
    print(f"Loaded {len(rows)} claims; resuming at index {start}")

    bundles = list(done)
    for i in range(start, len(rows)):
        rec = rows[i]
        print(f"\n=== [{i + 1}/{len(rows)}] problem {rec.get('problem_id')} ===")
        bundle = pipeline.build_bundle(
            rec, taxonomy, sources, model=args.model,
            per_query=args.per_query, top_k=args.top_k)
        bundles.append(bundle)
        save_jsonl(out_path, bundles)  # write after each claim (resumable)
        time.sleep(2)  # be polite to the APIs

    # Persist any taxonomy growth (new custom subfields).
    taxonomy.save()
    print(f"\nDone. {len(bundles)} bundles -> {out_path}")


if __name__ == "__main__":
    main()
