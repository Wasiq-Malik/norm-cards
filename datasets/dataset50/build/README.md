# How dataset50 was built

50 source papers (10 each: robotics, interpretability, vision, NLP, math + ML theory), 100 claims,
478 reference experiments, 20 shared format-2 subfield cards. Run everything from the repo root.

| Step | What | Where |
|---|---|---|
| 1. Find papers | arXiv search on venue comments (`co:"ICML 2026"` etc.) per field, v1 >= 2026-02-01 | `scripts/harvest.py` |
| 2. Screen | venue regex (no workshops, Findings, "under review"), then gpt-6-luna scores each abstract | `scripts/screen.py` |
| 3. Pick | by hand from the top of each field; science papers held out for a later test set | `picks.json`, `science_heldout.json`, `drafts/SKIPPED.md` |
| 4. Draft | claims and reference experiments written from each paper's arXiv HTML by Claude Opus 5.5 | `drafts/` (30 new), `drafts_orig20/` (original 20, redone) |
| 5. Label | gpt-6-luna subfield classifier, 3 samples per claim, labels kept when 2 agree | `scripts/reclassify.py`, `subfield_labels/` |
| 6. Taxonomy | vocabulary audit (+12 subfields, -6 placeholders), then merges | `scripts/taxonomy_audit.py`, `scripts/taxonomy_merge.py` |
| 7. Consolidate | merge nested labels, drop catch-alls, 7 hand removals, cards need >= 2 papers | `scripts/consolidate.py` -> `../subfields.json` |
| 8. Dataset files | claims file and per-claim references from the drafts | `scripts/make_dataset.py` |
| 9. Cards | `norm_cards.subfield_gather` (luna queries + relevance screen, full text only), source papers removed, `build_cards --format 2` on luna | `scripts/build_cards.sh`, `scripts/leakage.py` |
| 10. Run | bootstrap, assemble, propose (sharded), judge (sharded) | `scripts/run_eval.sh`, `scripts/report_data.py` |
| 11. Field Norms run | the previous result: proposers get the card or passages under a Field Norms key (`propose --prompt fieldnorms`); single-pass screen, then `evaluate --add-passes` to three passes; Qwen3.8-27B and gpt-5.6-terra/-luna added | `scripts/run_fieldnorms_full.sh`, `scripts/run_k3_and_luna.sh`, `scripts/judge_fn_add.sh`, `subset30.txt` |
| 12. Checks | old vs new prompt on the same claims; Decisions API (gpt-6-luna, no text) as a second judge | `scripts/compare_prompts.py`, `scripts/decisions_judge.py`, `scripts/compare_decisions.py` |
| 13. Types | one type per reference experiment by main purpose (setup / headline / threat / analysis / other), gpt-6-luna, 3 samples per claim, majority; Claude Opus 5.5's hand labels as the check (95% agree) | `scripts/label_types.py` -> `types.json`, `types_opus.json` |
| 14. Current setup | `propose --prompt fieldnorms2` (card as Field Norms, passages as Relevant Literature, empty fields say None) at budget 5; tried on subset30, then all 100 claims at budget 9 and 5 | `scripts/run_fn2_subset30.sh`, `scripts/run_fn2_full.sh`, `scripts/run_fn2_b5_subset30.sh`, `scripts/run_fn2_b5_full.sh`, `scripts/run_fn2_b5_qwen.sh`, `scripts/compare_fn2.py` |
| 15. Report | per-proposer, per-type data for the report page, and the page update | `scripts/report_data.py`, `scripts/build_report.py` |

Steps 1-3 and 5-6 were run once from `results/dataset50/` and are kept as a record: they read
inputs not stored here (the 3,465-row candidate pool) or edit `norm_cards/data/taxonomy.json` in
place and are not idempotent. Steps 7, 8, 10 and `leakage.py` rerun from the repo as they are;
`consolidate.py` reproduces `subfield_labels/consolidation.json` byte for byte.

References are model-written and **not human-reviewed**. Kept exceptions from the original 20:
2510.20487 (first posted Oct 2025, no venue), 2608.09928 (ICML workshop), five Findings papers.

To add a proposer, run `scripts/run_eval.sh` with your model in `--models` (an open-weight model
needs an entry in `norm_cards.config.OPEN_MODELS` and its endpoint and key in `.env`). Proposals
reuse each claim's stored subclaims when run with `--merge` into an existing proposals directory.
