# Reference experiments — problem arxiv2608_28229

**Claim.** General context-free-grammar-constrained decoding of instruction-tuned language models under finite token budgets can ensure completed-output validity rather than merely preserving prefix feasibility, while improving the correctness of generated structures. The validity bar is that 100% of returned outputs are accepted by the target grammar, with no invalid or truncated output returned when an accepted completion is certified within the available token budget. The quality bar is that task-specific correctness exceeds that of the strongest applicable prior constrained decoder under the same base model and token budget.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2608.28229

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Fix the shared evaluation protocol for the structured-generation experiments. Run LLAMA-3.1-8B-INSTRUCT, LLAMA-3.2-3B-INSTRUCT, and QWEN2.5-7B-INSTRUCT on json-mode-eval's 100 examples, Spider's official 1,034-example validation split, and the 6,185-example LTL drone-planning golden dataset, using a uniform token budget T = 120. Measure JSON syntactic validity, schema validity, perplexity, and seconds per output; SQL grammar acceptance, execution accuracy, perplexity, and time; and LTL parser acceptance, Büchi-automaton equivalence to the gold formula, perplexity, and time. BASE LM samples at temperature 0.8 for 10 independent generations per input. SAMPLE-VERIFY uses those same 10 raw generations, removes syntactically invalid outputs, and for each of the 10 fixed seeds randomly selects one survivor when available. Report means and standard deviations over seeds for stochastic baselines and one run for deterministic methods. All experiments use an Intel Xeon Silver 4416+ CPU and one NVIDIA L40S 48 GB GPU.

### Experiment 1

Measure the reusable grammar-processing cost required before decoding. Convert each CFG to a PDA and precompute bounded summaries. CFG-to-PDA conversion for SQL takes 0.67 seconds, and SQL summary precomputation with H = 13, supporting up to three nested queries, takes 3 hours 40 minutes. LTL summary precomputation with H = 50, supporting up to 50 nested parentheses, takes 0.42 seconds. For schema-specific JSON grammars, conversion and summary precomputation together take 0.29 seconds on average with a standard deviation of 0.11 seconds.

### Experiment 2

Test completed JSON validity and schema correctness under the common 120-token budget. Run SWYB with alpha = 0.25 and beam width 2 on the 100-example json-mode-eval dataset with LLAMA-3.1-8B-INSTRUCT, LLAMA-3.2-3B-INSTRUCT, and QWEN2.5-7B-INSTRUCT, and compare it with BASE LM, SAMPLE-VERIFY, Guidance, Outlines, XGrammar, SynCode Strict, and GenLM with NP = 2 and NP = 5. SWYB returns 100.0% syntactically valid and 100.0% schema-valid outputs for all three models. The highest non-SWYB schema-validity results are 90.0% with Outlines on LLAMA-3.1-8B, 90.0% with Outlines on LLAMA-3.2-3B, and 89.0% with Outlines or GenLM on QWEN2.5-7B. SWYB's decoding times are 3.96, 3.46, and 4.05 seconds per output, and its perplexities are 1.13, 1.14, and 1.11, respectively.

### Experiment 3

Test whether the decoder preserves syntax while improving executable SQL. Run SWYB on Spider's 1,034-example validation split with beam width 4 and alpha values of 0.25 for LLAMA-3.1-8B-INSTRUCT, 0.75 for LLAMA-3.2-3B-INSTRUCT, and 0.50 for QWEN2.5-7B-INSTRUCT; compare with BASE LM, SAMPLE-VERIFY, SynCode, and GenLM using NP = 5 for the Llama models and NP = 4 for Qwen. SWYB obtains 100.0% syntactic correctness on every model and execution accuracies of 61.1%, 51.5%, and 62.8%, respectively. The highest non-SWYB execution accuracies are 59.2% from GenLM on LLAMA-3.1-8B, 49.4% from GenLM on LLAMA-3.2-3B, and 61.5% from SAMPLE-VERIFY on QWEN2.5-7B. SWYB's decoding times are 9.92, 6.25, and 13.03 seconds per output, and its perplexities are 1.22, 1.43, and 2.10.

### Experiment 4

Test whether the result extends to LTL formulas and semantic equivalence. Run SWYB on the 6,185-example LTL drone-planning dataset with beam width 4 and alpha values of 0.75 for LLAMA-3.1-8B-INSTRUCT, 0.50 for LLAMA-3.2-3B-INSTRUCT, and 0.50 for QWEN2.5-7B-INSTRUCT; compare with BASE LM, SAMPLE-VERIFY, and SynCode. SWYB obtains 100.0% syntactic correctness for all three models and Büchi-automaton-equivalence accuracies of 29.6%, 22.1%, and 31.6%, respectively. SynCode, the highest non-SWYB task-accuracy method, obtains 24.4%, 15.2%, and 31.0%, with syntactic correctness of 99.9%, 99.9%, and 100.0%. SWYB's decoding times are 1.69, 0.78, and 1.53 seconds per output, and its perplexities are 1.23, 1.27, and 1.09.

### Experiment 5

Separate the effects of beam search and distance-guided scoring on SQL generation. On Spider, evaluate four SWYB configurations: neither distance-guided scoring nor beam search, four beams without distance-guided scoring, distance-guided scoring with alpha = 0.5 and no beam search, and both distance-guided scoring with alpha = 0.5 and four beams. For LLAMA-3.2-3B, LLAMA-3.1-8B, and QWEN2.5-7B, execution accuracy is 26.0%, 22.1%, and 22.2% with neither component; 50.9%, 60.4%, and 61.6% with four beams only; 26.3%, 22.2%, and 23.8% with distance-guided scoring only; and 51.2%, 60.9%, and 62.8% with both. Syntactic correctness remains 100% in all 12 settings. Perplexity changes from 3.67, 2.77, and 4.11 with neither component to 1.41, 1.22, and 2.10 with both.

### Experiment 6

Separate the effects of beam search and distance-guided scoring on LTL generation. On the LTL dataset, evaluate the same four configurations: neither component, four beams only, distance-guided scoring with alpha = 0.5 only, and both distance-guided scoring with alpha = 0.5 and four beams. For LLAMA-3.2-3B, LLAMA-3.1-8B, and QWEN2.5-7B, task accuracy is 16.5%, 29.2%, and 30.5% with neither component; 21.8%, 29.4%, and 31.2% with four beams only; 16.8%, 24.9%, and 30.5% with distance-guided scoring only; and 22.1%, 29.6%, and 31.6% with both. Syntactic correctness remains 100% in all 12 settings. Perplexity changes from 1.43, 1.73, and 1.17 with neither component to 1.27, 1.23, and 1.09 with both.

### Experiment 7

Rule out insufficient generation length as the explanation for the JSON failures of comparison decoders. Repeat the JSON experiments for Guidance, Outlines, SynCode, and GenLM with NP = 2 and NP = 5 at max_new_tokens values of 512, 1024, 2048, and 4096 instead of the default 120. Across those budgets, Guidance's schema validity remains 90% for LLAMA-3.1-8B, 85% for LLAMA-3.2-3B, and 91% for QWEN2.5-7B. Outlines remains 91% and 92% on the two Llama models and gives 93%, 93%, 91%, and 93% on Qwen. SynCode remains 83% on LLAMA-3.1-8B, gives 79%, 79%, 80%, and 79% on LLAMA-3.2-3B, and remains 92% on Qwen. GenLM schema validity stays between 89.8% and 92.9% across models, particle counts, and budgets. SynCode's time on LLAMA-3.1-8B rises from 13.27 seconds at 512 tokens to 113.81 seconds at 4096 tokens, while its schema validity remains 83%. No tested method reaches 100% schema validity at any of the four enlarged budgets.
