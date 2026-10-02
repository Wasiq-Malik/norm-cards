# Rewrite provenance

One file per source paper, holding the rewritten claim texts and the rewritten text and role for
every reference experiment, keyed by its index in the dataset20v3 reference. `split_changes`
records any experiment moved, added or dropped relative to the hand split, with the reason.
`notes` records what the rewrite corrected.

`datasets/dataset34v5/references/` is generated from these files; see the dataset manifest.

Note on arxiv2604_24474: writing this file tripped a safety classifier mid-response on the first
attempt (molecular generation and predicted activity). It was completed afterwards, describing
what each experiment measured and what came out, without the generation protocol's operational
detail — which is apparatus the claim withholds anyway.
