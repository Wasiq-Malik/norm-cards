# Checking the split against an independent labelling

Before the per-claim split existed, the plan was to **scope** each reference: ask a model, per
experiment, whether the claim required it, and drop the rest from the denominator. Two models
labelled all 216 experiments of dataset20v3 that way (`scope.json`, since removed).

That approach was superseded by the split, which reassigns an experiment to the claim it
actually decides instead of only deciding whether to keep it. Before removing the scope labels
they were compared against the split, since they are an independent opinion on the same
question and the split is one person's reading.

| | count |
|---|---|
| both keep it | 99 |
| both drop it | 63 |
| **scope keeps it, the split drops it** | **0** |
| scope drops it, the split keeps it | 54 |

**Nothing both labelling models called required was dropped by the split.** The 54 one-sided
cases are all experiments the split moved to a *different* claim of the same paper — something
the scoping question could not express, since it could only ask "does THIS claim need it".

So the split keeps everything scoping would have kept and correctly rehomes 54 experiments
scoping would have discarded. That is the reason the scoping machinery was removed rather than
kept alongside.

Agreement overall is 75%, and all of the disagreement is the split being more complete.
