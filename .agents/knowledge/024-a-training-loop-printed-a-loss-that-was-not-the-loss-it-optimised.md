# 024. A training loop printed a loss that was not the loss it optimised

Each micro-batch was one record and its loss was divided by that record's
supervised-token count, so every record weighed equally whatever its length and
the printed training loss was a mean of per-record means, not comparable with
the per-token held-out loss. Around it: warmup and cosine ran over the planned
steps, so early stopping ended a run far above the floor; records were drawn
uniformly with replacement, so a fifth of them were never seen at a short
budget; and the data recipe let a quota discard a quarter of the answers and
starve the writer's own text of tokens.

How to avoid it: compute the step loss as a mean over its supervised tokens,
anneal on stop as well as on schedule, draw records in shuffled epochs, and
measure a recipe's shares in tokens that reach the loss, not in records
offered. Read the code that produces a number, with file and line, before
trusting it, and diff a prepared file against its input: a file byte-identical
to its input exposed a rewrite that silently skipped every record.
