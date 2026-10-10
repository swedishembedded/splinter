# Pre-registration: procedures and rules with enough asks (T2, second set)

Written after the first test split, in which the procedure kind had 6 asks and could not meet its gate, and before the test split of this set is run.

Sets: `data/teaching/scenarios-more-test.jsonl` (8 procedures, 3 asks each; 8 rules, 4 asks each: 24 and 32 asks) with vocabulary and rule forms not used in the first set. The dev split (`scenarios-more-dev.jsonl`, 2 procedures and 2 rules) is used for debugging and is not a result.
Design, model, arms, units and statistics as in `prereg/teaching-test.md`. Qwen3-8B, one run; the other models only under `prereg/teaching-models.md`'s rules if this set is later run on them.

Hypotheses:
- H4 (procedures): the lower 95% Wilson bound of the taught pass rate over asks is at least 0.80 and the untaught rate is below 0.20.
- H5 (rules): the same, pooled over the eight rules.
Per-rule rates are reported. A rule whose untaught arm already passes at least half of its asks is reported but cannot support H5 alone; if the pooled untaught rate is not below 0.20 the gate is not met for that reason and is reported so.
Limit: text checks (a start word, an end word, a word count, a word to include, a word to avoid); one model; utterances written by the implementer in the forms the plain reader reads.
