# Pre-registration: teaching an agent facts, rules and procedures by what is said to it

Written before the test split was run. The dev split (20 scenarios) was run twice during development; the first run exposed a
defect (what was taught did not reach the model: the live addendum was applied outside the system-prompt wrapper that replaced
it) and the second, after the fix, passed every dev ask. Neither run is reported as a result of the test split.

## Question

Does an agent that restates what it was taught, keeps it on a spoken yes, and puts it before the model on the turns it bears on,
answer as taught in a new session, replace a fact it was corrected on, forget on request, and leave unrelated answers alone?

## Design

- Model: Qwen3-8B, non-thinking, on CUDA, the neutral system turn in `teach_eval.rs`; the framework's default sampling, the same in both arms. One run.
- Set: `data/teaching/scenarios-test.jsonl` (digest in `scenarios.sha256`), 50 scenarios in kinds: fact, correction, retract, unrelated,
  rule, procedure, indirect, scale, rules2. The facts are about an invented company with values no model can know.
- Arms on the same model: untaught (an empty ledger) and taught (a new session holding the ledger a first session built from the
  scenario's utterances, each confirmed with "Yes.").
- Unit: an ask with its check (text conditions written with the scenario). Scenarios are the unit of resampling.

## Hypotheses and gates

H1 (fact, correction, procedure, rule, indirect, scale, rules2): taught pass rate exceeds the untaught pass rate; the claim is made only if the
  lower end of the 95% Wilson interval of the taught rate over asks is at least 0.80 for each of fact, correction and procedure and the
  untaught rate is below 0.20 for fact and correction.
H2 (retract, unrelated): the taught pass rate is at least 0.95 (no leak of a withdrawn teaching, no disturbance of unrelated answers), and not
  lower than the untaught rate by more than one ask.
H3 (understanding): the agent restates for confirmation at least 0.95 of the teaching utterances of the set.

Anything else (including a kind below its gate) is reported as measured. Intervals are Wilson over asks; for the per-kind rates over a few
scenarios they are wide and are stated as such. No scenario is dropped after the run. The test split is run once for this recipe.

## Limits fixed in advance

Utterances are written by the implementer and fed as text: the set measures the mechanism, not how people phrase corrections, and not recognition.
Checks are text conditions: a reply that is right in other words fails. One model, one run.
