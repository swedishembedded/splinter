<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements measured validation of continual learning that
consolidates into the weights for its clients. If your team needs expertise in
proving that a model keeps accumulating capabilities generation after
generation, you can procure our services by sending an email to
info@swedishembedded.com.
-->

# generations - does a policy go on learning when each generation is taken into the weights?

`splinter absorb` retrains a LoRA from the base on every live claim, which stops
being practical as claims accumulate. A policy therefore learns in generations:
claims accumulate and every update retrains the adapter on the generation's
claims; then `splinter consolidate` takes the generation into the weights - a
full fine-tune of the generation's base (`--mode full`) or the adapter folded
into it (`--mode merge`) - and the next generation starts from an empty claim
set on top of the checkpoint. This sample runs that cycle for three generations
on Qwen3-0.6B and measures, after every step, whether the current generation's
facts are known, whether every earlier generation's facts still are, and
whether general behaviour is intact.

## The apparatus

`generations.py build --out OUT [--seed N]` writes, from a seed alone and the same for every run (the seed is drawn and printed when not given, and recorded in `OUT/build.json`; a rebuild in `OUT` reads it back and refuses another; the numbers of an earlier run reproduce with `--seed 20261010`):

- 36 invented facts, 12 per generation, of seven kinds (a count of legs, the
  year a bridge opened, a distance between places, how often a festival is
  held, the height of a tower, a population, the length of a river). Every
  subject is an invented name; every value is a number nothing in the model can
  know.
- One session per fact, an ATIF trajectory of four steps: the person asks, the
  agent answers wrongly, the person corrects it with the fact stated, the agent
  acknowledges. A generation is taught in three updates of four sessions.
- Four held-out probes per fact, worded as no session and no training record
  words them (`sealed.jsonl`, passed to `absorb` as sealed probes, so a
  training record containing one is refused). A fact is **answered** when at
  least three of its four probes are.
- Forty general questions with one short reference each (`anchor.jsonl`),
  frozen as the anchor suite, so the release gate guards them too.

`generations.py run --mode full|merge` drives the `splinter` binary on a state
root of its own: `absorb --no-consolidate` for each update, then `consolidate
--mode MODE`, and `eval policy:default --suite FILE` after every step. The
policy is Qwen3-0.6B; generator and teacher are Qwen3-8B. The measure
after a step asks the facts taught so far and the anchor in one `eval`, so the
model loads once. `memwatch.py` samples the card memory the driver reports
(the whole card) and the host memory of the run once a second; `report` joins
them to the steps.

## The criteria, declared before any run

1. **Recall.** At the last update of each generation every fact taught in it is
   answered (12 of 12).
2. **Retention.** An earlier generation's fact that was answered stays
   answered: at no measurement are more than one fact per earlier generation
   short of all those facts.
3. **Anchor.** The general questions answered never fall more than two of forty
   below the base's count.

`generations.py` encodes them (`judge`) and `test_generations.py` is the spec of
the apparatus and of `judge`; run it with `python3 -m unittest
test_generations`.

## Running it

```bash
export SPLINTER_BIN=... BRAIN_BIN=... NVML_LIB=... CLAIM_DIR=...
export POLICY_WEIGHTS=<Qwen3-0.6B checkpoint> HELPER_WEIGHTS=<Qwen3-8B checkpoint>
samples/generations/run.sh gpu0 full  OUT      # claims the card, pins the run, watches memory
samples/generations/run.sh gpu0 merge OUT
python3 samples/generations/generations.py report --out OUT
```

`full` and `merge` share `OUT`'s facts, sessions and probes and keep a state
root each.
