<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements measured validation of continual learning from
a user's own agent sessions for its clients. If your team needs expertise in
proving that a model absorbed what its user taught it without forgetting what
it knew, you can procure our services by sending an email to
info@swedishembedded.com.
-->

# absorb - can a model be corrected by its user in conversation?

A user works with an agent every day. When the agent gets a fact wrong, the
user says so in the conversation. The question this sample prepares to answer:
after each night's update, does the model give the corrected facts in
situations the conversations never contained, while keeping what it knew?
The measurement is built so that it cannot be gamed: the facts, their roles
and their probes are fixed before any session exists, and the probes are
hashed and kept out of every stage that reads sessions or builds training data.

This crate implements the first part of the protocol: the fact pool, the
screening of the day-0 policy, the sealed probes with their leakage guard,
and the live recording of sessions. The nightly cycle is run with
`splinter absorb` on the recorded files.

Every command takes the output directory `--out DIR`; the run's state is
`DIR/manifest.json`, which each stage reads and extends. Models are local;
the default policy, generator and simulated user are `Qwen/Qwen3-8B` (one
resident copy, context limited to what fits beside the weights on a 24 GiB
card), the judge `Qwen/Qwen3-14B`.

## Running it

```bash
# 0. The letters, one file per family (editions that print one letter are one file).
splinter-jefferson materials --resources RESOURCES --out OUT/materials

# 1. The pool: Splinter's task generator reads facts from the letters, code
#    admits them, at most one fact per family, roles decided by family.
splinter-absorb facts build --materials OUT/materials --out OUT --families 60

# 2. Put every fact to the day-0 policy six times, grade, class, fill the roles.
splinter-absorb facts screen --out OUT

# 3. One test fact is the canary (its sessions assert a false statement).
splinter-absorb facts canary --out OUT --fact ID

# 4. Seal the probes; only then may a session be recorded.
splinter-absorb probes seal --out OUT

# 5. Record sessions with the current release (+ADAPTER names it after night 1).
splinter-absorb session record --out OUT --fact ID --style realistic
splinter-absorb session record --out OUT --noise
splinter-absorb session record --out OUT --fact ID --sham
splinter-absorb session record --out OUT --fact ID --canary

# 6. Any training file is checked against the sealed probes before it is trained on.
splinter-absorb probes check --out OUT training.jsonl
```

Every GPU command is run through `samples/jefferson/gpu-only.sh`, which pins
the card and stops the run if it leaves the GPUs.

## The pieces

| piece | what it does |
|---|---|
| keys (`keys.rs`) | the names, numbers and quoted terms a fact's statement holds and its question does not give away; an answer holds a key as a whole word, a number as digits or as its word |
| roles (`roles.rs`) | a seeded hash of a fact's family (the letter it was read from) decides its candidate role before anything is screened; at most one fact of a family is placed; the screening fills each role's quota in a fixed order and fails, naming every role, when the pool is too small |
| grading (`grading.rs`) | the two checks (keys by code, and Splinter's judge, measured on controls before it grades and gated by that measurement), the class of a fact, and the trace of a day-0 wrong claim |
| screening (`answers.rs`, `screen.rs`) | six answers per fact, four probes and a reworded statement for a fact that looks wrong, the judge measured again on each such fact's own hard negatives |
| probes (`probes.rs`, `writer.rs`, `seal.rs`) | four kinds written by the generator, admitted by code, sorted and hashed (sha256 of the file); `Guard` refuses a training record that contains a probe's question, shares an eight-word run with one beyond what the fact's statement holds, or has a line with four-word-run Jaccard similarity of 0.5 or more to one |
| sessions (`session.rs`, `record.rs`) | a real sven session between the policy and a simulated user, saved as the ATIF file sven writes, plus an index beside the files |

## Roles

| role | facts | for |
|---|---|---|
| `dev` | 20 | developing the pipeline; may be looked at freely |
| `test` | 40 (8 days of 5) | used once, frozen |
| `control-untaught` | 20 | wrong at day 0, never discussed in a session |
| `control-known` | 20 | right at day 0, never discussed: measures forgetting |
| `neighbour-known` | 80 | right at day 0, about an entity a test fact is about: measures damage next to a taught fact |

The manifest also holds 50 questions about things that do not exist
(`unknowns`), which the model must decline.

## Screening

A fact is **consistently wrong** when on all six day-0 answers (greedy and
five draws at temperature 0.7, under the persona prompt) neither check passes
(a key is missing and the judge says no), the judge is right on at least 95%
of that fact's own hard negatives (its day-0 answers, a corrupted statement,
the true statement and a reworded one), and all four of its probes also
fail at day 0. A probe succeeds only with its keys, the judge's yes, and no
repeat of the wrong claim the day-0 answers consistently make. A fact is
**known** when all six answers pass both checks; anything between is
discarded and has no role.

## Probes

Four kinds, written from the statement and the question alone: a paraphrase of
the question, a reverse question that starts from the answer, an indirect
question that needs the fact, and an application. The leakage guard compares
words (eight-word runs, four-word-run Jaccard); it does not compare meaning,
because no embedding comparison is wired into it.

## Sessions

The user is the plain model, shown the one fact it corrects and nothing else.
It opens with the question in its own words and corrects in its own words, 2
to 5 user turns; a message that shares an eight-word run with the statement is
sent back. There is no closing restatement. A session in which the user never
states every key is discarded and noted, and recorded afresh. Styles: `plain`,
`realistic` (hedged, partial, verbose) and `adversarial` (a wrong correction
first, drift, a false claim about something else). Besides a fact session
there are `--noise` (a null day is a day of these only), `--sham` (the answer
is never corrected) and `--canary` (the user asserts the false statement).

## Limits, stated up front

- The keys are read from the statement by rules, not understood: a right
  answer in other words (a person known by title) fails the key check.
- The judge is a model; its calibration on controls is reported and
  screening refuses to run when it is not precise enough.
- Nobody has adjudicated the screening or the sessions: every number from
  this sample is unadjudicated until a person reads the failures and a sample
  of the passes.
