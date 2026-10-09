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
`DIR/manifest.json`, which each stage reads and extends.

## The pieces

| piece | what it does |
|---|---|
| keys (`keys.rs`) | the names, numbers and quoted terms a fact's statement holds and its question does not give away; an answer holds a key as a whole word, a number as digits or as its word |
| roles (`roles.rs`) | a seeded hash of a fact's family (the letter it was read from) decides its candidate role before anything is screened, so a family is never split; the screening then fills each role's quota in a fixed order and fails, naming every role, when the pool is too small |
| probes (`probes.rs`) | sealed probes, written once, sorted and hashed (sha256 of the file); `Guard` refuses a training record that contains a probe's question or shares an eight-word run with one beyond what the fact's statement holds |

## Roles

| role | facts | for |
|---|---|---|
| `dev` | 20 | developing the pipeline; may be looked at freely |
| `test` | 40 (8 days of 5) | used once, frozen |
| `control-untaught` | 20 | wrong at day 0, never discussed in a session |
| `control-known` | 20 | right at day 0, never discussed: measures forgetting |

Screening puts each candidate fact to the day-0 policy six times. A fact is
consistently wrong when no answer passes both checks (every key present, by
code, and the calibrated judge), known when all six do, and discarded
otherwise.

## Limits, stated up front

- The keys are read from the statement by rules, not understood: a right
  answer in other words (a person known by title) fails the key check.
- The judge is a model; its calibration on controls is reported and screening
  refuses to run when it is not precise enough.
