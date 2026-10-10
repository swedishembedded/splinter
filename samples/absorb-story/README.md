<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements measured validation of continual learning from
a user's own agent sessions for its clients. If your team needs expertise in
proving what a nightly update keeps, replaces and drops, you can procure our
services by sending an email to info@swedishembedded.com.
-->

# absorb-story - what the nightly loop does in the situations it will meet

The `absorb` sample asks whether a model can be corrected in conversation. This
one asks what `splinter absorb` does over many consecutive nights when the
user changes their mind, repeats themselves, contradicts themselves, lies,
asks to forget, leaks a secret or has a day with nothing to say. Every fact
is invented, so the base model cannot know it and nothing leaks, and every
session is scripted: the user's side plus a canned wrong or right agent turn,
written by code as the ATIF files `splinter absorb` takes, byte for byte the
same each time.

A storyline is a list of nights. After every night's `splinter absorb` the
same fixed evaluation asks every item introduced so far its question in three
wordings (greedy, closed book, the deployed `policy:default`) and grades each
answer by code on the values it shows. The result is a table of items by runs.

## The grading

An item has values (the right one, the stale ones, a rival entity's), each a
set of keys. An answer is reduced to word tokens, so `1,219` and `1219`,
`07:30` and `7:30`, `five` and `5` agree and a key matches whole words only.
Against the value that is right on that night an answer is `T` (that value
alone), `M` (it and another), `S` (another value only) or `O` (no value at
all). A cell is the count of letters over the three wordings, `3T` or
`2T1O`. What is right on a night follows from the sessions: the last thing
taught, a forgetting counting as teaching nothing.

## Predictions

Each situation has its prediction in `src/main_story/scenarios.rs`, committed
before any night ran. The runner freezes them, with the sessions and items,
into `storyline.json` of the run directory and refuses to continue a run
whose definition has changed since. The report prints each prediction beside
the answers that were quoted. A prediction is a statement code can check
(`Taught`, `Silent`, `Never`) or, for the open design question (the user
lying about something the base knows), prose.

## Storylines

| name | what it is |
|---|---|
| `main` | eight nights: contradiction, reversion, reinforcement, self-contradiction, interference, dependent facts, a lie, forgetting, a secret, noise and a null day, and six more situations: a correction as a question, a fact among twenty requests, a correction in the agent's voice, feet then metres, a name spelled two ways, a negation |
| `rerun` | night 1 of `main` in a fresh state: are the answers identical? |
| `scale-once` | fifteen facts in one night |
| `scale-spread` | the same fifteen facts, three a night over five nights |
| `reverse` | the sessions of `main` in the opposite order of days (chatter and forgettings dropped) |
| `continue` | the first six nights of `main` with `--continue-from-release` |
