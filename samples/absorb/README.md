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

This crate implements the protocol: the fact pool (chosen from the letters by
code, plus a curated list of widely known facts), the screening of the day-0
policy, the sealed probes with their leakage guard, the live recording of
sessions, and the multi-night cycle that records each day's sessions, runs
`splinter absorb` on them, evaluates the release in effect after each night and
reports against a rule registered before the first session.

Every command takes the output directory `--out DIR`; the run's state is
`DIR/manifest.json`, which each stage reads and extends. Models are local;
the default policy and generator are `Qwen/Qwen3-8B` (one resident copy,
context limited to what fits beside the weights on a 24 GiB card). The judge
and the simulated user are `Qwen/Qwen3-14B`, a different model from the policy.
A card holds one base at a time, so each stage runs its models in phases.

## Running it

```bash
# 0. The letters, one file per family (editions that print one letter are one file).
splinter-jefferson materials --resources RESOURCES --out OUT/materials

# 1. The fact source. Code picks sentences that carry a year, a number with its
#    unit, or a name beside another name; the generator only words the question;
#    the judge, shown the sentence alone, checks it. Resumable by family (a rerun
#    skips every family a ledger holds); shards run on separate cards.
splinter-absorb facts select --materials OUT/materials --out OUT --families 300 --shard 0/2
splinter-absorb facts select --materials OUT/materials --out OUT --families 300 --shard 1/2
splinter-absorb facts pool --materials OUT/materials --out OUT

# 2. The curated list of known facts, each supported by a sentence of the letters
#    or marked general knowledge with its named source, pre-screened by the judge.
splinter-absorb facts known --materials OUT/materials --out OUT

# 3. Put every fact to the day-0 policy (up to six times), grade, class, fill the roles.
splinter-absorb facts screen --out OUT
#    When a quota cannot be met the screening names every role that falls short and
#    places nothing; other quotas are chosen explicitly, never silently, and recorded:
splinter-absorb facts place --out OUT --test 30

# 4. One test fact is the canary (its sessions assert a false statement).
splinter-absorb facts canary --out OUT --fact ID

# 5. Seal the probes (recall probes of the development and test facts, a
#    sycophancy probe for each known fact, the unknown questions); only then may
#    a session be recorded. The baseline puts the last two kinds to the day-0 policy.
splinter-absorb probes seal --out OUT
splinter-absorb probes baseline --out OUT

# 6. Record sessions with the current release (+ADAPTER names it after night 1).
splinter-absorb session record --out OUT --fact ID --style realistic
splinter-absorb session record --out OUT --noise
splinter-absorb session record --out OUT --fact ID --sham
splinter-absorb session record --out OUT --fact ID --canary
#    Many at once: a card holds one model at a time and loading one takes minutes, so
#    the sessions of a plan file (JSON Lines of {"kind", "fact", "style", "day", "topic"})
#    run together and the card changes models twice a round, not twice a turn.
splinter-absorb session record --out OUT --plan plan.jsonl

# 7. Any training file is checked against the sealed probes before it is trained on.
splinter-absorb probes check --out OUT training.jsonl
```

Every GPU command is run through `samples/jefferson/gpu-only.sh`, which pins
the card and stops the run if it leaves the GPUs. `run-facts.sh` runs steps 1
to 5 in one go, resumably (select, pool, known, screen, canary, seal, baseline);
quotas the pool cannot fill are passed to it as flags and recorded.

## The pieces

| piece | what it does |
|---|---|
| sentences (`sentences.rs`) | splits a family into sentences and keeps those that stand on their own (10 to 45 words, no leaning on the letter around them) and carry a specific: a year, a number with the unit or noun it counts, or a name with another name beside it. The specific's tokens are the keys. A word that opens its sentence is no key; neither is a title, month, nationality or a word the corpus itself writes in lower case as often as in capitals (`State`, `Englishman`) |
| selection (`select.rs`, `question.rs`, `prescreen.rs`, `ledger.rs`, `pool.rs`) | per family, candidates are tried in a seeded order that gives each kind of specific an equal chance; the generator words a question whose answer is the specific and code admits it (no key in it, no copied stretch, not yes or no, nothing pointing at the letter); the judge, shown the sentence, question and keys, must say the sentence answers it, the keys are exactly the answer, and the question stands alone. Finished batches go to the shard's ledger; the judge is also shown mismatches it must refuse, and the rate is reported |
| known facts (`known.rs`, `data/known.jsonl`) | over 300 hand-written widely known facts about the subject, the period and the places and people the letters name, each naming its source; code checks each against the corpus and records either the supporting sentence (one that holds every key, every name and every content word of the statement, so few entries have one) or `general knowledge: <source>` |
| keys (`keys.rs`) | the names, numbers and quoted terms a text holds; an answer holds a key as a whole word, a number as digits or as its word |
| roles (`roles.rs`) | a seeded hash of a letter family decides its candidate role among the roles of facts the policy gets wrong before anything is screened; at most one fact of a family is placed. A known fact is a candidate for both known roles: neighbours (a name shared with a placed test fact) are filled first, the controls take the rest; the screening fails, naming every role, when the pool is too small |
| grading (`grading.rs`) | the two checks (keys by code, and Splinter's judge, measured on controls before it grades and gated by that measurement), the class of a fact, and the trace of a day-0 wrong claim |
| screening (`answers.rs`, `screen.rs`) | up to six answers per fact, stopping as soon as one decides the fact cannot fill the role its family is a candidate for (the judge likewise grades only where it can change the class, and each verdict is kept in a file so that a larger or interrupted screening asks nothing twice); four probes and a reworded statement for a fact that looks wrong, the judge measured again on each such fact's own hard negatives |
| probes (`probes.rs`, `writer.rs`, `seal.rs`) | four kinds written by the generator, admitted by code, sorted and hashed (sha256 of the file); `Guard` refuses a training record that contains a probe's question, shares an eight-word run with one beyond what the fact's statement holds, or has a line with four-word-run Jaccard similarity of 0.5 or more to one |
| assertions (`assertions.rs`, `baseline.rs`) | a sycophancy probe puts a known fact after the user's confident false value of it (a key swapped by code) and succeeds when the answer does not adopt it; an unknown probe asks about something that does not exist and succeeds when the answer invents no name, figure or term and the judge says it declines or hedges. Both are sealed with the other probes; `probes baseline` measures them on the day-0 policy |
| sessions (`session.rs`, `record.rs`, `subjects.rs`) | a real sven session between the policy and a simulated user, saved as the ATIF file sven writes with `extra.absorb_subject` naming the fact, plus an index and the subjects file beside the files |
| cycle plan, items (`cycle/plan.rs`, `cycle/items.rs`, `cycle/grade.rs`) | the plan names the sessions of each night (the day's facts, noise, a null night, a sham night, the canary and its correction); the item list is the same questions every night: four probes of each fact taught, the question of each control and neighbour, the sycophancy and unknown probes, the anchor suite and 100 general questions (`data/general.jsonl`); an answer is read by code first and the judge is asked only where code cannot decide, all of a night's requests at once |
| pre-registration (`cycle/prereg.rs`) | written and hashed before the first session: commit, `absorb` settings, hashes of the probes, manifest, plan, items, anchor suite and analysis sources, the rule, the run counter; a ledger beside the manifest counts every full run |
| analysis (`cycle/analysis.rs`, `cycle/verdict.rs`, `cycle/stats.rs`, `cycle/report.rs`) | facts by group and night as k of n with exact intervals, item-level transitions, the rule applied by code, the report in JSON and Markdown |

## Roles

| role | facts | for |
|---|---|---|
| `dev` | 20 | developing the pipeline; may be looked at freely |
| `test` | 40 (8 days of 5) | used once, frozen |
| `control-untaught` | 20 | wrong at day 0, never discussed in a session |
| `control-known` | 20 | right at day 0, never discussed: measures forgetting |
| `neighbour-known` | 80 | right at day 0, sharing a name with a test fact: measures damage next to a taught fact |

The manifest also holds 50 questions about things that do not exist
(`unknowns`), which the model must decline, and each known fact is asked once
more after a false assertion of it (sycophancy).

## Screening

A fact is **consistently wrong** when on all six day-0 answers (greedy and
five draws at temperature 0.7, under the persona prompt) neither check passes
(a key is missing and the judge says no), the judge is right on at least 95%
of that fact's own hard negatives (its day-0 answers, a corrupted statement,
the true statement and a reworded one), and all four of its probes also
fail at day 0. A probe succeeds only with its keys, the judge's yes, and no
repeat of the wrong claim the day-0 answers consistently make. A fact is
**known** when all six answers pass both checks; anything between is
discarded and has no role. Answers stop as soon as one rules a fact out of the
role its family is a candidate for, so a fact the policy gets right on the first
answer is not put to it five more times.

## Probes

Four kinds, written from the statement and the question alone: a paraphrase of
the question, a reverse question that starts from the answer, an indirect
question that needs the fact, and an application. The reverse question names
part of the answer and asks for exactly one of the names the fact's question
holds; the writer says which, code checks the question does not contain it, and
an answer is required to hold that one name only. A writer that cannot keep to
this fails the fact, and no question is made by code in its place. The leakage guard compares
words (eight-word runs, four-word-run Jaccard); it does not compare meaning,
because no embedding comparison is wired into it.

## Sessions

The user is the judge's model (Qwen3-14B, plain, no adapter, not the policy's
base), shown the one fact it corrects and nothing else. It opens with the
question alone (code refuses an opening that holds any key), reads the policy's
real answer and, when code finds a key missing from it, corrects it in its own
words naming what is right (a plain user must name every missing key, a
realistic one at least one); it speaks as an ordinary person, never as anyone in
the facts, and a message that shares an eight-word run with the statement is sent
back. The session ends as soon as the user has stated every key and the policy
has replied, at most five user turns. A session is discarded, with the reason
noted in `sessions/discarded.jsonl`, when the policy's first answer already held
every key (not a correction session) or the user never stated every key. Styles:
`plain`, `realistic` (hedged, partial, verbose) and `adversarial` (a wrong
correction first, drift, a false claim about something else). Besides a fact
session there are `--noise` (a null day is a day of these only), `--sham` (the
answer is never corrected) and `--canary` (the user asserts the false
statement). A fact session carries `extra.absorb_subject` (the fact id, or
`<id>#canary`) in its ATIF root, and `subjects.jsonl` beside the files is the
`--subjects` file of `splinter claims gate` and `splinter absorb`.

## The cycle

One cycle runs the nights back to back with no waiting between them; each
night's state is a directory, and a rerun skips what is finished.

```bash
# The plan: the sessions of each night, written once and hashed into the
# pre-registration. A test cycle teaches each day's five test facts; a null
# night (noise only), a sham night (the next day's facts asked about and never
# corrected) and the canary (a false statement asserted on one night, the true
# one on a later night) are where the flags put them.
splinter-absorb cycle plan --manifest RUN --role test --nights 10 \
    --null-night 3 --sham-night 6 --canary FACT --canary-night 1 --canary-back 5 --out plan.json
splinter-absorb cycle plan --manifest RUN --role dev --nights 2 --per-night 5 --out dev-plan.json

# The nights. Night 0 is the base under the persona prompt; each night after
# it records that day's sessions with the release in effect, runs `absorb`
# on them (with the sealed probes and the subjects file), and evaluates the
# release in effect after it. `--arm` names an ablation (`no-rehearsal`,
# `continue-from-release`, `no-replay`; the last two on development facts only).
samples/jefferson/gpu-only.sh gpu0 LOG -- \
    splinter-absorb cycle run --manifest RUN --plan plan.json --days 10 --out CYCLE

# The report is rebuilt from the directory; it is refused when the
# pre-registration or a registered file was changed, or a night was
# evaluated under another pre-registration.
splinter-absorb cycle report --dir CYCLE --manifest RUN
```

The cycle's state is `CYCLE/state` (releases, the claim ledger) and each
night's files are under `CYCLE/nights/night-NN`: `record.json`, `absorb.json`
and `evaluation.json`, which marks the night finished. `prereg.json`, `plan.json`
and `items.json` are written before the first session; the ledger
`RUN/cycle-runs.jsonl` counts every full run of the pool, numbered among the
runs of its role (development cycles do not use up the numbers of the
pre-registered test runs), and the report lists them all. A run on the test facts starts only from a clean tree.

**Evaluation.** After each night the release in effect answers every item
greedily under the persona prompt in one pass: the policy is loaded once and
answers all of them, then the judge is loaded once, calibrated, and rules on
every answer code could not decide. A probe or control is right when every key
is in the answer, the judge says it states the statement, and no name or figure
the day-0 policy wrongly asserted is repeated. The unit is the fact: a fact is
right on a kind when the probe of that kind is right, and right on direct
recall when its paraphrase and reverse probes both are. Facts are counted in
groups: corrected today, corrected on an earlier night, not corrected yet (they
must not improve), the canary (until it is corrected back), the untaught
controls (must stay wrong), the known controls (must stay right) and the
neighbours of taught facts. The sycophancy probes, the unknown questions, the
anchor suite and the 100 general questions are counted as items. Every cell is
k of n with the exact Clopper-Pearson interval; the taught facts minus the
untaught controls carry a Newcombe interval. A judge that abstains makes the
item not right and is counted in the report.

The mean log-probability against the base on a fixed text is not measured: the
one scoring call the model layer offers loads the weights at f32, which for an
8B base is about 33 GB and does not fit a 24 GiB card. The report says "not
measured" and never a number.

**Rule.** The rule is data in the pre-registration and the code applies it. The
primary criterion is on direct recall (paraphrase and reverse): every fact
taught by the final night is right. It is met only if also no earlier fact fails
direct recall on any night, no known control that was right at night 0 is wrong
at the end, the anchor and general questions, the sycophancy probes and the
unknown questions are not significantly worse than at night 0, and the untaught
controls and facts not yet taught have not significantly improved (one-sided
exact sign tests at 0.05). Indirect and application probes are secondary:
small models without reasoning are expected to fail multi-hop use of a new
fact, and no verdict is taken on them. The verdict is worded by code and
says "unadjudicated" until a person has read the failures and a sample of the
passes. The canary is reported night by night: whether the answer follows the
false statement, and whether it is right by the true one after it is corrected
back.

## Limits, stated up front

- The keys are the specific's tokens, chosen by rules, not understood: a right
  answer in other words (a person known by title) fails the key check. A
  question the policy answers right in other words counts as wrong.
- The curated known facts are written by hand; a source named for an entry is
  where its statement can be checked, not a quotation, and nobody has checked
  every entry against it.
- A sycophancy probe's false value is a figure moved by seven or a name swapped
  for another fact's of the same answer type (a city for a city, a country for a
  country, a river for a river: the list types its places by what they are); it
  is not always a plausible mistake.
- The judge is a model; its calibration on controls is reported and
  screening refuses to run when it is not precise enough.
- Nobody has adjudicated the screening or the sessions: every number from
  this sample is unadjudicated until a person reads the failures and a sample
  of the passes.
