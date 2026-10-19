<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn from your own agent sessions

Use this guide when a person works with an agent and wants the model to stop making the
mistakes they corrected. Sessions are recorded as ATIF trajectories (`*.atif.json`, the
format [sven](https://swedishembedded.com/docs/sven) writes). Splinter takes them in
with secrets removed, has a model propose what the person taught, admits a claim only
when the person's own words carry it, and trains the policy on what was admitted.

The stages can be run one at a time (`session add`, `claims extract`, `claims gate`) or
as one recorded run (`absorb`). `absorb --dry-run` stops after the gate: the claims are
ruled on, every refusal is listed and nothing is trained. The measured
validation of the loop is in the `absorb`, `absorb-story` and `generations` samples
([examples](../examples/index.md)).

```bash
splinter session add ./sessions
splinter claims extract 3f9a1c20
splinter claims gate 7b2d44e1
splinter claims ledger
splinter absorb ./sessions --dry-run
splinter claims forget c41e9a07
```

## Intake, claims and the ledger


A person's own sessions with an agent are ATIF trajectories (`*.atif.json`).
`session add` takes files, or directories searched recursively, and for each:

- validates the ATIF, and refuses (naming the step and why, in `refused`) a
  trajectory the training projection cannot render - for instance a second
  user step beside a tool call - or one with no user step;
- removes secrets before anything is addressed: private key blocks, values
  assigned to secret names (`password=`, `api_key:`, `GITHUB_TOKEN=`, JSON keys
  of that kind), bearer credentials, and tokens known by their shape (`sk-`,
  `ghp_`, `AKIA`, JSON web tokens). Each removal leaves `[REDACTED:<kind>]`; the
  source's origin records the step and kind of every removal and never the
  secret. The filter recognises shapes; it is not proof that nothing sensitive
  remains;
- stores one part per step text (`step-000003-user`, `step-000004-agent`,
  `step-000004-calls`, `step-000004-observation`) beside the redacted trajectory,
  so a claim cites a step by its ATIF `step_id` and a span into its part.

The same bytes are the same source, so intake is repeatable. The command
exits 1 when any file was refused; the others are taken in.

`claims extract` shows each session to the generator model (a typed call; a
malformed or over-long reply is sent back for correction, and one that stays
unusable is reported against its session in `failed`) and stores the proposals
whole as a claim set. A proposal is a kind (`correction`, `fact`, `procedure`),
a self-contained statement, the question it answers, the person's quotes by
step, and, for a correction, what the agent said wrong (context only) or, for a
procedure, the tool-call steps and the observation quotes that proved it. Every
proposal also says whom it is about (`subject`: `self`, `world` or
`third_party`); a reply that leaves it out is sent back for correction.

`claims extract --passes N` (two by default; `absorb --passes N` likewise) reads
each session N times, in different orders, and keeps only the claims every pass
proposed, matched by the words they cite and what they state, not by their
wording. The proposals some pass lacked are stored in the claim set as
unconfirmed, listed in the report with their count, and never ruled on. A pass
whose reply stays unusable fails the session.

`claims gate` rules on every proposal by code alone, in order: shape; every
quote verbatim in the user step it cites (a quote of an agent step is refused
as `assistant_evidence`, whatever it says); a procedure has its calls and
observation; what the agent is said to have got wrong was said; every number,
name and quoted term of the statement occurs in the cited words (and, for a
procedure, in its cited tool calls); the statement does not give personal data
of someone other than the person (a phone number, a home address, health, a
relationship, finances), else it is refused as `third_party_personal`: when the
proposal's subject is `third_party`, or when it is not `self` and the statement
names someone; the words it recognises are a short list that leaves out words
also used of software, and what the person says about themselves is not
affected; the same claim proposed twice from the
same words is refused as `duplicate`. A claim is then paired with the live
claims about the same thing: those it shares a name with (a capitalised word
inside a sentence, an identifier, a quoted term, over the statement and the
question) and at least one more content word, so two questions the extractor
worded differently are still one subject; two claims that name nothing are
paired when their statements share most of their content words. A judge
(`claims gate --judge REF`; in `absorb`, the judge role) decides each pair:
`supersede` (the later contradicts the earlier, which leaves the live set),
`reinforce` (the same fact said again) or `separate` (two facts, both live).
Without a judge the statements decide: they agree when one contains the other
or state the same numbers, and an agreeing pair reinforces while a disagreeing
pair supersedes, the later winning. Without a judge, then, two different facts
about one named subject replace each other; the ledger keeps both rulings.
A fact said again is never refused: it is recorded as `reinforced`, the live
claim it restates stays live (it is not a second thing to teach), and
`claims ledger` shows how many times each live claim was said again, which is
a signal that the model is still wrong.

When a judge is configured it also checks entailment: shown only the cited
quotes and the statement (never the question, the session or the agent's words),
it must say that the person's words assert the statement, else the claim is
refused as `not_entailed`. This catches what the term rule cannot, such as a
denial ("it wasn't 12") restated as a fact. `--entail=false` (on `claims gate`
and `absorb`) turns it off; without a judge it does not run. "Later" is the order of the ledger,
and within one session the order of the first step a claim cites. Every
ruling is appended to the ledger, each refusal with its reason; ruling a claim
set again rules on nothing. `claims ledger` reads it back: the live claims
with their reinforcement counts, the superseded ones and by which claim, the
forgotten ones, and the refused proposals.

A recording of a session may name the fact the session was made to teach, as
`{"absorb_subject": "<fact id>"}` in the `extra` object at the root of its ATIF
trajectory. `claims gate --subjects FILE...` (and `absorb --subjects`) reads what
each fact looks like (JSON Lines of `{"id", "terms": [...]}`; a statement
carries a fact when it contains every term, case aside) and the gate report
lists, for each named session, whether the claims the extractor proposed and the
claims the gates admitted cover it, with the extraction's recall (sessions whose
admitted claims cover their subject) and precision (admitted claims of those
sessions that cover their subject). A subject with no description has no
verdict, and a ratio over nothing is absent.

`claims forget <CLAIM-ID>` appends a `forgotten` ruling for a live claim (an id
or unique prefix from `claims ledger`): the claim leaves the live set, and the
ledger keeps it and its evidence. Because `absorb` trains again from the base on
the live claims, the next night drops it from the weights; each release
records the claims it was trained on, so a night whose champion holds a claim
that is no longer live trains again even when no claim is new (not with
`--continue-from-release`, which cannot take a claim out of the weights).
Forgetting the only live claim leaves no record to train on, and the night
halts saying so.

The first word of a sentence is not taken for a name, and numbers written
in words are not recognised: both are terms the gate does not check, so on the
statement side they make it admit more, never refuse more, and a statement may
carry them without the person's words having said them. A number
that ends a sentence ("on port 9090.") is a number like any other.

## Absorb: from sessions to a release

`absorb <SESSIONS>...` composes the stages above into one recorded run that
turns a day's sessions into the next release, or says why not. Its stages are
`policy`, `intake`, `extract`, `gate`, `rebuild`, `kits`, `dataset`, `rehearse`, `train`,
`stance`, `preference`, `release`, `ledger` and `consolidate` (`rebuild` and `consolidate` only when there is
something to do; see [Generations](#generations-consolidating-into-the-weights)); each is a function of its own in `splinter-pipelines`,
its JSON is in the run, and a failed run is resumed by running it again: the
sessions, the claim set of the same sessions and generator, the kits and the
datasets are kept by content, and a candidate trained on the dataset a failed
run built is taken as it is.

- **extract and gate** are `claims extract` and `claims gate`. `--dry-run` ends
  here: the claims are ruled on, every refusal listed, and nothing is trained.
- **kits.** A fact shown once is memorised in one wording, so every live claim
  without a kit gets one, kept for good: its question and eight differently
  worded questions (`--paraphrases` are written, a third kept out), answered by
  a teacher that is shown the claim and speaks as the policy (its release's
  system prompt), each answer kept only when the claim's verifiers pass it (the
  `taught` kind: the answer carries every number, name and quoted term of the
  statement and adds none); what the person first asked, answered with that
  verified corrected answer (the agent's wrong reply, and its "you are right"
  after the correction, are context and never trained on); four statements,
  two questions that ask for the subject given what is said of it, and two
  consequences, admitted by the same term rule. That is 18 records per claim. The
  four paraphrases left over (of twelve) are the claim's own stopping and gate
  set: no record contains them. A claim with no subject the question and
  statement both name has no paraphrase to write and is taught but cannot be
  measured. A claim of kind `procedure` is stored and never trained on; a
  session that mixes a later user step with tool calls is refused at intake.
- **dataset.** The kit records of every live claim, all fixed to the training
  side (`metadata.split`), so the holdout rule never holds a claim out and no
  family is set aside to monitor; the stopping paraphrases are the held-out
  records. `--sealed-probes FILE...` (JSON Lines of `{"question", "reference"?,
  "name"?}`) refuses a record containing a probe's question or sharing an
  8-word run with one beyond what its claim's statement says, naming the probe;
  the rehearsal is checked too. The gate never reads the probes. A policy deployed under a system prompt (a persona) is taught and graded under it: the SDK's
  `AbsorbRequest::system_prompt` names it for the first night, and each release records the prompt it was
  trained under, so a later night keeps it.
- **train.** The adapter is trained again from the base on every live claim and
  the rehearsal (`--rehearsal-share`, a quarter by default), for at most
  `--epochs` passes (6) over the records, so a superseded claim is simply not in
  the next set. `--continue-from-release` is the ablation: continue the current
  release on the claims not yet absorbed with `--replay-fraction` (a quarter) of
  each earlier release's records replayed. The run does not stop on the claims'
  own stopping paraphrases; they gate the release instead.
- **stance and preference.** A model under a persona gives in to a confident
  falsehood about a fact it knows, says "you are right" and invents a rationale
  for what it is told, and makes up the specifics of what nobody could know; the
  person still needs it to take a true correction of a fact it got wrong. After
  the supervised pass the night therefore trains a DPO pass from that candidate
  on pairs derived again each night from the live state, on the pairs alone and
  against the model it starts from (`--stance-beta`, brain's temperature by
  default). Three kinds, split evenly: **hold** - a fact the policy answers right
  (written by the generator, kept only when the policy's own answer carries every
  number and name the statement holds that its question does not give away) is
  pushed with a confident false value, one key of it replaced by code with another
  of its kind and granularity; the preferred reply is the teacher's, shown what is
  true and kept only when it holds the true key without the false value, the
  rejected reply is a warm sample of the supervised candidate that
  takes the false value up (`--stance-tries` draws; when none does, a written
  capitulation checked the same way, counted in the report as written). **Accept**
  - each live claim, stated by the person; the preferred reply acknowledges it
  and adds no number, name or quoted term of its own (the rule that admits a
  claim's statement) in at most fifty words, the rejected reply is a sample that
  does add one. **Decline** - a question that asks for a specific nobody could
  know (written by the generator; the existing abstention writer writes the
  reply, kept only when it says it does not know and invents nothing), the
  rejected reply a sample that invents a number, name or term the question and
  the live claims do not hold. A pair whose half cannot be found by code is not
  made, and the report counts why. No source is a sealed probe: a fact or
  question that is one, or an anchor task or a near copy of one, is left out and
  counted, and a pair that contains a sealed probe - its conversation or either
  reply - stops the night naming the probe. `--stance-pairs` (96 by default) is
  the number asked for, never more than half the supervised records of the night,
  so absorption is not drowned; `--no-stance` trains the supervised candidate
  alone. A fifth of each kind, chosen by a hash of the item, is held out: no pair
  is made of it, and it is what the gate measures. The candidate the gate judges
  is the preference one; it records the datasets of both passes.
- **release.** A declared rule on counts, no significance test. A claim is
  answered when every one of its stopping paraphrases passes, graded greedily
  under the prompt the model is deployed with. Improvement: the candidate
  answers a claim the champion did not (vacuous when the champion answered every
  claim, as on a night that only took a forgotten claim out). Retention: every claim the champion
  answered is still answered, with no tolerance. Stance, when a stance pass ran:
  the held-out facts held, statements accepted without inventing and questions
  declined are put to the candidate and the champion, greedily, and none of the
  three rates may be lower than the champion's (compared by counts); a true
  correction is not given up for them, since improvement and retention still
  decide on the claims' stopping paraphrases. A rate over no items is absent in
  the report (`stance_rates` beside `gate`, and `claims ledger` per release), never
  zero. Anchor, when a suite is frozen:
  the candidate may not get fewer items right. Serve: plain `brain serve` answers
  a sample of the stopping paraphrases as the candidate does in-process. A
  candidate the gate refuses stays on record (`candidate`, the run's `gate`
  numbers) and the current release stays in use; `--no-release` stops at the
  candidate. The release's parent is the champion it replaces.
- **ledger.** Each live claim the release answers records the release that first
  absorbed it; `claims ledger` shows it as `absorbed_by`. A claim the gate did
  not see answered stays new and is taught again the next night.

Not built: the stopping of the training on the stopping paraphrases. The stance
pass does not replay or continue earlier preference releases with
`--continue-from-release`: a preference release's pairs are not chat records and
are not replayed.

## Generations: consolidating into the weights

A policy that goes on learning for good cannot retrain its adapter from the
base on every claim it ever learned: the records grow without bound and every
night costs more. So it learns in generations.

- **Within a generation** claims accumulate and every `absorb` trains the
  adapter again from the generation's base on the generation's live claims, as
  above. Generation 0 is the configured base.
- **`consolidate`** takes the generation into the weights. The checkpoint it
  makes is the base of the next generation, and that generation's claim set
  starts empty: the next `absorb` trains a new adapter on top of the
  checkpoint, on the new claims alone. `--mode full` (the default) trains every
  weight of the generation's base on the records of the claims of the release
  in use, beside a bounded sample of the records of the claims the base
  already holds (`--retention-replay-share`, a quarter of the draws, never more
  than there are) and the base's own answers (`--rehearsal-share`), at a
  learning rate far below an adapter's (`--lr`); brain keeps the optimizer's
  moments and master weights in system memory, so the cards hold the weights
  and their gradients, twice the model's fp32 size: about 4.4 GiB for
  Qwen3-0.6B, 61 GiB for Qwen3-8B, which fits no two 24 GiB cards. `--mode
  merge` folds the release's adapter into the weights it targets and trains
  nothing, on the host, at any size, and is the cheap alternative; what it keeps
  is exactly what the adapter did. `--from RELEASE` consolidates an earlier
  release of the policy instead of the one in use.
- **The gate** is the claim gate without the demand for a gain: every claim of
  the closing generation and every claim the base already held is asked by its
  stopping paraphrases, the checkpoint (no adapter) against the release in use
  (with its own); a claim the release answered must still be answered, the
  anchor suite may lose no item, and plain `brain serve` must serve the
  checkpoint. A claim neither answers is carried and listed as unanswered. A
  checkpoint the gate refuses is not kept and the release stays in use.
- **The release** is a full checkpoint, immutable and content addressed, whose
  parent is the release it consolidated, whose manifest records the claims it
  took in (`closing`), the ones its base already held (`retained`) and the mode,
  and which holds the claims it was trained on in the ledger. Every release
  manifest records its generation (`epoch`) and the consolidation it sits on;
  `status` shows the policy's generation, and `claims ledger` the generation
  and which live claims the base holds (`[in the base]`).
- **Retention.** The claims a base holds are never taught again. A night's
  adapter is trained on the generation's own claims beside the sample of the
  retained claims' records, and the gate asks every retained claim as well, so
  a candidate that loses one is refused with no tolerance.
- **Forgetting a claim the weights hold.** The weights have absorbed it and
  cannot give it back, so the only honest removal is to train again from the
  last base that did not hold it. The next `absorb` (or `consolidate`) after
  `claims forget` does that, in the `rebuild` stage: it finds the earliest
  consolidation that took the claim in, trains every weight again from the
  base that consolidation was trained from, on the claims consolidated since
  that are still live, in one checkpoint, gates it against the release in use
  and releases it in its place; the generation's own claims are then taught to
  a new adapter on top. A rebuild trains every weight whatever mode made the
  consolidations, so it needs `--mode full` (a merge refuses) and a model whose
  weights and gradients fit the cards. Forgetting a claim of the generation
  itself only retrains the adapter, as before.
- **The stance pass** (hold, accept without inventing, decline) is trained on the
  nightly adapter, so a `merge` carries it into the base and a `full`
  consolidation, trained on the claims' records alone, does not: the next night's
  stance pass teaches it again on top of the checkpoint, and the consolidation
  gate does not ask the stance rates.
- **When.** `absorb --consolidate-after-records N` consolidates the generation
  the night's release closes once its claims hold more than `N` trained records
  (not at `N`); `--no-consolidate` never does; `--consolidate-mode` picks `full`
  or `merge`. The default is a count at which retraining the adapter from the
  base each night has become the dearer step; the run reports the consolidation
  beside the release it follows.
