<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Datasets reference

A dataset is training data projected from experience sets by one view, stored with a manifest
that names where every record came from. `dataset build` makes one, `dataset export` copies
it with its manifest into a directory, and `train` reads it. The guide to using them is
[Build datasets and train a candidate](../guides/training.md).

## Views


`dataset build` views: `sft-final`, `sft-step`, `critic`, `preference` (a prompt and two
continuations of it, a verified pass preferred over a verified fail; see below), `feedback`
(each verified attempt's continuation, desirable when it passed, for KTO; see below),
`verifier`, `decision`, `retrieval`, `outcome`, `denoise`, `cpt`, `voice`
(the writer's own text as answers, from every text part of the sources the
experience sets cite; see [Learn how a person thinks](../guides/personas.md)), `rehearsal` (a model's own answers as it
gave them, every experience with a final answer that no verifier decided
wrong, under the default prompt), `policy-gradient` (every action of every graded attempt,
weighted by the attempt's advantage; see below), `self-distill` (the student's own conversation
of every attempt with a teacher context built from what the verifier knows; see below), `transition`
(what the environment answered: one record per attempt, passed or failed, each tool observation
supervised in the context through the action it answers and no model turn supervised; an attempt
made open-book is counted as `open_book`; one transition (task, call, observation) is supervised at
most `--per-transition-cap` times, default 2, the rest `repeated_transition`; `--loss-weight W` writes
`loss_weight` on every supervised observation, which brain scales their loss by; `--control shuffled`
gives each observation another's text as the negative control). `--limit N` keeps at most N records of any
view, evenly spread over the projection, the rest counted as `over_limit`. The
default `--min-strength` is `consistency`; `--strip` defaults to `all`
(the student sees only the instruction - also for a teacher's solve,
whatever it was shown). Every conversation starts with the system turn
every solve, `ask` and probe runs under: one short system prompt of
Splinter's own, in place of sven's coding-agent prompt, so the policy is
trained under the prompt it answers under. The exception is the
`sft-step` view of a repository solve: its records open with the prompt the
solve's own profile records (the coding-agent prompt, `coding-agent-1`, for
a solve made under it), the dataset's manifest records that prompt, and a
model trained on it is asked under it. A solve whose recorded prompt this
build holds no text of is left out; runs under two different prompts do not
make one dataset until `--system-prompt` writes them all under one. Chat views are written as
brain's `generic-messages-v2`, `preference` as brain's
`generic-preference-v1` (one pair per line: the shared prompt, the
chosen and the rejected continuation) and `feedback` as brain's `generic-feedback-v1` (one
judged completion per line).
The `cpt` view (continued pretraining on the text of sources) is written as
`generic-messages-v2` too: each text is one assistant turn with nothing before
it, supervised whole, so training minimises the next-token loss over all of its
tokens. Objectives brain cannot train (contrastive, reward) need
`--export-only`.

### Policy gradient

The `policy-gradient` view is reinforcement learning from graded attempts. Where `sft-step`
imitates the actions of verified work, this view keeps the failures too and says how each
attempt did against the others at the same task:

- An attempt is a benchmark trial; its conversations (the roles of a harness) are one
  rollout. The verdict `ingest-bench` records names the trial, the task id and version, the
  model that ran it and the metrics the verifier measured; an experience whose verdict names
  no trial or task is left out as `no_rollout`.
- The reward of a pass is `1.0`. A failure earns `--fail-ceiling` (default `0.5`, always
  below one, so any pass outranks any failure) times the weighted sum of the metrics named
  by repeated `--reward-metric NAME=WEIGHT` (metrics in [0, 1], weights non-negative and
  summing to one); with no `--reward-metric` a failure earns nothing. A failed attempt that
  recorded no such metric, or one outside [0, 1], stops the build naming the trial and the
  metric; it is never read as zero. Splinter's core knows no metric: the names are yours.
- `--reward shaped-partial` (a named reward; `verdict`, the default, is the rule above) gives a
  failure partial credit for the share of its task's weighted checks it passed,
  `functional_correctness` (`f`), above the share a naive answer earns (`f_naive`):
  `C * max(0, (f - f_naive) / (1 - f_naive))`, with `C` the `--fail-ceiling` (default `0.5`).
  A pass still earns `1.0` and is still the only pass. `f_naive` is the task's `naive_floor` in
  the `practice.json` of a `--practice DIR` (repeatable), matched to the trial by the task's
  content digest; a task without one is measured from `0`. The manifest records the credit,
  every floor, the `practice.json` sha256 of each directory read (`sources`, empty when none
  was) and `groups_without_floor`; each group reports its `naive_floor` and `digest`. It takes
  no `--reward-metric`. It changes what a failure teaches, so it is an arm of its own, never a
  default.
- `--overlong-max-tokens L` adds DAPO's soft over-length penalty to every reward, from the
  completion tokens the attempt recorded (its score's usage on its verdict, else the
  `tokens.completion` of its manifest, else sven's trajectory total, else the sum over its agent
  steps; a source that states no completion count is passed over, never read as zero): nothing
  up to `L - B` tokens, `(L - B - n) / B` from there to `L`, and `-1` past `L`, with the buffer
  `B` from `--overlong-buffer` (default `L / 4`). An attempt that recorded no count stops the
  build naming it. The manifest records both thresholds as `overlong: {max_tokens, buffer}`, and
  a manifest whose buffer is zero or longer than the budget is refused when read.
- A group is the attempts at one task version made by one policy (the model recorded on the
  verdict); each trial counts once however many conversations it had. The records carry
  rewards, and brain computes each attempt's advantage from its group's rewards (brain's
  training-record contract, version 2). `--advantage` names how, and the dataset's manifest
  records it for `train` to ask brain for: `standardised` (the default, GRPO's: the reward
  less the group mean over the group's standard deviation), `mean-centred` (the reward less
  the mean), `leave-one-out` (the reward less the mean of the other attempts) or `positive`
  (standardised against the whole group, then clamped at zero: `train` sends brain
  `advantage_floor: zero`, so every attempt sets the baseline and nothing is pushed down, since
  a wrong answer shares most of its tokens with a right one and pushing it down damages those).
  Datasets built for kinds that differ in normalisation or floor are not trained together. A
  group of one attempt, or whose attempts all earned the same, teaches nothing and its
  experiences are counted as `no_group_signal`.
- `dataset build` reports what the groups can teach per token (`signal`, printed as
  `learning signal` lines). Each group's learning signal is `LS = V / mean cost`: `V` the
  variance of its rewards shrunk towards a Beta posterior (`V = (n * s2 + 2 * v0) / (n + 2)`,
  `s2` the population variance of the `n` rewards, `v0` the Beta(1, 1) posterior's expected
  `p * (1 - p)` with the rewards as fractional successes), a cost the attempt's completion
  plus uncached prompt tokens from its score's usage (else its manifest's token counts, all
  uncached). It is reported per task and per family: the lineage the staged task's `task.toml`
  names (`bench_defect_lineage`, from `--practice`), else the task itself, so seeds of one
  template count once. The summary gives the informative-group share and `sum Var(r)` per
  million completion tokens, both uniform and as the variance-per-cost draw would weigh the
  groups (`sqrt(V) / mean cost`, uninformative groups dropped). Each trial in the group report
  carries its `cost` and its `completion_tokens` (the count the over-length penalty read); an
  attempt that counted none has none, and a cost needs both its completion and its prompt count.
- An attempt of an informative group none of whose conversations the view can render (shown
  images without `--images`, a step a record cannot represent; each counted under its reason)
  still took part in its group: one record of its instruction, nothing supervised, carries its
  reward and group with `"baseline_only": true`, so brain counts it in the group's baseline and
  trains none of it. The group report marks it `baseline_only`.
- Each action of each conversation of a written attempt is one record, rendered exactly as
  `sft-step` renders it (the requests its model was sent, `--images`, `--roles`,
  `--only-actions`) with the attempt's reward, its group's key (a digest of the task, its
  version and the policy) and the attempt's name in the record's top-level `reward`, `group`
  and `rollout` fields, the keys brain's policy-gradient trainer reads. Step labels judge
  nothing here.
- An action whose agent step recorded what the policy sampled (sven's `prompt_token_ids`,
  `completion_token_ids` and `logprobs`, with `logprob_kind: policy` and a tokenizer digest) is
  written with exactly those, so nothing is re-rendered: its supervised assistant message carries
  `prompt_token_ids`, `token_ids` and `behavior_logprobs`, and the record carries
  `tokenizer_digest`. An action without a usable trace is still written, without any of these,
  and brain trains it against its start-of-run proxy; the manifest's `tokens` report (printed
  by `dataset build` as `sampled tokens`) counts
  `exact` actions and `proxy` ones by reason: `context_reconstructed` (the conversation was rebuilt
  from the trajectory, not read from the requests the model was sent, so brain's check of the
  prompt ids would refuse it), `no_trace`, `sampler_logprobs` (figures taken after
  top-k/top-p), `count_mismatch`, `invalid_ids`, `invalid_logprobs` (a log-probability that is not a finite number at most zero), `scrubbed` (the scrub changed text of the
  conversation, and ids decode to the unscrubbed text), `mixed_tokenizer`, `split_step` and `prompt_replaced` (the record's system prompt was replaced after projection: the
  ids are of the prompt the policy was sent).
  Fields are never attached in part.
- The build report and the manifest list every group (task, version, policy) with each
  trial's reward and whether it was written, including groups that yielded no record. The exam seal is
  unchanged: a held-out trial is refused, or left out with `--drop-leaking`. The same
  experiences always give the same bytes.

`train` trains such a dataset by brain's clipped policy gradient; see
[Build datasets and train a candidate](../guides/training.md).

#### Self-distillation

Where every attempt at a task failed the advantage is zero. Two ways to use those groups:
`--view policy-gradient --cliff-distill` distils the groups whose attempts all failed alike (and
keeps the reward of every group with a spread of rewards, so one build holds both kinds of
records), and `--view self-distill` distils every attempt, passed or failed, with no reward. A group is `graded` (its rewards differ), `all-fail` or `all-pass`; the report and
manifest give each group's kind and whether it was distilled.

A distilled record is the same chat record (the attempt's own conversation, nothing privileged in
it) with a top-level `teacher_context`: `{"at": <index>, "messages": [{"role": "user", "content":
...}]}`, the privileged turns inserted before message `at` (right after the first user turn) for
the teacher only. Brain reads it, renders the teacher's conversation with the model's own chat
template and refuses a record whose assistant tokens differ between the two renderings. The text
is built from the experience alone: the task's reference change (privileged kind `Reference`) and
hidden check commands, the commands the deciding verdict records as failed (phase, exit status,
failing and missing tests; the verifier keeps digests of its raw output, never the output), and the
verifier's `verification.reason` that `ingest-bench` copies into the verdict's evidence. An
attempt with none of these is left out as `no_privileged_context`. The exam seal is unchanged.

The reasoning an agent step recorded (ATIF `reasoning_content`) is the `reasoning_content`
of the assistant message it becomes, trimmed. Brain's chat template renders it as the
turn's think block, in either thinking mode; a step that recorded none carries none, and
reasoning that holds think markup leaves the trajectory out as `reasoning_in_text`. A
conversation read from the requests its model was sent keeps what it was sent: the text and
reasoning untrimmed and each call's arguments as the model wrote them, since a chat template
prints them as given and the prompt ids a record carries name that context.

A workspace trajectory is cleaned of the machine that ran it: the harness's
scratch directory is named relative to itself, the owner and group of a long
listing and the name of a home directory become `user` (but `root` and `agent`, the
account a benchmark sandbox runs its agent as, which every such machine shares). An action that repeats
an earlier call of its trajectory with the same result is context and not an
action to learn (counted as `repeated_action`), so the tail of a loop is
shown to what follows and never trained.

Passes of one task are alternative solutions only when their tool calls differ. `--distinct-calls` keeps one
of the passes whose calls (name and arguments, whitespace normalised) are the same, and
`--ignore-call-argument NAME` (repeatable) leaves an argument out of that comparison: the free text a
call carries beside what it does, such as a stated `intent`, differs between two runs that made the same calls
and is no second solution. The manifest records the names.

The sft-step view does not teach reading the repository's history. An action whose tool call
walks or shows commits (`git log`, `rev-list`, `reflog`, `show`, `blame`, `cat-file`, `fsck`, a
commit before `HEAD` such as `HEAD~1`) or reads git's storage (`.git/objects`, `.git/refs`,
`.git/logs`, `.git/HEAD`) is, by default, not supervised and stays in the context of what follows
(`repository_history`). `--repository-history truncate` ends the demonstration before it (each
action after it counted as `after_repository_history`); `keep` learns it. The working tree and its
state against `HEAD` (`git status`, `git diff`, `git diff HEAD`) stay readable. The manifest and
the report record the policy as `curation: {"policy": "no-repository-history", "mode": ...}`.
A student trained on a teacher's `git log --oneline -20` learns to run `git log --all` and
`git reflog`, which a benchmark's leak audit fails.

A trajectory is projected only when its source survives the projection whole.
Results are paired to calls by id within their step; a step whose calls share
an id is left out and counted under `duplicate_call_id` in the manifest's
`excluded` map. Ids repeat across steps (a model numbers each response's calls
from zero), so a record names its calls `s<step>-<id>`, unique in the
conversation. The projection must keep every call and result of the steps it
renders, else the trajectory is left out as `not_conserved`. Brain's chat
parser then checks every record, with the same pairing rule, before a dataset
is stored, so a dataset with a repeated call id, a call answered twice or a
result before its call cannot be stored.

### Preference and feedback

The `preference` and `feedback` views learn from graded attempts by comparison and by judgement,
for brain's `preference` objective (the pair losses DPO, IPO, SimPO and ORPO) and its `feedback`
objective (KTO).

- An attempt counts only when it is verified: decided pass or fail at `--min-strength` or above.
  A trial the infrastructure failed (the examiner's score status `environment_error`,
  `model_error` or `agent_error`; the intake gives it no verdict) is counted as
  `infrastructure_failure` and is never a rejected or undesirable sample. A pass decided below
  the minimum strength is `unverified_pass` and is never chosen or desirable; such a fail is
  `too_weak`; an attempt with no decision is `undecided`.
- A group is one task version attempted by one policy, as for the policy-gradient view. A
  verified pass and a verified fail of one group, answered by the same model role (`--roles`
  keeps only some) and decided at the same strength, are a pair. Their conversations are
  rendered as the model was shown them; where they first differ (call ids aside), the messages
  before are the shared prompt, never trained on, and each side's continuation from there to
  its last action is a candidate: its assistant turns are trained on, the tool results between
  them are context. A candidate is written as one turn, or as an array of the turns of the run
  from the point the two parted. Two attempts shown different prompts (the messages before the
  first action, or the tool list) are `different_prompt`; one that is the other's beginning is
  `no_divergence`; an attempt whose group holds no verified attempt of the other verdict is
  `no_counterpart`.
- A repair is a conversation whose first steps are copied from a failed attempt (a teacher's
  continuation of the student's own transcript) and that verified as passed. It is paired with
  a verified fail of its task whose own conversation that prefix is: the copied prefix is the
  prompt, and only the two runs from where they part are candidates. A repair whose prefix is
  not the failed attempt's is `prefix_mismatch`. A continuation that copies the student's whole
  transcript leaves the student no suffix of its own and makes no pair (`no_divergence`).
- `--pairs-per-group N` (default 4) bounds the pairs of one group or one repaired task: the
  candidate pairs are ordered by a hash of `--pair-seed` (without it, a seed derived from the
  experiences, so building again pairs the same) and the two experiences, the first that make a
  record are kept and the rest counted as `over_limit`.
- `feedback` makes one record of each verified attempt: the messages before its first own
  action are the prompt (a repair's copied prefix included), the rest to its last action is the
  completion, `desirable` when it passed.
- A candidate or completion whose tool calls read the repository's history is left out as
  `repository_history` under the default `--repository-history drop-step` (and `truncate`): a
  pair can neither drop an action nor end one side early; `keep` keeps it. The manifest names
  the policy.
- Every line carries the attempts it comes from in `metadata.rollout`: the group's key, the
  rollouts (the chosen, then the rejected one; a completion's own), the tokenizer the policy's
  server encoded them with (`tokenizer_digest`, from each step's policy trace) and, for a pair,
  its source (`attempts` or `repair`). Two sides of a pair naming different tokenizers are
  `mixed_tokenizer_digest`; a build whose records name two tokenizers is refused: build one
  dataset per policy.
- The manifest records `pairing` (`{"rule": "verified-pass-over-verified-fail/shared-prefix",
  "per_group": N, "seed": S, "attempts": n, "repairs": n, "recorded": n}`) for `preference`, and
  `feedback` (`{"desirable": n, "undesirable": n, "desirable_weight": w, "undesirable_weight":
  w}`) for `feedback`: the smaller class is weighed up by the ratio of the counts, so both weigh
  the same, and `train` sends those weights unless `--desirable-weight` or
  `--undesirable-weight` names others.

A pair recorded as a preference (a `preferred_over` relation between two experiences) is still
made from the two final answers, before the derived pairs.

## Loading a dataset

One interface reads every dataset a run uses, as the Hugging Face `datasets` library does
(`splinter_data::load`):

```rust
let dict = load_dataset::<ChatRecord>("swedishembedded/vim-headless-mastery-sft", None, &options)?;
let train = dict.get("train").unwrap();
let loader = DataLoader::new(train, BatchSampler::new(RandomSampler::new(17), 32, false));
for batch in loader.epoch(0) { /* a Vec<&ChatRecord> */ }
```

`load_dataset(reference, config, options)` gives a `DatasetDict`: the splits of the dataset by name, each a
`Dataset` of typed records with random access (`len`, `get`, `iter`, `select`, `shuffle`). The reference is

- a directory holding `SPLIT.jsonl` files at its root or under `data/` (a configuration is a subdirectory of the
  same places);
- a dataset product a factory run wrote (a directory with `dataset.json`);
- a Hub repository, `owner/name` or `hf:owner/name@revision`, resolved to a commit, downloaded into the cache
  the options name and checked against the sizes and hashes the Hub lists. Every split says the commit it was read
  at, also when a tag or branch was asked for, so a run never trains on a moving name.

`eval`, `val` and `dev` name the `validation` split. The record type is the caller's (`Record`: one JSON line in,
the same line back); `ChatRecord` is the supervised chat record of the trainers, and `dataset.import` and
`dataset.extend` read their records through it. `DataLoader`, `BatchSampler`, `RandomSampler` and
`SequentialSampler` give batches in an order that is a function of the seed and the epoch only, so two processes
that name one seed walk a dataset the same way. A seed is always named.

Brain reads token streams for its trainers (`TokenDataset`); this loader reads the records before they are
tokenized and is the one place splinter reads a dataset.

## Benchmark trials

An experience of a benchmark trial (`experiences ingest-bench`, see
[Benchmark trials as training data](../guides/bench-trials.md)) is training data only
when its task is in the benchmark's `train` split. A build refuses an experience of an
`eval` task, or of a task the register holds as `eval`; `--drop-leaking` leaves it out
and lists it instead.

## Held-out and monitoring split

`train` holds a tenth of a dataset's records out for scoring and carves its monitoring set from
the rest; the release gate draws the same held-out set from the same records. A dataset built by
`dataset build` decides both by seed and stratum: every record carries `metadata.split_draw`,
the seed of the dataset's split and its stratum - the domain its task exercises, the first part
of a benchmark task's `dimension` (`support` of `support.missing-info-defect-photo`), else the
task's kind. Within a stratum the groups (a benchmark task, a family of source text) are taken in
the order of a hash of the seed and the group, and the strata are interleaved in proportion to
their records, so held-out and monitoring sets hold every stratum at its share wherever the
records sit in the file. Without `--split-seed N` the seed is derived from the experiences the
records come from, so building again from them splits the same; another seed draws another
split. The manifest and the build report record it as `split`: `{"rule": "seeded-stratified",
"seed": N, "strata": {"<stratum>": records, ...}}`. Records written before datasets carried a
draw are held out from the newest end of the file, as they were when they were trained on.

## Manifest

Every dataset is stored under the state root by content address, beside a manifest that
names where each record came from: the experience and task it was projected from, and the
source content under them. The manifest also keeps the seeds the build drew (for example
`--mix-seed`) and the build's controls. A record of the writer's own text (the `voice` view)
names the part it prints and no experience, so a dataset shows what is the writer's and what a
model wrote. The manifest is what lets `splinter lineage` walk from a release back to the
bytes it learned from.

## Checking a stored dataset

`splinter dataset check DATASET-ID --from REF [--max-block TOKENS]` has the trainer read a stored dataset before any
device is claimed: brain's `check_dataset` over every record, read as chat records, preference pairs or judged
completions as its format is, rendered for the base `--from` names (the policy's
without it) and in the mode a supervised run reads it (reasoning when the records carry it, one example per record
when the dataset was built `--pack`). It reports the records and examples, the tokens and the supervised tokens per
example (total, min, mean, p50, p90, p99, max), the examples that supervise nothing, the examples longer than
`--max-block`, the longest records by line and every record the trainer refuses with why; for a preference or feedback dataset also the
pairing policy or class weights its manifest records and the tokenizers its records name. It exits 1 when the trainer
refuses any record or the records name more than one tokenizer. It asks a brain service started with `--training --artifact-store DIR`, with that directory given as
`--brain-artifact-store`.
