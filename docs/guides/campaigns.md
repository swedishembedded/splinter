<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Campaigns

A campaign is learning as generations of one resumable run. In each generation the champion practises on the
examiner's train-split practice tasks, a candidate is trained on everything practised so far, and one promotion gate
decides on the examiner's signed verdicts whether the candidate replaces the champion. Every stage is recorded before the
next one starts. Kill the process anywhere and run the command again: it resumes at the first stage without a record.

```bash
splinter campaign run campaign.toml
splinter campaign status campaign.toml
splinter campaign rollback campaign.toml --to 0 --reason "regressed in production"
```

`campaign run` runs generations until the campaign has run `generations` of them or a budget stops it. `--gpus N` is the
number of cards a training run holds, for counting GPU-hours. Ctrl-C cancels the stage in progress, a training job on
the brain server included. `campaign status` prints what the ledger records: every generation's candidate and decision,
the champion, its history and what was spent. Nothing runs. `campaign rollback` moves the champion back to the one
generation `--to` made (0 is the base). The move is appended to the champion's history with its `--reason`. Asked again
after it was recorded, the same rollback records nothing more. A generation whose candidate never became champion cannot
be rolled back to, and neither can the current champion.

## The campaign file

```toml
name = "support-4b"            # letters, digits, -, _ and . (the ledger directory's name)
generations = 3
base = "local:<models>/Qwen/Qwen3-4B"
# champion = "local:<models>/Qwen/Qwen3-4B+<adapter>"  # optional: generation 0's champion, an adapter on the base
seed = "12126459309185286876"  # optional; drawn, printed and recorded when left out

[rollout]
attempts = 4                   # attempts of each task
retire_at = 0.9                # a task the champion passes this often is mastered and not practised again
retire_after = 4               # ... on at least this many graded attempts
# curriculum = "capability"    # optional: draw the tasks by capability gap (below); default "all"
# tasks = 64                   # with "capability": the most tasks a generation practises
# transfer = { capabilities = 2, count = 40 }   # with "capability": fresh exercises for the weakest capabilities

[curate]
view = "sft-step"              # the dataset view (dataset build --view)
max_block = 16384              # optional: leave out longer records
pack = false

[train]
rank = 16
steps = 200                    # optional: sized to the dataset when left out
learning_rate = 0.0002         # optional
records_per_step = 4           # optional
max_block = 16384              # optional

[budget]                       # optional limits, summed over the whole campaign
rollouts = 2000
gpu_hours = 12.0
tokens = 50000000

[practice]                     # where the champion practises: `bench task practice` output, train side only
dir = "<practice-dir>"
bench = "<bench-checkout>"
jobs = 16                      # trials run at once
# harbor, sven_build and profile as for `tasks harbor practice`

[examine]                      # who examines: bench
bench = "<bench-checkout>"
public_key = "<bench-checkout>/.bench-data/private/verdict.key.pub"  # bench's public key; bench signs with its own
protocol = "direct-completion"   # or "agent" (below)
private = "<bench-private>"      # the examiner's item-level grades (or its jobs); never read by Splinter
dev = "tasks/support"                # the development split
retention = "tasks/anchor"           # the retention suite
slices = ["exams/support-s1", "exams/support-s2", "exams/support-s3"]  # one fresh sealed slice per generation
base = "Qwen/Qwen3-4B"
proposer_commit = "d81c60d99"
brain_commit = "e9044bbeb"
serving = ["--brain", "<brain-binary>", "--models-dir", "<models>", "--api-key-file", "<brain-key>", "--device", "gpu0"]
```

On the agent protocol bench runs Sven on every task instead, against the endpoint brain serves each model under
(the one `[brain]` in Splinter's configuration names, as for the rollouts). Replace `serving` with:

```toml
[examine]
protocol = "agent"
dev = "tasks/support"          # bench run collections; the slices are sealed exams in bench's private directory
retention = "tasks/anchor"
slices = [".bench-data/exams/support-s1", ".bench-data/exams/support-s2"]
# bench, public_key, private, base, proposer_commit, brain_commit as above

[examine.agent]
profile = "<bench-checkout>/profiles/<profile>.toml"  # each model's profile is derived from it: the main role points at its endpoint
min_gain = 0.05                # bench verdict --min-gain (bench's own outcome; the gate decides on the aggregates)
serving_build = "brain-e9044bbeb"  # declared by both sides; bench pairs only runs of one build
dev_set = "eval"               # optional: bench run --set for the dev collection
sven_build = "<sven-binary>"   # optional: BENCH_SVEN_BUILD
```

Generation 0's champion is the base, unless `champion` names an adapter on the base: then that model practises first,
every candidate is still trained from the base, the gate compares the first candidate with it by the adapter's `sha256:`
digest, and `rollback --to 0` returns to it. A campaign is continued from the champion it began with.

A seed above what a TOML integer holds is written as a string, decimal or `0x` hex. A campaign is continued as it began:
a changed `name`, `base`, `[rollout]`, `[curate]` or `[train]`, other practice material (by its digests), or other
examiner collections, base or serving flags are refused. Start a new campaign to change them. The seed
is read back from the ledger, and a different one is refused. `generations` and `budget` may be raised to continue. A
campaign needs one sealed slice per generation. Bench draws them beforehand from private seeds (`bench examine` over an
exam generated with `make <domain>/exam/generate SEED_FILE=...`).

## The stages of a generation

1. **rollout.** The champion attempts every practice task it has not mastered, `attempts` times, as
   `splinter tasks harbor practice` runs it. A task is mastered once its pass rate over the earlier rollouts reaches
   `retire_at` on at least `retire_after` graded attempts. Older generations' attempts count half each time the champion
   changes. The graded export is linked into the stage's directory.
   Under `curriculum = "capability"` the tasks are chosen by capability gap instead. Bench labels each practice task
   with the capability ids of its taxonomy; the earlier rollouts are counted per capability (graded passes and fails,
   each failure's diagnosed class; an attempt the infrastructure stopped is in no denominator). A capability weighs
   `(1 - r)(1/2 + f/2)(1/2 + m/2)`: `r` its smoothed pass rate, `f` the share of its measured tasks in the frontier band,
   `m` the share of its graded failures that were the model's own. A task weighs the heaviest of its capabilities, times
   1 in the frontier band or unmeasured, 1/4 if never solved, 0 if mastered; a task the infrastructure stopped on every
   attempt is never chosen. `tasks` of them are drawn without replacement in proportion to their weights, from a seed
   of the campaign's. With `transfer`, the `capabilities` heaviest measured capabilities get fresh exercises: bench draws
   `count` tasks from the collection's generator and keeps those that exercise one (`bench task practice --count --seed
   --capability`, into the stage's `transfer/`), the champion practises them beside the chosen tasks, and their trials
   join the same export. The rollout record keeps every task's weight, the tasks set aside for the infrastructure, and
   the capabilities asked for. A campaign under `all` keeps its stage keys and records as before.
2. **ingest.** The export is admitted as `splinter experiences ingest-bench` admits it. Admitting it again adds nothing.
3. **curate.** One dataset over the experience of every generation so far, through `[curate]`. Its held-out split is
   drawn by the generation's own seed. The record names the dataset by its id, the digest of its manifest. A generation
   with no training material ends here and the champion stays.
4. **train.** A candidate trained from the base on that dataset, as a served brain job. Its candidate id and brain's
   `resume_key` are both derived from the stage's identity. If a crash stops a training job, the next run asks brain again
   under the same key. Brain then continues the run from its saved state, or answers with the result it recorded, and
   trains nothing again.
5. **evaluate.** Three verdicts of the candidate against the champion: over `dev`, over `retention`, and over the
   generation's own slice. On direct completion they are `bench examine` verdicts and the dataset is passed to bench for
   its overlap audit. On the agent protocol bench runs each collection once per model (`bench run`, under a profile
   pointed at the model's endpoint, both in the session `<name>-gNNNN` on the declared serving build, the jobs in
   `<private>/jobs/<name>/gNNNN/<dev|retention|sealed>/<champion|candidate>`), and pairs each two jobs into one
   `bench verdict` that names the pair by their candidate files, takes the retention suite as its anchor and spends the
   slice. A rerun of the stage runs again from fresh job directories.
6. **gate.** The promotion gate, on the three verdicts (below).
7. **promote.** The candidate becomes the champion, or the champion stays and the gate's reasons are recorded.

A budget is checked before the rollout and the train stage. A stage that would exceed one does not start. The campaign
stops and says which limit stopped it. GPU-hours are training's wall time times `--gpus`. Tokens are the prompt and
completion tokens of every rollout trial, summed from the usage in each trial's score record. A trial whose record states
none counts nothing, and a rollout none of whose trials does records no tokens.

## The promotion gate

The gate reads only the examiner's signed aggregates. It never reads an item. It accepts two schemas:
`bench.examination/2` (direct completion) and `bench.verdict/3` (the agent protocol). Each verdict is verified in
Splinter's own process. Bench's Ed25519 signature is checked over the document's canonical bytes with bench's public key
(`public_key`); Splinter holds no signing key, so it cannot make a verdict it would accept. Then come the schema, the
examiner's own validity (a valid examination, a bench verdict that is not `Invalid` and whose integrity checks all hold,
and a candidate role, never a control arm) and integrity flags, and the candidate and predecessor the verdict names.
These must be the candidate's adapter and the champion's (by their `sha256:` digests, none for the base). The level
must be `alpha = 0.05`. A verdict that fails any check is rejected with a typed reason. On a `bench.verdict/3` the gate
reads the `retention` section for the retention rule and the `overall` section otherwise. A candidate is promoted only
when all three rules hold:

| Rule | Verdict | Holds when |
|---|---|---|
| Dev gain | `dev` | the two-sided 95% family-clustered interval of the paired gain is above 0, and the section's `families` is at least 6 (`too_few_families` otherwise, also when the count is missing; bench's interval itself is never recomputed) |
| Retention | `retention` | the lower bound of the one-sided 95% interval of the paired change is above -0.05 |
| Sealed slice | the generation's slice | it verifies, it is over sealed exams, and each of them has a burn record that names this candidate and this champion |

The sealed slice's gain is recorded and never decided on. A slice is spent by the first verdict issued over it. Bench
carries the pair that spent it in the verdict's signed burn record (`burn`, or one per exam in `burns`), so a verdict
over a slice an earlier decision spent is rejected, and so is a burn record of an exam the verdict is not over. Every
failed rule adds a typed reason to the gate's record. The reasons are `verdict` (with the role
`dev`, `retention` or `sealed`, and a rejection: `unsigned`, `bad_signature`, `schema`, `malformed`, `invalid`,
`flagged`, `candidate`, `predecessor`, `level`, `not_sealed`, `no_burn`, `burn_of_another_exam`, `spent`),
`no_dev_gain`, `dev_unmeasured`, `inferior` and `retention_unmeasured`. The record also names its `rule` (`examined`)
and `exam_verified: true`.

A verdict signed with bench's retired shared key (`hmac-sha256`) is `unsigned` here: that is a clean break. Make bench's
key pair with `bench keygen` and give the campaign its `verdict.key.pub`.

**Where no sealed slice applies.** A decision with no examiner exam for its candidate (a release from `learn`, whose exam
the learner generated itself) is decided by one explicit rule, `promotion::decide_unexamined`. It applies the dev-gain and
retention rules to the caller's own measurements and has no sealed slice. Its record says `rule: not_exam_verified` with
why, and `exam_verified: false`. It is never a silent bypass. A campaign always decides with the examiner's verdicts.

## The ledger

Everything a campaign has done is one append-only file, `<state>/campaigns/<name>/ledger.jsonl`, the stage ledger of the
campaign's directory. One process holds it at a time. Each stage writes its outputs under
`<state>/campaigns/<name>/gNNNN/<stage>/`. A line is one of:

- `{"record": "begin", "unit", "key", "attempt"}`: an attempt of a unit started;
- `{"record": "commit", "unit", "key", "attempt", "outputs", "summary"}`: the unit is done. `outputs` name files or
  directories below the campaign's directory with their digests, and `summary` is the stage's record;
- `{"record": "invalidate", "unit", "key", "reason"}`: a commit whose outputs no longer hash was set aside.

The units are `campaign` (the campaign's identity and seed), `gNNNN/rollout`, `gNNNN/ingest`, `gNNNN/curate`,
`gNNNN/train`, `gNNNN/evaluate`, `gNNNN/gate` and `gNNNN/promote` for each generation, and `rollback/N` for each
rollback. A unit's key is the digest of everything its result depends on: its settings, its draw of the campaign's seed,
the champion, and the outputs and records of the stages before it. It never depends on a time or a path. A commit under a
key is reused and its work is not run again, so a stage delivered twice is a no-op. A begin without a commit is an
interrupted attempt, which the next run takes over. A torn last line is cut off. A run killed anywhere and resumed records
the same commits (unit, key, outputs and summary) as a run that was never killed. Only the attempt numbers count the
interruptions.

The records (`summary`):

| Unit | Holds |
|---|---|
| `campaign` | `identity` (the specification's identity, the rollouts' and the examiner's), `seed` |
| `rollout` | `champion`, `tasks`, `trials` (task, reward, passed), `rollouts`, `tokens` (prompt and completion, from the trials' usage) |
| `ingest` | `train_set`, `experiences`, `refused` |
| `curate` | `dataset` (the manifest digest), `records`, `sets` |
| `train` | `candidate` (generation, model, candidate id, adapter digest), `resume_key`, `steps`, `gpu_hours` |
| `evaluate` | `slice`; outputs `dev`, `retention`, `sealed` (the three verdicts) |
| `gate` | `promoted`, `rule`, `exam_verified`, `reasons`, `pair`, and the aggregates of each verdict that verified |
| `promote`, `rollback/N` | `kind` (`promote`, `keep`, `rollback`), `generation`, `from`, `to`, `reason` |

The champion is generation 0's (the base, or `champion`) until the first promotion, and then what the newest `promote` or `rollback` record moved it to.
