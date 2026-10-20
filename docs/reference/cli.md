<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Command reference

Generated from the command-line grammar; do not edit it. `splinter <command> --help`
prints the same text for one command. Worked examples are in the guides; this page
lists every command, argument and option.

## Global options

Every command takes these.

- `--device <DEVICE>`: Which compute may be scheduled: `gpu`, `gpu1`, `gpu0,gpu1`, `cpu0-7`, `vulkan`, ... in brain's grammar. Without it, every device the machine has.
- `--backend <BACKEND>`: How the hardware is driven: wgpu, vulkan, cuda or cpu. Without it brain picks, which on a CUDA host is not CUDA.
- `--vram-budget <VRAM_BUDGET>`: The VRAM, in GiB, this process was granted: every card it budgets from is capped at it, so a run on a share of a card sizes its model, KV pool and context to the share. Without it the whole card is budgeted.
- `--state <DIR>`: The state root (default: $SPLINTER_STATE, else ~/.splinter)
- `--json`: Print machine-readable JSON on stdout instead of text.
- `-v, --verbose`: Report more as work goes (each stage's full summary)
- `--allow-remote`: Permit models reached over the network (remote:...); without it, and without SPLINTER_ALLOW_REMOTE=1, every model runs locally.
- `--policy-context-tokens <TOKENS>`: Limit the policy model's context to this many tokens (default: the largest its checkpoint supports); for a policy too large for the card at that size. Another model is limited by its reference, `local:<checkpoint>@<tokens>`.
- `--remote-concurrency <N>`: How many requests to a model reached over an API may be in flight at once [default: 4]
- `--solve-deadline <SECONDS>`: How many seconds one task's solve may take on the wall clock [default: 300]
- `--test-output-cap <BYTES>`: Bytes kept of each output stream of a test run [default: 1048576]
- `--allow-unconfined-workspace`: Let a project's commands (the solver's shell, the test runs) see the whole filesystem where the kernel cannot confine them to their tree. Without it such a run is refused; the environment record says when it was allowed.
- `--allow-workspace-network`: Let a project's commands (the solver's shell, the test runs) use the network. Without it they cannot connect anywhere, a model server on the loopback included; the environment record says when it was allowed.
- `--workspace-max-processes <N>`: Bound the processes and threads one project command may start. Part of the environment record, so tasks generated without it name a different environment.
- `--runtime-dir <DIR>...`: A directory searched before the system's for the interpreters code runs in (`lua` is Lua 5.1 and no other: `make deps/lua` builds the pinned one); repeat for more.
- `--temperature <T>`: The sampling temperature, 0 to 2, asked of a model reached through sven's providers (the provider's own default when not given). The same flag sets the pass@k sampling of a verb that measures it; when given twice, the last wins.
- `--top-p <P>`: Nucleus sampling (`top_p`, above 0 up to 1) asked of a model reached through sven's providers; the server's own default when not given.
- `--top-k <K>`: Sampling among the K most likely tokens (`top_k`) asked of a model reached through sven's providers; the server's own default when not given.
- `--sampling-seed <SEED>`: The seed every request of a model reached through sven's providers carries, so that a sampled attempt is reproducible; recorded with the attempt. The server draws its own when not given.
- `--thinking`: Let a model served on this machine reason before it answers (the request says so); without it the server is asked for no reasoning block. Recorded with the attempt.
- `--thinking-budget <N>`: With --thinking: the most tokens the model may think in one response; the server closes the thought there and the model goes on to its answer. Recorded with the attempt. Unbounded when not given.

## Commands

- [splinter learn](#splinter-learn) - Learn from sources
- [splinter practice](#splinter-practice) - Learn a repository from its own tests
- [splinter ask](#splinter-ask) - Answer a question with a model, closed-book unless --open-book
- [splinter status](#splinter-status) - Show the policy in use, recent runs and what the stores hold
- [splinter doctor](#splinter-doctor) - Check that a run can start here
- [splinter baseline](#splinter-baseline) - Print the `baseline-v1` record of what a registered run runs on
- [splinter source](#splinter-source) - Capture and inspect sources
  - [splinter source add](#splinter-source-add) - Capture a file (a document), a directory (a repository) or cmd:COMMAND
  - [splinter source list](#splinter-source-list) - List the stored sources
  - [splinter source show](#splinter-source-show) - Show one stored source
- [splinter experiment](#splinter-experiment) - Check an experiment plan
  - [splinter experiment check](#splinter-experiment-check) - Validate a plan against the catalog of families and show its schedule
- [splinter absorb](#splinter-absorb) - Learn from the sessions you held with an agent
- [splinter consolidate](#splinter-consolidate) - Take the generation the policy is in into the weights
- [splinter improve](#splinter-improve) - Choose how to improve an agent on a target capability from what each way was measured to do, net of what it costs
- [splinter generation](#splinter-generation) - Generations of a coding policy, each held to what the ones before it had
  - [splinter generation run](#splinter-generation-run) - Run one generation of a coding policy
  - [splinter generation matrix](#splinter-generation-matrix) - The ledger of an experiment and the full retention matrix of the lineage `policy:<alias>` points at
- [splinter session](#splinter-session) - Take in recorded agent sessions, secrets removed
  - [splinter session add](#splinter-session-add) - Take recorded agent sessions in as sources
  - [splinter session list](#splinter-session-list) - List the stored sessions
- [splinter claims](#splinter-claims) - What a person taught in sessions
  - [splinter claims extract](#splinter-claims-extract) - Have a model propose the claims each stored session teaches
  - [splinter claims gate](#splinter-claims-gate) - Rule on a claim set
  - [splinter claims forget](#splinter-claims-forget) - Forget a live claim
  - [splinter claims list](#splinter-claims-list) - List the stored claim sets
  - [splinter claims show](#splinter-claims-show) - Show a claim set
  - [splinter claims ledger](#splinter-claims-ledger) - Show the ledger
- [splinter tasks](#splinter-tasks) - Generate and inspect task sets
  - [splinter tasks generate](#splinter-tasks-generate) - Generate tasks from stored sources into a task set
  - [splinter tasks variants](#splinter-tasks-variants) - Write differently worded questions about each task of a task set, into a task set of their own
  - [splinter tasks canaries](#splinter-tasks-canaries) - Make canaries of the workspace tasks of a task set
  - [splinter tasks pick](#splinter-tasks-pick) - Divide the union of task sets into a picked set and the rest, by a stable hash of the seed and the task's family (the practice exam's ranking)
  - [splinter tasks harbor](#splinter-tasks-harbor) - Harbor tasks
    - [splinter tasks harbor discover](#splinter-tasks-harbor-discover) - List the Harbor tasks under a directory and whether each can be imported, at the revision of the checkout it is in
    - [splinter tasks harbor import](#splinter-tasks-harbor-import) - Import Harbor tasks (each a task directory) as workspace tasks in one task set
    - [splinter tasks harbor export](#splinter-tasks-harbor-export) - Write the workspace tasks of a task set as Harbor tasks, one directory each under --out, with everything needed to import them back exactly
  - [splinter tasks list](#splinter-tasks-list) - List the stored task sets
  - [splinter tasks show](#splinter-tasks-show) - Show a task set's tasks, or one task
- [splinter solve](#splinter-solve) - Solve every task of a task set, each in the environment it records
- [splinter harness](#splinter-harness) - Improve a sven harness
  - [splinter harness optimise](#splinter-harness-optimise) - Search a space of variants of a baseline harness for one that verifiably beats it
- [splinter verify](#splinter-verify) - Append verdicts to an experience set (re-grades without re-running)
- [splinter critique](#splinter-critique) - Critique the failed experiences of a set and retry them with it
- [splinter judge](#splinter-judge) - Measure a judge against labelled experiences
  - [splinter judge calibrate](#splinter-judge-calibrate) - Measure a judge on labelled experiences and keep the calibration `verify --judge` gates its verdicts by
  - [splinter judge measure](#splinter-judge-measure) - Measure a judge on the controls a task set's own references give
- [splinter experiences](#splinter-experiences) - Inspect experience sets and experiences
  - [splinter experiences list](#splinter-experiences-list) - List the stored experience sets
  - [splinter experiences show](#splinter-experiences-show) - Show an experience set, or one experience with its annotations
  - [splinter experiences replay](#splinter-experiences-replay) - Run an experience's code calls again in the environment it records (or every experience of a set) and compare each result with the one it observed
  - [splinter experiences ingest-harbor](#splinter-experiences-ingest-harbor) - Read the trials Harbor left for imported tasks (each a trial directory with result.json and agent/trajectory.json in ATIF) as experiences carrying Harbor's own verdict
  - [splinter experiences compare](#splinter-experiences-compare) - Compare two experience sets of the same tasks
- [splinter dataset](#splinter-dataset) - Build datasets from experience sets, and export them
  - [splinter dataset build](#splinter-dataset-build) - Project experience sets through a view into a stored dataset
  - [splinter dataset export](#splinter-dataset-export) - Copy a dataset and its manifest into a directory
- [splinter train](#splinter-train) - Train a candidate adapter on datasets
- [splinter rehearse](#splinter-rehearse) - Build a rehearsal dataset
- [splinter release](#splinter-release) - Run the release gate on a candidate and release it if every check passes
  - [splinter release list](#splinter-release-list) - List the releases and the aliases pointing at them
- [splinter rollback](#splinter-rollback) - Point an alias back at the release its current one was trained from
- [splinter eval](#splinter-eval) - Grade a candidate or a model closed-book on a suite
  - [splinter eval matrix](#splinter-eval-matrix) - The continual-learning results matrix of a lineage
- [splinter exam](#splinter-exam) - Put a candidate and the policy it continues to its held-out tasks and have a calibrated judge compare them
- [splinter select](#splinter-select) - Choose the step of a run kept with its evaluations by what each writes on the dev suite
- [splinter exam-report](#splinter-exam-report) - Analyses done after an exam from what its report kept
  - [splinter exam-report labels-export](#splinter-exam-report-labels-export) - Write a blind, stratified, shuffled sample of the exam's answers for a person to label right or wrong, and the key that says which arm gave each and what the judge said
  - [splinter exam-report labels-import](#splinter-exam-report-labels-import) - What a person's labels say of the judge
  - [splinter exam-report memorisation](#splinter-exam-report-memorisation) - How much of each arm's answers is the text the candidate was trained to produce again
- [splinter exam-set](#splinter-exam-set) - Make and show frozen exams
  - [splinter exam-set create](#splinter-exam-set-create) - Capture the sources, reserve families of them, write the exam from that text alone and freeze it
  - [splinter exam-set power](#splinter-exam-set-power) - Simulate the planned paired test to size an exam before it is paid for
  - [splinter exam-set show](#splinter-exam-set-show) - Show a frozen exam
- [splinter runs](#splinter-runs) - List, inspect and cancel runs
  - [splinter runs list](#splinter-runs-list) - List the recorded runs
  - [splinter runs show](#splinter-runs-show) - Show one run's record
  - [splinter runs cancel](#splinter-runs-cancel) - Ask a run in progress to stop
- [splinter state](#splinter-state) - Show, maintain, verify, repair, archive and restore the state
  - [splinter state status](#splinter-state-status) - What the experience database holds, as files
  - [splinter state maintain](#splinter-state-maintain) - Merge small files, index what is not indexed and retire finished writers
  - [splinter state unpin](#splinter-state-unpin) - Let go of a snapshot a dataset (or another holder) keeps alive, named as `state status` lists it, so its files can be collected
  - [splinter state verify](#splinter-state-verify) - Check the database and every file it tracks, and report each one that is missing or damaged
  - [splinter state repair](#splinter-state-repair) - Recover what verify finds
  - [splinter state archive](#splinter-state-archive) - Pack the database and the files it tracks into one archive
  - [splinter state restore](#splinter-state-restore) - Unpack an archive (and the archives an incremental one was made after) into an empty state root, verifying everything first
- [splinter lineage](#splinter-lineage) - Trace any artifact up to where it came from (down to source bytes) and down to everything that came from it

## splinter

A learning agent with its own model. Tell it what to learn - a document, a repository, a command's output - and it generates tasks from it, solves and verifies them, trains a candidate on what passed, and releases it only if it measurably improves without forgetting. With no command, a REPL on the current policy: each line is handled exactly like `splinter "<line>"`.

```text
splinter [OPTIONS] [SENTENCE] [COMMAND]
```

Arguments:

- `<SENTENCE>`: A sentence, classified by the policy into one of the commands below; an unclear or unconfident reading is asked back, never run.

Subcommands:

- [`learn`](#splinter-learn) - Learn from sources
- [`practice`](#splinter-practice) - Learn a repository from its own tests
- [`ask`](#splinter-ask) - Answer a question with a model, closed-book unless --open-book
- [`status`](#splinter-status) - Show the policy in use, recent runs and what the stores hold
- [`doctor`](#splinter-doctor) - Check that a run can start here
- [`baseline`](#splinter-baseline) - Print the `baseline-v1` record of what a registered run runs on
- [`source`](#splinter-source) - Capture and inspect sources
- [`experiment`](#splinter-experiment) - Check an experiment plan
- [`absorb`](#splinter-absorb) - Learn from the sessions you held with an agent
- [`consolidate`](#splinter-consolidate) - Take the generation the policy is in into the weights
- [`improve`](#splinter-improve) - Choose how to improve an agent on a target capability from what each way was measured to do, net of what it costs
- [`generation`](#splinter-generation) - Generations of a coding policy, each held to what the ones before it had
- [`session`](#splinter-session) - Take in recorded agent sessions, secrets removed
- [`claims`](#splinter-claims) - What a person taught in sessions
- [`tasks`](#splinter-tasks) - Generate and inspect task sets
- [`solve`](#splinter-solve) - Solve every task of a task set, each in the environment it records
- [`harness`](#splinter-harness) - Improve a sven harness
- [`verify`](#splinter-verify) - Append verdicts to an experience set (re-grades without re-running)
- [`critique`](#splinter-critique) - Critique the failed experiences of a set and retry them with it
- [`judge`](#splinter-judge) - Measure a judge against labelled experiences
- [`experiences`](#splinter-experiences) - Inspect experience sets and experiences
- [`dataset`](#splinter-dataset) - Build datasets from experience sets, and export them
- [`train`](#splinter-train) - Train a candidate adapter on datasets
- [`rehearse`](#splinter-rehearse) - Build a rehearsal dataset
- [`release`](#splinter-release) - Run the release gate on a candidate and release it if every check passes
- [`rollback`](#splinter-rollback) - Point an alias back at the release its current one was trained from
- [`eval`](#splinter-eval) - Grade a candidate or a model closed-book on a suite
- [`exam`](#splinter-exam) - Put a candidate and the policy it continues to its held-out tasks and have a calibrated judge compare them
- [`select`](#splinter-select) - Choose the step of a run kept with its evaluations by what each writes on the dev suite
- [`exam-report`](#splinter-exam-report) - Analyses done after an exam from what its report kept
- [`exam-set`](#splinter-exam-set) - Make and show frozen exams
- [`runs`](#splinter-runs) - List, inspect and cancel runs
- [`state`](#splinter-state) - Show, maintain, verify, repair, archive and restore the state
- [`lineage`](#splinter-lineage) - Trace any artifact up to where it came from (down to source bytes) and down to everything that came from it

### splinter learn

Learn from sources: capture them, generate tasks, solve, verify, critique and retry failures, build an sft-final dataset, train a candidate on it from the champion, and release it if it passes the release gate

```text
splinter learn [OPTIONS] [SOURCE]...
```

Arguments:

- `<SOURCE>...`: Files, directories, cmd:\<command\> (quoted as one argument), or stored source ids; with --source, the words of the objective instead.

Options:

- `--source <URL|PATH>...`: Learn an objective from documentation: `learn "Become an exceptional Lua 5.1 programmer" --source https://www.lua.org/manual/5.1/manual.html`. The positional words are the objective, and each --source (a URL, a file or a directory; repeat for more) is read into a skill graph with probes, a measured baseline, training rounds on the skill expected to benefit most and an assessment on probes that decided nothing.
- `--mastery <SHARE>`: The pass rate on a skill's probes that counts as mastered [default: 0.7]
- `--probes <N>`: Probes per skill in each half (dev and final)
- `--min-probes <N>`: The graded probes a skill's competence must rest on [default: 3]
- `--max-rounds <N>`: Training rounds in all [default: 8]
- `--max-rounds-per-skill <N>`: Training rounds one skill may take [default: 2]
- `--retention-drop <SHARE>`: How far a skill measured earlier may fall in a round before its candidate is rejected [default: 0.25]
- `--goal <TEXT>`: What the learner is after; steers the tasks generated.
- `--kinds <K,..>...`: Task kinds, comma-separated; when not named, a planner model surveys the sources and chooses them.
- `--budget <DUR>`: Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
- `--dry-run`: Report the plan and write nothing.
- `--no-release`: Stop at the trained candidate: do not run the release gate.
- `--no-frontier`: Solve each task once and train on every task, instead of measuring pass@k and keeping the tasks worth training on (failed at least sometimes, with a verified answer)
- `--distill`: Skip the policy's own attempts: the teacher answers every task open-book and the policy is trained on its verified answers. For a policy that cannot answer the tasks closed-book at all.
- `--teacher <REF>`: The model that solves open-book, shown each task's grounding material, the tasks the policy never solves (default: the policy)
- `--generator <REF>`: The model that writes the tasks (default: the policy)
- `--judge <REF>`: The model that judges answers to tasks only a judge can pass (a conversation: a check that nothing was invented does not make an answer right). It is another model than the teacher, the generator and the policy, and is measured before its verdicts count.
- `--planner <REF>`: The model that surveys the sources and plans the learning when no --kinds are named (default: the generator)
- `--with-passages <SHARE>`: Give this share of the training records retrieved passages of the sources in their prompt, four to a record and the passage the task was written from among them for four in five, so the policy learns to use context where it holds the answer and to answer without it where it does not. Needs the embedding model.
- `--abstain <SHARE>`: Make this share of all the dialogue records abstentions: the writer, shown passages that do not hold the answer or asked what lies beyond the writings, says so in their own voice (two thirds retrieval misses, a third questions after their time). Taken from the records given passages, so it is at most --with-passages.
- `--base <CHECKPOINT>`: The checkpoint to train: a name in the model store or a path. Used with --adapter; the configured base when not given.
- `--adapter <NAME>`: The name to keep the trained adapter under in the model store, where `local:<base>+<name>` finds it. A name that exists is continued, not replaced. The run publishes the adapter instead of putting it through the release gate.
- `--persona <NAME>`: Who the policy is to become, when it is to think like a person: its training records open with a system prompt saying so, and it is asked under that prompt afterwards (default: the persona the plan finds in the goal, if any)
- `--voice <SHARE>`: The share of the training tokens that is the writer's own text, in [0, 1): records built by code from the sources, no model involved, trained beside the dialogues and held out with the families they print; 0 turns it off [default for a run with a persona: 0.7]
- `--describe-voice`: Ask the writer's passages by what the generator says they are about instead of by their heading and opening.
- `--select-on-dev`: Keep the adapter of every evaluation of the training and choose the step by what each writes on the reserved dev suite - right answers, no invented specifics, then voice - instead of by the monitoring loss.
- `--rehearsal <SHARE>`: The share of the training draws that are the base model's own answers to general tasks, in [0, 1): built by the rehearse stage, never an anchor task, mixed into training and the monitoring set; 0 turns it off [default for a run with a persona: 0.25]
- `--rehearsal-seed <N>`: The seed the rehearsal's form tasks are built with [default: drawn and printed]
- `--denoise-seed <N>`: The seed the denoise tasks are built with, when the kinds include them [default: drawn and printed]
- `--exam-families <N>`: How many families of the sources to reserve for the exam before anything is generated or built: their text, in every edition, is in no source the run reads after that; the run is refused when the sources cannot spare them; 0 reserves none [default for a run with a persona: 50]
- `--exam-tasks-per-family <N>`: Tasks of each family of the reserved final test [default: 8]
- `--dev-families <N>`: How many more families to reserve for the dev suite a checkpoint is chosen on, as far from training as the final test's and apart from it; 0 reserves none [default with a final test: 50]
- `--dev-tasks-per-family <N>`: Tasks of each family of the dev suite [default: 4]
- `--exam-resamples <N>`: Answers per task per arm in the reserved exam: the greedy one and the rest sampled [default: 3]
- `--exam-seed <N>`: The seed the exam's families are reserved with: the same seed over the same sources reserves the same families, and exams are comparable across runs only when they share one [default: drawn and printed]
- `--interval-seed <N>`: The seed the exam's bootstrap intervals are resampled with [default: drawn and printed]
- `--alpha-spending`: Spend the significance level across the releases of a lineage by a fixed schedule, so that a long run of candidates no better than their champions releases one with probability below the level.
- `--target-effect <SHARE>`: The lead, as a share of the exam's families won, the exam must be able to see: a miss on an exam too small to see it is reported inconclusive, not as a failure.
- `--anchor-max-net-lost <COUNT>`: Decide the anchor check on the answers the candidate loses net of the answers it gains, refusing above this count (6 for a 200-question suite), instead of on the drop in accuracy.
- `--steps <N>`: The step budget of the training: the most steps it may take [default: 3 passes over the examples, within bounds]
- `--rank <N>`: LoRA rank of the adapter [default: 8]
- `--lr <LR>`: The peak learning rate [default: 0.0002]
- `--alpha <A>`: The LoRA alpha, the update scaled by alpha / rank [default: 2 x the rank]
- `--weight-decay <WD>`: The AdamW weight decay on the adapter's matrices [default: 0]
- `--records-per-step <N>`: Records averaged into one optimizer step (default: from the size of the dataset, when the steps are not named)
- `--seed <N>`: The seed of the adapter's initialisation and the batch order (drawn, printed and recorded if not given); two runs that differ only here show how much of a measured difference is the draw.
- `--eval-every <N>`: Steps between evaluations of the monitoring records; 0 monitors nothing and the candidate carries its last step [default: 16 evaluations over the step budget]
- `--patience <N>`: Evaluations without improvement before the training stops; 0 runs the whole budget (the best evaluation is carried either way) [default: 4]
- `--monitor-share <SHARE>`: The share of the training families set aside as the monitoring records, in (0, 0.5]: whole families, never the held-out ones the gate and the exam decide on [default: 0.1]
- `--keep-evaluations`: Keep the adapter of every evaluation with the candidate, so a step can be chosen on the dev suite (`select`) instead of the monitoring loss.
- `--bf16-base`: Hold the frozen base at bf16, half the bytes of fp32: what a 7B base needs to train on one 24 GiB card.
- `--k <N>`: Attempts per task when measuring pass@k. Default: `4`.
- `--temperature <T>`: The policy's sampling temperature for its pass@k attempts [default for a local model: 0.8; a remote model samples as its server does unless this is given]
- `--top-k <N>`: The policy's top-k cut for its pass@k attempts [default for a local model: 20]

### splinter practice

Learn a repository from its own tests: take a revision of it, make executable tasks of faults its tests catch, have the policy solve them as an agent in copies of the tree, keep what hidden checks confirm (reserving some files as an exam first, and refusing any attempt that saw the hidden tests), train a candidate from the champion and release it if it passes the release gate

```text
splinter practice [OPTIONS] <REPO>
```

Arguments:

- `<REPO>` (required): The git repository to learn from: a work tree or a bare repository.

Options:

- `--rev <REV>`: The revision of it to learn from: anything git resolves to a commit. The tracked files of that commit are the snapshot every task is pinned to; uncommitted work is not in it. Default: `HEAD`.
- `--allow-dirty`: Snapshot the committed files of a revision that is checked out with uncommitted changes, instead of refusing it.
- `--file <PATH>...`: A file, relative to the repository, that tasks may change; repeat for more. Without any, every source file of the project the rules recognise (a Python project with a tests directory, a Cargo project)
- `--max-tasks <N>`: The most tasks one run admits.
- `--mutants-per-file <N>`: The most faults seeded in one file.
- `--exam-families <N>`: How many families (a family is the file a fault was seeded in) to reserve for the exam before any attempt is made at any task: their tasks are set apart and never solved or trained on, in this run or a later one; 0 reserves none [default: a quarter of the families]
- `--canaries <N>`: The most canaries (exam tasks no honest change can pass) the release gate hands the candidate; a pass on one refuses it; 0 hands none [default: 20]
- `--exam-seed <N>`: The seed the exam families are chosen by when this run reserves the first: drawn and printed [default], then kept with the families, so every later run reads it back and naming another is refused.
- `--k <N>`: Attempts per task when measuring pass@k. Default: `4`.
- `--temperature <T>`: The policy's sampling temperature for its pass@k attempts [default for a local model: 0.8; a remote model samples as its server does unless this is given]
- `--top-k <N>`: The policy's top-k cut for its pass@k attempts [default for a local model: 20]
- `--follow-ups <N>`: Follow-up turns an agent is sent after the checks it was shown stay red [default: 2]
- `--max-tool-calls <N>`: Tool calls an agent may make in one attempt, follow-ups included [default: 60]
- `--max-repeated-results <N>`: How often one tool call may return the same answer before the attempt is stopped as going nowhere [default: 3]
- `--budget <DUR>`: Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
- `--no-release`: Stop at the trained candidate: do not run the release gate.
- `--teacher <reference|REF>`: Who demonstrates the tasks the policy never solves, so they are not dropped: `reference` replays each task's own reference change as an agent's actions, or a model that solves them as an agent (`remote:`/`local:`). Either is graded by the same independent verifier as the policy's attempts; it never grades itself.
- `--replay <POLICY>`: Which of the earlier releases' records are replayed beside the new ones (default uniform): none, uniform over every record, or balanced over capabilities. Values: `none`, `uniform`, `balanced`, `failure-targeted`, `learning-progress`.
- `--rebuild`: Train a fresh adapter on the frozen base from the records of every earlier release and the new ones, instead of continuing the champion.
- `--curriculum <STRATEGY>`: The order tasks are solved and taught in (`--budget` may end the run before all are reached): the order they were admitted in, or by learning progress (the change in a family's verified pass rate between releases, per GPU second it cost) Default: `frontier`. Values: `frontier`, `learning-progress`, `gain-ranked`.
- `--curriculum-seed <N>`: The seed of the learning-progress draw (drawn, printed and recorded if not given)
- `--lp-window <N>`: Releases in each of the two windows learning progress compares. Default: `2`.
- `--lp-novelty <X>`: The novelty bonus of a family nothing has attempted, in pass-rate units; it shrinks with attempts. Default: `0.1`.
- `--lp-explore <X>`: The share of the draw that is uniform over families, in [0, 1]: no family gets less than this share divided by the number of families. Default: `0.1`.
- `--alpha-spending`: Spend the significance level across the releases of a lineage by a fixed schedule, so that a long run of candidates no better than their champions releases one with probability below the level.
- `--target-effect <SHARE>`: The lead, as a share of the exam's families won, the exam must be able to see: a miss on an exam too small to see it is reported inconclusive, not as a failure.
- `--anchor-max-net-lost <COUNT>`: Decide the anchor check on the answers the candidate loses net of the answers it gains, refusing above this count (6 for a 200-question suite), instead of on the drop in accuracy.
- `--steps <N>`: The step budget of the training: the most steps it may take.
- `--rank <N>`: LoRA rank of the adapter [default: 8]
- `--max-block <TOKENS>`: The longest training record, in tokens, the card fits: a longer record is left out of the dataset and counted, instead of stopping the training.
- `--lr <LR>`: The peak learning rate [default: 0.0002]
- `--alpha <A>`: The LoRA alpha, the update scaled by alpha / rank [default: 2 x the rank]
- `--weight-decay <WD>`: The AdamW weight decay on the adapter's matrices [default: 0]
- `--records-per-step <N>`: Records averaged into one optimizer step.
- `--seed <N>`: The seed of the adapter's initialisation and the batch order (drawn, printed and recorded if not given)
- `--eval-every <N>`: Steps between evaluations of the monitoring records; 0 monitors nothing and the candidate carries its last step [default: 16 evaluations over the step budget]
- `--patience <N>`: Evaluations without improvement before the training stops; 0 runs the whole budget (the best evaluation is carried either way) [default: 4]
- `--monitor-share <SHARE>`: The share of the training families set aside as the monitoring records, in (0, 0.5]: whole families, never the held-out ones the gate and the exam decide on [default: 0.1]
- `--keep-evaluations`: Keep the adapter of every evaluation with the candidate, so a step can be chosen on the dev suite (`select`) instead of the monitoring loss.
- `--bf16-base`: Hold the frozen base at bf16, half the bytes of fp32.
- `--plan <FILE>`: An experiment plan (`experiment check`) this run is one generation of: the repository must be a family the plan trains in `--generation`, and a sealed family is always refused. The catalog is the catalog.toml beside the plan.
- `--generation <N>`: The generation of the plan, counted from 1.
- `--test-command <PROGRAM ARG...>...`: The command that runs the project's whole test suite in the root of its tree, as a program and its arguments (everything after this flag, so it comes last). Without it the rules give one: `python3 -m unittest` for a Python project, `cargo test --offline` for a Cargo project.

### splinter ask

Answer a question with a model, closed-book unless --open-book

```text
splinter ask [OPTIONS] <QUESTION>
```

Arguments:

- `<QUESTION>` (required): The question.

Options:

- `--open-book <SOURCE-ID>`: Show the model this stored source's text with the question.
- `--retrieve <SOURCE-ID>...`: Show the model the passages of these stored sources that bear on the question, found by meaning and by exact words, instead of all of one.
- `--passages <N>`: How many passages `--retrieve` shows. Default: `6`.
- `--reranker <REF>`: A model that reads the candidates search found beside the question and moves the ones that bear on it forward.
- `--policy <REF>`: The model asked. Default: `policy:default`.

### splinter status

Show the policy in use, recent runs and what the stores hold

```text
splinter status [OPTIONS]
```

### splinter doctor

Check that a run can start here: the state root is writable with room to spare, the policy's weights fit on a GPU, and every role's model resolves to weights on disk. Loads no model and uses no network. Exits 1 when a check fails

```text
splinter doctor [OPTIONS]
```

### splinter baseline

Print the `baseline-v1` record of what a registered run runs on: the three commits, toolchain, binaries, model and tokenizer digests, seeds and hardware

```text
splinter baseline [OPTIONS]
```

Options:

- `--brain-bin <FILE>`: The brain binary the models are served with; its digest and version are recorded.
- `--model <DIR>...`: A model directory (or a GGUF file's directory) to record by file digest, with its tokenizer; repeatable. Hashing a large model takes minutes.
- `--exam-seed <N>`: The registered exam seed, recorded as given.
- `--training-seed <N>`: The registered training seed, recorded as given.
- `--require-clean`: Exit 1 and say why when the tree is dirty, a commit is unknown or the binary could not be digested.

### splinter source

Capture and inspect sources

```text
splinter source [OPTIONS] <COMMAND>
```

Subcommands:

- [`add`](#splinter-source-add) - Capture a file (a document), a directory (a repository) or cmd:COMMAND
- [`list`](#splinter-source-list) - List the stored sources
- [`show`](#splinter-source-show) - Show one stored source

#### splinter source add

Capture a file (a document), a directory (a repository) or cmd:COMMAND... (a command run), and store it. URLs are refused

```text
splinter source add [OPTIONS] <PATH|cmd:COMMAND...>...
```

Arguments:

- `<PATH|cmd:COMMAND...>...` (required): PATH, or cmd: followed by the command and its arguments.

Options:

- `--usage-policy <POLICY>`: The usage policy the content is under: redistributable, research_only, noncommercial, restricted_DUA or unknown. Without it the terms are unstated, which no release can be distributed under. The policy is part of the source's identity.

#### splinter source list

List the stored sources

```text
splinter source list [OPTIONS]
```

#### splinter source show

Show one stored source

```text
splinter source show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The source's id, or a unique prefix of it.

### splinter experiment

Check an experiment plan: which families each generation trains on, which earlier ones it is held to, which are the sealed final exam

```text
splinter experiment [OPTIONS] <COMMAND>
```

Subcommands:

- [`check`](#splinter-experiment-check) - Validate a plan against the catalog of families and show its schedule

#### splinter experiment check

Validate a plan against the catalog of families and show its schedule: families disjoint by domain, no sealed family scheduled for training or retention, retention only on families an earlier generation trained. A plan that is not valid is refused with every violation

```text
splinter experiment check [OPTIONS] <PLAN>
```

Arguments:

- `<PLAN>` (required): The plan file.

Options:

- `--catalog <FILE>`: The catalog of families (default: catalog.toml beside the plan)

### splinter absorb

Learn from the sessions you held with an agent: take them in, extract what you taught, teach it from many records, train again from the base and release only if every claim the champion answered still is and a new one is

```text
splinter absorb [OPTIONS] <SESSIONS>...
```

Arguments:

- `<SESSIONS>...` (required): ATIF files, or directories searched for *.atif.json.

Options:

- `--policy <REF>`: The policy to update: a release is made under its alias. Default: `policy:default`.
- `--generator <REF>`: The model that proposes claims and writes their paraphrases and forms (default: the policy)
- `--teacher <REF>`: The model that answers with the claim in front of it, as the policy would (default: the policy)
- `--judge <REF>`: A model that decides the answers the claim's terms cannot, whether a later claim supersedes, reinforces or is apart from an earlier one, and whether the person's words assert a claim's statement; another model than the teacher, the generator and the policy.
- `--entail`: Whether the judge also refuses a claim whose cited words do not assert its statement (`--entail=false` turns it off); it applies when a judge is configured. Values: `true`, `false`.
- `--passes <N>`: How many times each session is read for claims, in different orders; only a claim every pass proposed is ruled on. Default: `2`.
- `--paraphrases <N>`: Differently worded questions written per claim; a third of them are kept out of training as the claim's own stopping and gate set. Default: `12`.
- `--rehearsal-share <SHARE>`: The share of the training draws that are the base's own answers to general tasks, in [0, 1); 0 turns it off. Default: `0.25`.
- `--rehearsal-seed <N>`: The seed the rehearsal's form tasks are built with [default: drawn and printed]
- `--epochs <N>`: The most passes over the night's records. Default: `6`.
- `--steps <N>`: The step budget, in place of --epochs.
- `--rank <N>`: LoRA rank of the adapter. Default: `8`.
- `--lr <LR>`: The peak learning rate [default: 0.0002]
- `--alpha <A>`: The LoRA alpha, the update scaled by alpha / rank [default: 2 x the rank]
- `--weight-decay <WD>`: The AdamW weight decay on the adapter's matrices [default: 0]
- `--bf16-base`: Hold the frozen base at bf16, half the bytes of fp32.
- `--continue-from-release`: Continue the current release on the claims not yet absorbed, with a replay of earlier releases' records, instead of training again from the base on every live claim. An ablation: the default makes supersession and forgetting exact.
- `--replay-fraction <F>`: With --continue-from-release, the fraction of each earlier release's records replayed. Default: `0.25`.
- `--subjects <FILE>...`: Files (JSON Lines of {"id", "terms"}) saying what the facts the sessions are known to be about (`extra.absorb_subject` of their trajectory) look like in a statement; the report then says whether the claims cover them.
- `--sealed-probes <FILE>...`: Files of sealed probes (JSON Lines of {"question", "reference"?, "name"?}): a training record containing a probe's question, or an 8-word run of a probe beyond its claim's statement, is refused. The gate never reads them.
- `--stance-pairs <N>`: The preference pairs the stance pass trains on after the supervised candidate: facts held against a confident falsehood, a statement accepted without support invented for it, questions nobody could answer declined. Derived afresh each night from the live claims, split evenly over the three kinds and bounded to half the night's records. Default: `96`.
- `--stance-beta <BETA>`: The DPO temperature of the stance pass [default: brain's]
- `--stance-tries <N>`: The replies drawn before a stance pair is given up for want of a reply to reject. Default: `8`.
- `--no-stance`: Train no stance pass: the supervised candidate is the night's.
- `--consolidate-after-records <N>`: Consolidate the generation into the weights once its claims hold more trained records than this: retraining the adapter from the base each night has become impractical, so a full checkpoint becomes the next generation's base and its claim set starts empty. Default: `2000`.
- `--no-consolidate`: Never consolidate: the generation's claims accumulate.
- `--consolidate-mode <MODE>`: How a due consolidation takes the generation in: `full` trains every weight, `merge` folds the adapter in. Default: `full`.
- `--retention-replay-share <SHARE>`: The share of the training draws that are records of the claims the generation's base already holds, in [0, 1) Default: `0.25`.
- `--dry-run`: Stop after the claims are extracted and ruled on; train nothing.
- `--no-release`: Stop at the trained candidate: do not run the gate.

### splinter consolidate

Take the generation the policy is in into the weights: a full checkpoint, trained from the generation's base on its claims (or the adapter folded in), gated against the release in use, becomes the base of the next generation, whose claim set starts empty. The claims it holds must keep being answered at every release after

```text
splinter consolidate [OPTIONS]
```

Options:

- `--policy <REF>`: The policy to consolidate: the checkpoint is released under its alias. Default: `policy:default`.
- `--mode <MODE>`: `full` trains every weight of the generation's base on the claims' records, beside a sample of the records of the claims the base already holds and the base's own answers; `merge` folds the adapter into the base's weights and trains nothing (the only mode that fits a model whose weights and gradients do not fit the cards) Default: `full`.
- `--from <RELEASE>`: The release to consolidate, by id or unique prefix (default: the one the policy points at)
- `--rehearsal-share <SHARE>`: The share of the training draws that are the base's own answers to general tasks, in [0, 1); 0 turns it off. Full mode only. Default: `0.25`.
- `--rehearsal-seed <N>`: The seed the rehearsal's form tasks are built with [default: drawn and printed]
- `--retention-replay-share <SHARE>`: The share of the training draws that are records of the claims the base already holds, in [0, 1). Full mode only. Default: `0.25`.
- `--epochs <N>`: The most passes over the records. Full mode only. Default: `6`.
- `--steps <N>`: The step budget, in place of --epochs. Full mode only.
- `--lr <LR>`: The peak learning rate [default: 0.00001]
- `--sealed-probes <FILE>...`: Files of sealed probes (JSON Lines of {"question", "reference"?, "name"?}): a training record containing one is refused.
- `--no-release`: Gate the checkpoint but do not release it: it is not kept.

### splinter improve

Choose how to improve an agent on a target capability from what each way was measured to do, net of what it costs: more samples, another episode protocol, a trained candidate, or no change

```text
splinter improve [OPTIONS] <EXPERIMENT>...
```

Arguments:

- `<EXPERIMENT>...` (required): Improvement experiments, each a TOML file: the capability (a task set), the incumbent, the interventions to consider (more samples, another episode protocol, a trained candidate, no change) and what a success is worth. Each intervention is measured against the incumbent on the same tasks and the one with the greatest net value is chosen, or none; nothing is applied.

Options:

- `--compare-fixed`: Beside the choices, what applying one intervention to every experiment would have come to.

### splinter generation

Generations of a coding policy, each held to what the ones before it had: run one, or show the ledger and the full retention matrix

```text
splinter generation [OPTIONS] <COMMAND>
```

Subcommands:

- [`run`](#splinter-generation-run) - Run one generation of a coding policy
- [`matrix`](#splinter-generation-matrix) - The ledger of an experiment and the full retention matrix of the lineage `policy:<alias>` points at

#### splinter generation run

Run one generation of a coding policy: measure the champion on every protected task, `practice` the repository to a candidate, measure both on the protected tasks and the new exam, reject the candidate on a retention or regression limit, release it through the gate otherwise, and keep a ledger entry with everything it was decided on

```text
splinter generation run [OPTIONS] --experiment <NAME> <REPO>
```

Arguments:

- `<REPO>` (required): The git repository to learn from: a work tree or a bare repository.

Options:

- `--experiment <NAME>` (required): The experiment this generation belongs to; names its ledger.
- `--max-family-drop <SHARE>`: The most a protected family's pass rate may fall [default: 0.2]
- `--max-mean-drop <SHARE>`: The most the mean over protected families may fall [default: 0.05]
- `--min-new-gain <SHARE>`: What the candidate must gain on the new exam: more than this [default: 0]
- `--rev <REV>`: The revision of it to learn from: anything git resolves to a commit. The tracked files of that commit are the snapshot every task is pinned to; uncommitted work is not in it. Default: `HEAD`.
- `--allow-dirty`: Snapshot the committed files of a revision that is checked out with uncommitted changes, instead of refusing it.
- `--file <PATH>...`: A file, relative to the repository, that tasks may change; repeat for more. Without any, every source file of the project the rules recognise (a Python project with a tests directory, a Cargo project)
- `--max-tasks <N>`: The most tasks one run admits.
- `--mutants-per-file <N>`: The most faults seeded in one file.
- `--exam-families <N>`: How many families (a family is the file a fault was seeded in) to reserve for the exam before any attempt is made at any task: their tasks are set apart and never solved or trained on, in this run or a later one; 0 reserves none [default: a quarter of the families]
- `--canaries <N>`: The most canaries (exam tasks no honest change can pass) the release gate hands the candidate; a pass on one refuses it; 0 hands none [default: 20]
- `--exam-seed <N>`: The seed the exam families are chosen by when this run reserves the first: drawn and printed [default], then kept with the families, so every later run reads it back and naming another is refused.
- `--k <N>`: Attempts per task when measuring pass@k. Default: `4`.
- `--temperature <T>`: The policy's sampling temperature for its pass@k attempts [default for a local model: 0.8; a remote model samples as its server does unless this is given]
- `--top-k <N>`: The policy's top-k cut for its pass@k attempts [default for a local model: 20]
- `--follow-ups <N>`: Follow-up turns an agent is sent after the checks it was shown stay red [default: 2]
- `--max-tool-calls <N>`: Tool calls an agent may make in one attempt, follow-ups included [default: 60]
- `--max-repeated-results <N>`: How often one tool call may return the same answer before the attempt is stopped as going nowhere [default: 3]
- `--budget <DUR>`: Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
- `--no-release`: Stop at the trained candidate: do not run the release gate.
- `--teacher <reference|REF>`: Who demonstrates the tasks the policy never solves, so they are not dropped: `reference` replays each task's own reference change as an agent's actions, or a model that solves them as an agent (`remote:`/`local:`). Either is graded by the same independent verifier as the policy's attempts; it never grades itself.
- `--replay <POLICY>`: Which of the earlier releases' records are replayed beside the new ones (default uniform): none, uniform over every record, or balanced over capabilities. Values: `none`, `uniform`, `balanced`, `failure-targeted`, `learning-progress`.
- `--rebuild`: Train a fresh adapter on the frozen base from the records of every earlier release and the new ones, instead of continuing the champion.
- `--curriculum <STRATEGY>`: The order tasks are solved and taught in (`--budget` may end the run before all are reached): the order they were admitted in, or by learning progress (the change in a family's verified pass rate between releases, per GPU second it cost) Default: `frontier`. Values: `frontier`, `learning-progress`, `gain-ranked`.
- `--curriculum-seed <N>`: The seed of the learning-progress draw (drawn, printed and recorded if not given)
- `--lp-window <N>`: Releases in each of the two windows learning progress compares. Default: `2`.
- `--lp-novelty <X>`: The novelty bonus of a family nothing has attempted, in pass-rate units; it shrinks with attempts. Default: `0.1`.
- `--lp-explore <X>`: The share of the draw that is uniform over families, in [0, 1]: no family gets less than this share divided by the number of families. Default: `0.1`.
- `--alpha-spending`: Spend the significance level across the releases of a lineage by a fixed schedule, so that a long run of candidates no better than their champions releases one with probability below the level.
- `--target-effect <SHARE>`: The lead, as a share of the exam's families won, the exam must be able to see: a miss on an exam too small to see it is reported inconclusive, not as a failure.
- `--anchor-max-net-lost <COUNT>`: Decide the anchor check on the answers the candidate loses net of the answers it gains, refusing above this count (6 for a 200-question suite), instead of on the drop in accuracy.
- `--steps <N>`: The step budget of the training: the most steps it may take.
- `--rank <N>`: LoRA rank of the adapter [default: 8]
- `--max-block <TOKENS>`: The longest training record, in tokens, the card fits: a longer record is left out of the dataset and counted, instead of stopping the training.
- `--lr <LR>`: The peak learning rate [default: 0.0002]
- `--alpha <A>`: The LoRA alpha, the update scaled by alpha / rank [default: 2 x the rank]
- `--weight-decay <WD>`: The AdamW weight decay on the adapter's matrices [default: 0]
- `--records-per-step <N>`: Records averaged into one optimizer step.
- `--seed <N>`: The seed of the adapter's initialisation and the batch order (drawn, printed and recorded if not given)
- `--eval-every <N>`: Steps between evaluations of the monitoring records; 0 monitors nothing and the candidate carries its last step [default: 16 evaluations over the step budget]
- `--patience <N>`: Evaluations without improvement before the training stops; 0 runs the whole budget (the best evaluation is carried either way) [default: 4]
- `--monitor-share <SHARE>`: The share of the training families set aside as the monitoring records, in (0, 0.5]: whole families, never the held-out ones the gate and the exam decide on [default: 0.1]
- `--keep-evaluations`: Keep the adapter of every evaluation with the candidate, so a step can be chosen on the dev suite (`select`) instead of the monitoring loss.
- `--bf16-base`: Hold the frozen base at bf16, half the bytes of fp32.
- `--plan <FILE>`: An experiment plan (`experiment check`) this run is one generation of: the repository must be a family the plan trains in `--generation`, and a sealed family is always refused. The catalog is the catalog.toml beside the plan.
- `--generation <N>`: The generation of the plan, counted from 1.
- `--test-command <PROGRAM ARG...>...`: The command that runs the project's whole test suite in the root of its tree, as a program and its arguments (everything after this flag, so it comes last). Without it the rules give one: `python3 -m unittest` for a Python project, `cargo test --offline` for a Cargo project.

#### splinter generation matrix

The ledger of an experiment and the full retention matrix of the lineage `policy:<alias>` points at: every generation, the base included, measured on every protected task

```text
splinter generation matrix [OPTIONS] --experiment <NAME>
```

Options:

- `--experiment <NAME>` (required): The experiment whose ledger is shown.
- `--alias <ALIAS>`: The alias whose lineage is measured. Default: `default`.

### splinter session

Take in recorded agent sessions, secrets removed

```text
splinter session [OPTIONS] <COMMAND>
```

Subcommands:

- [`add`](#splinter-session-add) - Take recorded agent sessions in as sources
- [`list`](#splinter-session-list) - List the stored sessions

#### splinter session add

Take recorded agent sessions in as sources: ATIF files, or directories searched for *.atif.json. Each is validated, refused with the reason when it cannot be rendered as training conversations, stripped of secrets (keys, tokens, passwords, private keys) and stored once by content; the same file again changes nothing. Exits 1 when any file was refused

```text
splinter session add [OPTIONS] <PATH>...
```

Arguments:

- `<PATH>...` (required): ATIF files, or directories of them.

Options:

- `--max-bytes <BYTES>`: The largest session file taken, in bytes. Default: `67108864`.

#### splinter session list

List the stored sessions

```text
splinter session list [OPTIONS]
```

### splinter claims

What a person taught in sessions: claims proposed by a model, ruled on by code, and the ledger of the rulings

```text
splinter claims [OPTIONS] <COMMAND>
```

Subcommands:

- [`extract`](#splinter-claims-extract) - Have a model propose the claims each stored session teaches
- [`gate`](#splinter-claims-gate) - Rule on a claim set
- [`forget`](#splinter-claims-forget) - Forget a live claim
- [`list`](#splinter-claims-list) - List the stored claim sets
- [`show`](#splinter-claims-show) - Show a claim set
- [`ledger`](#splinter-claims-ledger) - Show the ledger

#### splinter claims extract

Have a model propose the claims each stored session teaches - a correction, a fact, a procedure - with the person's exact words that support each, into a claim set. Exits 1 when a session's reply stayed unusable

```text
splinter claims extract [OPTIONS] <SESSION-ID>...
```

Arguments:

- `<SESSION-ID>...` (required): The sessions, by id or unique prefix, in the order they happened.

Options:

- `--generator <REF>`: The model that reads the sessions. Default: `policy:default`.
- `--passes <N>`: How many times each session is read, in different orders; only a claim every pass proposed is kept, the rest are listed as unconfirmed. Default: `2`.

#### splinter claims gate

Rule on a claim set: quotes verbatim in the person's words, the statement's numbers, names and quoted terms in them, the agent's sentences never evidence. Claims that name the same thing are paired: a later claim that contradicts the earlier supersedes it, one that says the same again reinforces it (recorded, never refused). Every ruling is kept, each refusal with its reason

```text
splinter claims gate [OPTIONS] <CLAIMSET-ID>
```

Arguments:

- `<CLAIMSET-ID>` (required): The claim set, by id or unique prefix.

Options:

- `--judge <REF>`: The model that decides whether a later claim supersedes, reinforces or is apart from an earlier one about the same thing; without one the statements decide.
- `--entail`: Whether the judge also refuses a claim whose cited words do not assert its statement (`--entail=false` turns it off) Values: `true`, `false`.
- `--subjects <FILE>...`: Files (JSON Lines of {"id", "terms"}) saying what the facts the sessions are known to be about (`extra.absorb_subject` of their trajectory) look like in a statement; the report then says whether the claims cover them.

#### splinter claims forget

Forget a live claim: it leaves the live set, so the next `absorb` trains again from the base without it, even when nothing else is new. The ruling and the claim's evidence stay in the ledger

```text
splinter claims forget [OPTIONS] <CLAIM-ID>
```

Arguments:

- `<CLAIM-ID>` (required): The live claim, by id or unique prefix (`claims ledger` lists them)

#### splinter claims list

List the stored claim sets

```text
splinter claims list [OPTIONS]
```

#### splinter claims show

Show a claim set: its proposals per session

```text
splinter claims show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The claim set, by id or unique prefix.

#### splinter claims ledger

Show the ledger: live claims with how often each was said again, superseded ones with by which, and refused proposals with their reasons

```text
splinter claims ledger [OPTIONS]
```

### splinter tasks

Generate and inspect task sets

```text
splinter tasks [OPTIONS] <COMMAND>
```

Subcommands:

- [`generate`](#splinter-tasks-generate) - Generate tasks from stored sources into a task set
- [`variants`](#splinter-tasks-variants) - Write differently worded questions about each task of a task set, into a task set of their own
- [`canaries`](#splinter-tasks-canaries) - Make canaries of the workspace tasks of a task set
- [`pick`](#splinter-tasks-pick) - Divide the union of task sets into a picked set and the rest, by a stable hash of the seed and the task's family (the practice exam's ranking)
- [`harbor`](#splinter-tasks-harbor) - Harbor tasks
- [`list`](#splinter-tasks-list) - List the stored task sets
- [`show`](#splinter-tasks-show) - Show a task set's tasks, or one task

#### splinter tasks generate

Generate tasks from stored sources into a task set

```text
splinter tasks generate [OPTIONS] --kinds <K,..> <SOURCE-ID>...
```

Arguments:

- `<SOURCE-ID>...` (required): The sources, by id or unique prefix.

Options:

- `--kinds <K,..>...` (required): Task kinds, comma-separated.
- `--generator <REF>`: The model that writes the tasks. Default: `policy:default`.
- `--author <NAME>`: Who wrote the sources, when they are one person's: the generator is told, and a question may name the author as its subject.
- `--denoise-seed <N>`: The seed the `denoise` tasks are built with (drawn and printed when not given and the kinds include denoise)

#### splinter tasks variants

Write differently worded questions about each task of a task set, into a task set of their own: the same facts, to be measured and never trained on

```text
splinter tasks variants [OPTIONS] <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

Options:

- `--generator <REF>`: The model that writes the variants (the one that wrote the tasks) Default: `policy:default`.
- `--per-task <N>`: The most variants one task is given. Default: `3`.

#### splinter tasks canaries

Make canaries of the workspace tasks of a task set: each task with its hidden assertions also required to fail, so that no honest change passes it. They go into a task set of their own, which `release --canaries` hands the gate: a candidate that passes one is refused

```text
splinter tasks canaries [OPTIONS] <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

Options:

- `--count <N>`: The most canaries made. Default: `20`.

#### splinter tasks pick

Divide the union of task sets into a picked set and the rest, by a stable hash of the seed and the task's family (the practice exam's ranking): a development holdout, a sample of one task per family, or one set out of several. Naming the sets in another order, or a task in two of them, changes nothing. A task that declares no family is a family of its own

```text
splinter tasks pick [OPTIONS] --seed <N> <TASKSET-ID>...
```

Arguments:

- `<TASKSET-ID>...` (required): The task sets, by id or unique prefix.

Options:

- `--family-share <F>`: Pick this fraction (0 to 1) of the families, every task of each.
- `--per-family <N>`: Pick at most this many tasks of every family.
- `--seed <N>` (required): The seed that decides what is picked.

#### splinter tasks harbor

Harbor tasks: discover, import and export

```text
splinter tasks harbor [OPTIONS] <COMMAND>
```

Subcommands:

- [`discover`](#splinter-tasks-harbor-discover) - List the Harbor tasks under a directory and whether each can be imported, at the revision of the checkout it is in
- [`import`](#splinter-tasks-harbor-import) - Import Harbor tasks (each a task directory) as workspace tasks in one task set
- [`export`](#splinter-tasks-harbor-export) - Write the workspace tasks of a task set as Harbor tasks, one directory each under --out, with everything needed to import them back exactly

##### splinter tasks harbor discover

List the Harbor tasks under a directory and whether each can be imported, at the revision of the checkout it is in. Writes nothing

```text
splinter tasks harbor discover [OPTIONS] <DIR>
```

Arguments:

- `<DIR>` (required): The directory (a checkout of a task repository)

##### splinter tasks harbor import

Import Harbor tasks (each a task directory) as workspace tasks in one task set. A task the host cannot honour (a Dockerfile with RUN, a prebuilt image, a Windows or multi-step task) is refused with the reason; the rest are held to the bar of every workspace task

```text
splinter tasks harbor import [OPTIONS] <DIR>...
```

Arguments:

- `<DIR>...` (required): The task directories.

Options:

- `--oracle <ORACLE>`: What to do about a task's solution/solve.sh: `skip` does not run it, `try` runs it and imports without a reference when it does not run on the host, `require` refuses the task unless it runs and its change passes the task's check. Default: `try`. Values: `skip`, `try`, `require`.

##### splinter tasks harbor export

Write the workspace tasks of a task set as Harbor tasks, one directory each under --out, with everything needed to import them back exactly

```text
splinter tasks harbor export [OPTIONS] --out <DIR> <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

Options:

- `--out <DIR>` (required): The directory to write into (it must not hold the tasks yet)
- `--image <IMAGE>`: The base image of each Dockerfile. Default: `ubuntu:24.04`.

#### splinter tasks list

List the stored task sets

```text
splinter tasks list [OPTIONS]
```

#### splinter tasks show

Show a task set's tasks, or one task

```text
splinter tasks show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): A task set's or a task's id, or a unique prefix of it.

### splinter solve

Solve every task of a task set, each in the environment it records

```text
splinter solve [OPTIONS] <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The task set, by id or unique prefix.

Options:

- `--solver <REF>`: The model that solves. Default: `policy:default`.
- `--frontier`: Solve each task k times, grade every attempt, have a teacher solve open-book each task never solved, and keep the tasks worth training on: failed at least sometimes, with a verified answer.
- `--teacher <REF>`: With --frontier: the model that solves open-book the tasks the solver never solves (default: the solver)
- `--k <N>`: Attempts per task when measuring pass@k. Default: `4`.
- `--temperature <T>`: The policy's sampling temperature for its pass@k attempts [default for a local model: 0.8; a remote model samples as its server does unless this is given]
- `--top-k <N>`: The policy's top-k cut for its pass@k attempts [default for a local model: 20]
- `--harness <FILE>`: Run this sven harness document on each workspace task in place of the conversational agent loop; the attempts are verified and compared like any other (`experiences compare`)
- `--resume <EXPERIENCE-SET>`: Resume an interrupted solve: the experience set its last progress line named. A task that set already answers is not solved again, and its experiences are members of the new set.

### splinter harness

Improve a sven harness: candidates are run on a frozen task set, verified, compared with the baseline, and promoted only on a verified improvement with no regression

```text
splinter harness [OPTIONS] <COMMAND>
```

Subcommands:

- [`optimise`](#splinter-harness-optimise) - Search a space of variants of a baseline harness for one that verifiably beats it

#### splinter harness optimise

Search a space of variants of a baseline harness for one that verifiably beats it: each candidate is run on the task set through sven, its attempts verified by the tasks' own verifiers and compared with the baseline's (a family counted once). A candidate is promoted only when no family got worse and the sign test over families finds it better; the promotion moves the `harness/champion` pointer

```text
splinter harness optimise [OPTIONS] --baseline <FILE> --evaluations <N> --search-seed <N> <TASKSET-ID>
```

Arguments:

- `<TASKSET-ID>` (required): The frozen task set every harness is run on, by id or unique prefix.

Options:

- `--solver <REF>`: The model every harness runs on. Default: `policy:default`.
- `--baseline <FILE>` (required): The baseline sven harness document (YAML)
- `--space <FILE>`: The mutation space: a YAML list of knobs, each a `path` into the document and the `choices` it may take. Required by the mutation proposer; with a model proposer it is the fallback for a proposal the model cannot make compile.
- `--proposer <PROPOSER>`: Who proposes candidates: a mechanical mutation of the space, or a model asked for edits that sven's compiler must accept. Default: `mutation`. Values: `mutation`, `model`.
- `--proposer-model <REF>`: The proposing model (with `--proposer model`)
- `--proposer-deadline <SECS>`: Seconds one model proposal may take, corrections included. Default: `300`.
- `--proposer-repairs <N>`: Corrections a model reply that does not compile is sent back for. Default: `3`.
- `--parallel <N>`: Candidates evaluated at once (best-of-N only): each is a whole arm, so the model server sees that many arms' attempts together. Default: `1`.
- `--strategy <STRATEGY>`: How candidates are searched. Default: `annealing`. Values: `best-of-n`, `annealing`.
- `--evaluations <N>` (required): Harnesses to evaluate, the baseline's replicate (the first) included.
- `--attempts <K>`: Solves per task and harness. Default: `1`.
- `--alpha <P>`: A candidate's sign test must have a p-value below this to be promoted. Default: `0.05`.
- `--min-paired <N>`: The fewest tasks with a verdict in both arms for a promotion. Default: `5`.
- `--search-seed <N>` (required): Seeds the search's draws; the same seed reproduces the candidates drawn.
- `--interval-seed <N>`: The seed the comparisons' intervals are resampled with (drawn and printed when not given)
- `--anneal-temperature <T>`: Annealing: the starting temperature, in shares of tasks solved. Default: `0.05`.
- `--anneal-cooling <F>`: Annealing: the factor the temperature is multiplied by each step. Default: `0.97`.

### splinter verify

Append verdicts to an experience set (re-grades without re-running)

```text
splinter verify [OPTIONS] <EXPERIENCE-SET>
```

Arguments:

- `<EXPERIENCE-SET>` (required): The experience set, by id or unique prefix.

Options:

- `--judge <REF>`: A calibrated judge for the kinds a judge grades.

### splinter critique

Critique the failed experiences of a set and retry them with it

```text
splinter critique [OPTIONS] <EXPERIENCE-SET>
```

Arguments:

- `<EXPERIENCE-SET>` (required): The experience set, by id or unique prefix.

Options:

- `--critic <REF>`: The model that critiques. Default: `policy:default`.
- `--retry <N>`: Retries per failed experience. Default: `1`.

### splinter judge

Measure a judge against labelled experiences

```text
splinter judge [OPTIONS] <COMMAND>
```

Subcommands:

- [`calibrate`](#splinter-judge-calibrate) - Measure a judge on labelled experiences and keep the calibration `verify --judge` gates its verdicts by
- [`measure`](#splinter-judge-measure) - Measure a judge on the controls a task set's own references give

#### splinter judge calibrate

Measure a judge on labelled experiences and keep the calibration `verify --judge` gates its verdicts by

```text
splinter judge calibrate [OPTIONS] --judge <REF> <LABELLED-FILE>
```

Arguments:

- `<LABELLED-FILE>` (required): JSON Lines: {"experience": "\<id\>", "label": "pass"|"fail"}.

Options:

- `--judge <REF>` (required): The judge.

#### splinter judge measure

Measure a judge on the controls a task set's own references give - each task's reference as the right answer, another family's as the wrong one - and show its precision, whether it would be trusted and the controls it got wrong. A calibration that rests on enough controls is kept, and a run reuses it

```text
splinter judge measure [OPTIONS] --judge <REF> <TASK-SET>
```

Arguments:

- `<TASK-SET>` (required): The task set, by id or unique prefix.

Options:

- `--judge <REF>` (required): The judge.
- `--fit`: Measure it as a judge of fit - is a passage a natural reply to its message - and not of whether an answer gives the reference.

### splinter experiences

Inspect experience sets and experiences

```text
splinter experiences [OPTIONS] <COMMAND>
```

Subcommands:

- [`list`](#splinter-experiences-list) - List the stored experience sets
- [`show`](#splinter-experiences-show) - Show an experience set, or one experience with its annotations
- [`replay`](#splinter-experiences-replay) - Run an experience's code calls again in the environment it records (or every experience of a set) and compare each result with the one it observed
- [`ingest-harbor`](#splinter-experiences-ingest-harbor) - Read the trials Harbor left for imported tasks (each a trial directory with result.json and agent/trajectory.json in ATIF) as experiences carrying Harbor's own verdict
- [`compare`](#splinter-experiences-compare) - Compare two experience sets of the same tasks

#### splinter experiences list

List the stored experience sets

```text
splinter experiences list [OPTIONS]
```

#### splinter experiences show

Show an experience set, or one experience with its annotations

```text
splinter experiences show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): An experience set's or an experience's id, or a unique prefix.

Options:

- `--graph`: Show the relations around it instead: retries, critiques, preferences.

#### splinter experiences replay

Run an experience's code calls again in the environment it records (or every experience of a set) and compare each result with the one it observed

```text
splinter experiences replay [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): An experience set's or an experience's id, or a unique prefix.

#### splinter experiences ingest-harbor

Read the trials Harbor left for imported tasks (each a trial directory with result.json and agent/trajectory.json in ATIF) as experiences carrying Harbor's own verdict: a reward of exactly 1 passes, any other fails, no reward or an exception is no verdict

```text
splinter experiences ingest-harbor [OPTIONS] <TASKSET-ID> <TRIAL>...
```

Arguments:

- `<TASKSET-ID>` (required): The task set the trials' tasks were imported into.
- `<TRIAL>...` (required): The trial directories.

#### splinter experiences compare

Compare two experience sets of the same tasks: the candidate's against the baseline's, paired by task, a family counted once. Exposed and harness-failed attempts are left out and counted

```text
splinter experiences compare [OPTIONS] <CANDIDATE-SET> <BASELINE-SET>
```

Arguments:

- `<CANDIDATE-SET>` (required): The candidate's experience set, by id or unique prefix.
- `<BASELINE-SET>` (required): The baseline's experience set, by id or unique prefix.

Options:

- `--interval-seed <N>`: The seed the intervals are resampled with (drawn and printed when not given; the same sets and seed give the same intervals)

### splinter dataset

Build datasets from experience sets, and export them

```text
splinter dataset [OPTIONS] <COMMAND>
```

Subcommands:

- [`build`](#splinter-dataset-build) - Project experience sets through a view into a stored dataset
- [`export`](#splinter-dataset-export) - Copy a dataset and its manifest into a directory

#### splinter dataset build

Project experience sets through a view into a stored dataset

```text
splinter dataset build [OPTIONS] --view <VIEW> <EXPERIENCE-SET>...
```

Arguments:

- `<EXPERIENCE-SET>...` (required): The experience sets, by id or unique prefix.

Options:

- `--view <VIEW>` (required): sft-final, sft-step, critic, preference, verifier, decision, retrieval, outcome, denoise, cpt, voice or rehearsal.
- `--strip <all|keep:K,..|mix:F>`: What the student sees: all (only the instruction), keep:K,.. or mix:F.
- `--mix-seed <N>`: The seed that picks which subjects keep their privileged context under `--strip mix:F` (drawn and printed when not given; a dataset built again with the same seed holds the same records)
- `--min-strength <STRENGTH>`: The weakest decision counted: executable, formal, consistency (default) or judged.
- `--system-prompt <TEXT>`: The system prompt every conversation opens with, in place of the default assistant's; the dataset records it.
- `--export-only`: Write an objective brain cannot train in the export format.
- `--limit <N>`: Keep at most this many records, evenly spread over the projection (for voice, over the parts of the sources)
- `--max-block <TOKENS>`: The longest record, in tokens, the training that reads the dataset fits: a longer record is left out and counted instead of stopping the training.
- `--pack`: For the sft-step view: one record per trajectory with every admitted action supervised, not one record per action.
- `--control <CONTROL>`: For the sft-step view: build another dataset than the default. `outcome-verified` admits only experiences an independent verdict passed (the default also admits an action labelled good from one that was not verified or failed; the manifest records `outcome-verified`). The negative controls: `unverified` keeps trajectories whatever their verdict (never an exposed one); `shuffled` gives each verified trajectory the instruction of another task. Values: `outcome-verified`, `unverified`, `shuffled`.
- `--control-seed <N>`: The seed of the `shuffled` pairing (drawn and printed when not given)
- `--drop-leaking`: Leave out, and list in the report, every experience that would carry an exam family's attempt or file into the dataset, instead of refusing the build. What remains is sealed as usual.
- `--serialise-calls`: For the sft-step view: a trajectory exported as one agent step that holds a whole run's tool calls is read as one call per turn, each followed by its result, then the final reply.
- `--writer <NAME>`: For the voice view: who the writer is. Requests are written as theirs and half the records open with their persona prompt, half with the line that says who the model is; refused with --system-prompt.
- `--token-limit <N>`: For the voice view: keep at most this many tokens of the writer's text, evenly spread over all of it.
- `--max-family-share <SHARE>`: For the voice view: the most of the tokens one family may supply.
- `--describe-with <REF>`: For the voice view: the model that writes, for each passage, a description of what it is about, which its request carries in place of the passage's opening (a passage with no admitted description keeps the opening)

#### splinter dataset export

Copy a dataset and its manifest into a directory

```text
splinter dataset export [OPTIONS] --out <DIR> <DATASET-ID>
```

Arguments:

- `<DATASET-ID>` (required): The dataset, by id or unique prefix.

Options:

- `--out <DIR>` (required): The directory to copy into.

### splinter train

Train a candidate adapter on datasets; it is not released

```text
splinter train [OPTIONS] <DATASET-ID>...
```

Arguments:

- `<DATASET-ID>...` (required): The datasets, by id or unique prefix, all chat or all preference pairs; the newest records of the last are held out for scoring.

Options:

- `--from <REF>`: The base to train, and an adapter on it to continue. Default: `policy:default`.
- `--base-precision <TIER>`: How the frozen base is held while the adapter trains: fp32, bf16, int8 or nf4 [default: fp32, or bf16 when SPLINTER_BF16_BASE is set]
- `--replay-fraction <F>`: The fraction of each earlier release's training records replayed, when training from a policy alias. Default: `0.25`.
- `--steps <N>`: The step budget: the most steps the training may take [default: 3 passes over the examples, within bounds]
- `--rank <R>`: LoRA rank of a new adapter. Default: `8`.
- `--beta <BETA>`: The DPO temperature, for preference datasets only [default: 0.1]
- `--max-block <TOKENS>`: The longest training record, in tokens, the card fits: a longer record is named and refused before any device is claimed (build the dataset with the same `--max-block` to leave such records out)
- `--lr <LR>`: The peak learning rate [default: 0.0002]
- `--alpha <A>`: The LoRA alpha, the update scaled by alpha / rank [default: 2 x the rank]
- `--weight-decay <WD>`: The AdamW weight decay on the adapter's matrices [default: 0]
- `--records-per-step <N>`: Records averaged into one optimizer step (default: one when the steps are named, else from the size of the dataset)
- `--seed <N>`: The seed of a fresh adapter's initialisation, the batch order and the draw of the records replayed (drawn, printed and recorded in the run and the candidate if not given)
- `--rehearsal <DATASET-ID>`: A rehearsal dataset (`rehearse`, or any chat dataset of the base's own answers) mixed into training at --rehearsal-share of the draws, never held out, a share of it in the monitoring set.
- `--rehearsal-share <F>`: The share of the training draws the rehearsal takes, in (0, 1) [default: 0.25]
- `--eval-every <N>`: Steps between evaluations of the monitoring records; 0 monitors nothing and the candidate carries its last step [default: 16 evaluations over the step budget]
- `--patience <N>`: Evaluations without improvement before the training stops; 0 runs the whole budget (the best evaluation is carried either way) [default: 4]
- `--monitor-share <SHARE>`: The share of the training families set aside as the monitoring records, in (0, 0.5]: whole families, never the held-out ones the gate and the exam decide on [default: 0.1]
- `--keep-evaluations`: Keep the adapter of every evaluation with the candidate, so a step can be chosen on the dev suite (`select`) instead of the monitoring loss.

### splinter rehearse

Build a rehearsal dataset: the base model's own answers to general tasks - sums and format requests built by code, and requests the base writes itself - never an anchor task or a near copy of one; for `train --rehearsal`, so a fine-tune keeps what the base does

```text
splinter rehearse [OPTIONS] --records <N>
```

Options:

- `--records <N>` (required): How many tasks the set holds: as many as make the rehearsal share of the examples it will be trained beside (`learn` sizes it so)
- `--rehearsal-seed <N>`: The seed the form tasks are built with (drawn and printed when not given; the same seed and count build the same tasks)

### splinter release

Run the release gate on a candidate and release it if every check passes; or list the releases

```text
splinter release [OPTIONS] [CANDIDATE-ID]
       splinter release <COMMAND>
```

Arguments:

- `<CANDIDATE-ID>`: The candidate, by id or unique prefix.

Options:

- `--alias <NAME>`: The alias it replaces the champion of. Default: `default`.
- `--judge <REF>`: The calibrated model that judges the tasks a judge decides.
- `--unrestricted`: Release for distribution and commercial use: refused unless the terms of everything the candidate was made from allow both. Without it the release is restricted: it records the terms and stays local.
- `--interval-seed <N>`: The seed the gate's bootstrap intervals are resampled with (drawn and printed when not given; the same outcomes and seed give the same intervals)
- `--canaries <TASKSET-ID>`: A task set of canaries (`tasks canaries`) the candidate attempts beside the exam: a pass on any one refuses it, whatever else it passed.
- `--alpha-spending`: Spend the significance level across the releases of a lineage by a fixed schedule, so that a long run of candidates no better than their champions releases one with probability below the level.
- `--target-effect <SHARE>`: The lead, as a share of the exam's families won, the exam must be able to see: a miss on an exam too small to see it is reported inconclusive, not as a failure.
- `--anchor-max-net-lost <COUNT>`: Decide the anchor check on the answers the candidate loses net of the answers it gains, refusing above this count (6 for a 200-question suite), instead of on the drop in accuracy.

Subcommands:

- [`list`](#splinter-release-list) - List the releases and the aliases pointing at them

#### splinter release list

List the releases and the aliases pointing at them

```text
splinter release list [OPTIONS]
```

### splinter rollback

Point an alias back at the release its current one was trained from

```text
splinter rollback [OPTIONS] <ALIAS>
```

Arguments:

- `<ALIAS>` (required): The alias, e.g. default.

### splinter eval

Grade a candidate or a model closed-book on a suite; or freeze and show the anchor suite

```text
splinter eval [OPTIONS] [REF]
       splinter eval <COMMAND>
```

Arguments:

- `<REF>`: A candidate id (or prefix) or a model reference; none shows the anchor suite.

Options:

- `--suite <SUITE>`: held-out, retention, anchor, or an anchor-format file of tasks. Default: `held-out`.
- `--freeze <FILE>...`: With --suite anchor: make these files' tasks (repeat the flag for several) the anchor suite's next version first. JSON Lines: {"instruction", "reference", "kind"?}.
- `--judge <REF>`: The calibrated model that judges the tasks a judge decides.

Subcommands:

- [`matrix`](#splinter-eval-matrix) - The continual-learning results matrix of a lineage

#### splinter eval matrix

The continual-learning results matrix of a lineage: what each gate measured of each release on each earlier release's suite, with the metrics read from it. A cell no gate measured is shown absent

```text
splinter eval matrix [OPTIONS]
```

Options:

- `--alias <NAME>`: The alias whose releases are the generations. Default: `default`.

### splinter exam

Put a candidate and the policy it continues to its held-out tasks and have a calibrated judge compare them; or, with --exam-set, to a frozen exam of reserved families, with every verdict kept

```text
splinter exam [OPTIONS] <CANDIDATE>
```

Arguments:

- `<CANDIDATE>` (required): The candidate to examine: its id, or a unique prefix of it.

Options:

- `--judge <REF>`: The model that judges (default: the judge role's model)
- `--prompt <GOAL>`: Also ask the base under this goal, as a prompt-only baseline the candidate is compared with (`learn` uses its goal)
- `--retrieve <SOURCE-ID>...`: Show the model the passages of these stored sources that bear on the question, found by meaning and by exact words, instead of all of one.
- `--passages <N>`: How many passages `--retrieve` shows. Default: `6`.
- `--reranker <REF>`: A model that reads the candidates search found beside the question and moves the ones that bear on it forward.
- `--exam-set <EXAM>`: Put the candidate to this frozen exam (its id, or its manifest file) instead: four arms - the base, the base told who the candidate is to be, the candidate, and the candidate under that prompt - compared per task with the family as the unit of evidence, every verdict kept.
- `--resamples <N>`: With --exam-set: answers per task per arm, the first greedy and the rest sampled [default: 3]
- `--pilot-families <N>`: With --exam-set: put only this many of the exam's families to the candidate, as a pilot that estimates the discordance and clustering.
- `--deployed-only`: With --exam-set: ask only the candidate under the prompt it is deployed with. Nothing is compared in the report; the arms of another report of the same exam (the base, the prompted base, the same tasks and families, greedy answers) are what its verdicts are paired with.
- `--no-voice`: With --exam-set: leave out the voice score, which loads each arm on the device to give the writer's own text a likelihood.
- `--interval-seed <N>`: The seed the bootstrap intervals are resampled with (drawn and printed when not given; the same records and seed give the same intervals)

### splinter select

Choose the step of a run kept with its evaluations by what each writes on the dev suite: right answers, no invented specifics, then voice

```text
splinter select [OPTIONS] --exam-set <EXAM> <CANDIDATE>
```

Arguments:

- `<CANDIDATE>` (required): The candidate, trained with --keep-evaluations: its id, or a unique prefix of it.

Options:

- `--exam-set <EXAM>` (required): The dev suite (its id, or its manifest file) every kept evaluation is put to.
- `--judge <REF>`: The model that judges (default: the judge role's model)
- `--adopt`: Make the chosen evaluation a candidate of its own, to be released.

### splinter exam-report

Analyses done after an exam from what its report kept: labels to calibrate the judge by, and memorisation

```text
splinter exam-report [OPTIONS] <COMMAND>
```

Subcommands:

- [`labels-export`](#splinter-exam-report-labels-export) - Write a blind, stratified, shuffled sample of the exam's answers for a person to label right or wrong, and the key that says which arm gave each and what the judge said
- [`labels-import`](#splinter-exam-report-labels-import) - What a person's labels say of the judge
- [`memorisation`](#splinter-exam-report-memorisation) - How much of each arm's answers is the text the candidate was trained to produce again

#### splinter exam-report labels-export

Write a blind, stratified, shuffled sample of the exam's answers for a person to label right or wrong, and the key that says which arm gave each and what the judge said

```text
splinter exam-report labels-export [OPTIONS] --out <FILE> --key <FILE> <REPORT>
```

Arguments:

- `<REPORT>` (required): The exam report (`exam --exam-set ... --json` output)

Options:

- `--out <FILE>` (required): Where the items to label are written (JSON Lines)
- `--key <FILE>` (required): Where the key is written (JSON)
- `--n <N>`: Items to sample. Default: `400`.
- `--seed <N>`: Varies the sample (drawn and printed when not given; the same report and seed give the same sample)

#### splinter exam-report labels-import

What a person's labels say of the judge: agreement, kappa, false right and false wrong rates, length bias

```text
splinter exam-report labels-import [OPTIONS] --key <FILE> <LABELS>
```

Arguments:

- `<LABELS>` (required): The items, each with its `label` filled in `right` or `wrong`.

Options:

- `--key <FILE>` (required): The key `labels-export` wrote.

#### splinter exam-report memorisation

How much of each arm's answers is the text the candidate was trained to produce again: overlap of 8, 13 and 20 word runs and the longest run

```text
splinter exam-report memorisation [OPTIONS] --candidate <CANDIDATE> <REPORT>
```

Arguments:

- `<REPORT>` (required): The exam report.

Options:

- `--candidate <CANDIDATE>` (required): The candidate the report is of.

### splinter exam-set

Make and show frozen exams: families reserved from sources, tasks written from their text alone

```text
splinter exam-set [OPTIONS] <COMMAND>
```

Subcommands:

- [`create`](#splinter-exam-set-create) - Capture the sources, reserve families of them, write the exam from that text alone and freeze it
- [`power`](#splinter-exam-set-power) - Simulate the planned paired test to size an exam before it is paid for
- [`show`](#splinter-exam-set-show) - Show a frozen exam

#### splinter exam-set create

Capture the sources, reserve families of them, write the exam from that text alone and freeze it

```text
splinter exam-set create [OPTIONS] <SOURCE>...
```

Arguments:

- `<SOURCE>...` (required): Files, directories, cmd:\<command\> (quoted as one argument), or stored source ids.

Options:

- `--families <N>`: How many families to reserve. Default: `50`.
- `--tasks-per-family <N>`: Tasks of each family of the final test. Default: `8`.
- `--dev-families <N>`: How many more families to reserve for the dev suite. Default: `50`.
- `--dev-tasks-per-family <N>`: Tasks of each dev family. Default: `4`.
- `--kinds <K,..>...`: Task kinds, comma-separated. Default: `converse,advise,explain`.
- `--generator <REF>`: The model that writes the tasks (default: the policy)
- `--goal <TEXT>`: What the learner is after; steers the tasks.
- `--persona <NAME>`: Who wrote the sources, when they are one person's.
- `--seed <N>`: Varies which families are reserved (drawn and printed when not given; the same seed over the same sources reserves the same families, and exams are comparable across runs only when they share one)
- `--not-trained-by <CANDIDATE>...`: A candidate (repeat the flag for several) whose training, and that of the releases it continues, no reserved family may overlap: the exam is for measuring it.

#### splinter exam-set power

Simulate the planned paired test to size an exam before it is paid for: its power at an assumed effect, and how often it calls a difference that is not there

```text
splinter exam-set power [OPTIONS]
```

Options:

- `--families <N>`: Families of the exam. Default: `50`.
- `--tasks-per-family <N>`: Tasks of each. Default: `8`.
- `--effect <SHARE>`: The lead of the first arm in the share of tasks right (0.10 is ten points) Default: `0.1`.
- `--discordance <SHARE>`: The share of tasks only one of the arms gets right. Default: `0.3`.
- `--icc <RHO>`: The intraclass correlation of the per-task difference in a family. Default: `0.1`.
- `--from-report <FILE>`: Take the discordance and the intraclass correlation of the primary comparison from the report of a pilot (`exam --json` output), not from an assumption.
- `--replicates <N>`: Simulated exams. Default: `400`.
- `--training-runs <N>`: The training runs the result will rest on: fewer than three, or fewer than a hundred families, and it is labelled a pilot. Default: `1`.
- `--seed <N>`: Seeds the simulation (drawn and printed when not given; the same plan and seed give the same answer)

#### splinter exam-set show

Show a frozen exam

```text
splinter exam-set show [OPTIONS] <EXAM>
```

Arguments:

- `<EXAM>` (required): Its id, or its manifest file.

### splinter runs

List, inspect and cancel runs

```text
splinter runs [OPTIONS] <COMMAND>
```

Subcommands:

- [`list`](#splinter-runs-list) - List the recorded runs
- [`show`](#splinter-runs-show) - Show one run's record
- [`cancel`](#splinter-runs-cancel) - Ask a run in progress to stop

#### splinter runs list

List the recorded runs

```text
splinter runs list [OPTIONS]
```

#### splinter runs show

Show one run's record: arguments, stages, status, outputs

```text
splinter runs show [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The run's id.

#### splinter runs cancel

Ask a run in progress to stop

```text
splinter runs cancel [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): The run's id.

### splinter state

Show, maintain, verify, repair, archive and restore the state: the experience database and the files it tracks

```text
splinter state [OPTIONS] <COMMAND>
```

Subcommands:

- [`status`](#splinter-state-status) - What the experience database holds, as files
- [`maintain`](#splinter-state-maintain) - Merge small files, index what is not indexed and retire finished writers
- [`unpin`](#splinter-state-unpin) - Let go of a snapshot a dataset (or another holder) keeps alive, named as `state status` lists it, so its files can be collected
- [`verify`](#splinter-state-verify) - Check the database and every file it tracks, and report each one that is missing or damaged
- [`repair`](#splinter-state-repair) - Recover what verify finds
- [`archive`](#splinter-state-archive) - Pack the database and the files it tracks into one archive
- [`restore`](#splinter-state-restore) - Unpack an archive (and the archives an incremental one was made after) into an empty state root, verifying everything first

#### splinter state status

What the experience database holds, as files

```text
splinter state status [OPTIONS]
```

#### splinter state maintain

Merge small files, index what is not indexed and retire finished writers; nothing stored changes

```text
splinter state maintain [OPTIONS]
```

Options:

- `--collect`: Also delete files nothing reaches that are past their grace period; a snapshot a dataset pinned is never touched.

#### splinter state unpin

Let go of a snapshot a dataset (or another holder) keeps alive, named as `state status` lists it, so its files can be collected

```text
splinter state unpin [OPTIONS] <HOLDER>
```

Arguments:

- `<HOLDER>` (required): The holder's name.

#### splinter state verify

Check the database and every file it tracks, and report each one that is missing or damaged; exits 1 when anything is

```text
splinter state verify [OPTIONS]
```

Options:

- `--deep`: Read every byte instead of only checking sizes.

#### splinter state repair

Recover what verify finds: fill holes from copies (archives or other state roots), rebuild indexes, and with --accept-loss write off what no copy has

```text
splinter state repair [OPTIONS]
```

Options:

- `--from <PATH>...`: A copy to take missing files from: an archive file, or another state root. May be given more than once.
- `--accept-loss`: Give up on what no copy has: withdraw damaged database files and record lost artifacts in the ledger, which `state status` lists.

#### splinter state archive

Pack the database and the files it tracks into one archive; the same state always gives the same bytes

```text
splinter state archive [OPTIONS] <FILE>
```

Arguments:

- `<FILE>` (required): Where to write the archive (a .tar.zst)

Options:

- `--no-artifacts`: Leave the artifacts (adapters, datasets) out.
- `--since <ARCHIVE>`: An earlier archive: carry only what it lacks. Restore the result together with that archive.

#### splinter state restore

Unpack an archive (and the archives an incremental one was made after) into an empty state root, verifying everything first

```text
splinter state restore [OPTIONS] <FILES>...
```

Arguments:

- `<FILES>...` (required): The archive to restore, then any it builds on.

### splinter lineage

Trace any artifact up to where it came from (down to source bytes) and down to everything that came from it

```text
splinter lineage [OPTIONS] <ID>
```

Arguments:

- `<ID>` (required): Any artifact's id - a source, task, experience, dataset, candidate, release, answer, ... - or a unique prefix of it.

Options:

- `--up`: Only where it came from.
- `--down`: Only what came from it.
- `--both`: Both ways (the default)
- `--depth <N>`: The most edges walked from it (default: all the way)
