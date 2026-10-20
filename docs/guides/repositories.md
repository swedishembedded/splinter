<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn a repository from its own tests

Use `splinter practice` to teach the policy a codebase. The project's own tests
are the verifier: faults are seeded into the code, and a task exists only where the
tests catch the fault every time. The agent works in a copy of the tree and is graded
by tests it never sees.

The agent runs on [sven](https://swedishembedded.com/docs/sven); the commands it
runs are bounded as described under [Confinement](#confinement). Read
[limitations](../concepts/limitations.md) before pointing it at code you do not trust.


```bash
splinter practice ./shop --rev v1.4
splinter practice ./shop --rev HEAD --exam-families 2 --k 4 --no-release \
    --test-command python3 -m pytest -x
```

`practice` learns a codebase from its own tests, as one recorded run. The
repository is a git work tree or bare repository and `--rev` (default `HEAD`)
any revision git resolves to a commit; the tracked files of that commit are
archived as the snapshot every task is pinned to (a checked-out revision with
uncommitted changes is refused unless `--allow-dirty`). Its stages, in
order:

| Stage | What it does |
|---|---|
| `policy` | resolves `policy:default` once |
| `snapshot` | archives the revision; the same tree is the same snapshot |
| `profile` | finds how the tests are run (`python3 -m unittest` for a Python project with a `tests` directory, `cargo test --offline` for a Cargo project; `--test-command` and `--file` name them otherwise) and runs the suite twice on a clean copy: a red or flaky baseline stops the run |
| `tasks` | seeds faults by code in the files under test and keeps a task only where the project's own tests catch it every time, the change reversed makes its check pass, and the catching tests live in files of their own, which are taken out of the tree the agent is given; a task the store already holds, or a request that repeats one made for the same file, is not made again, so a second run over the same snapshot ends here |
| `reserve` | sets the exam's families aside (a family is the file a fault was seeded in) by a stable hash of their names, before any attempt; `--exam-families N` (default a quarter), and a family an earlier run reserved stays the exam |
| `solve` | the policy attempts each remaining task `--k` times as an agent in its own copy of the tree, with sven's file, search and shell tools, up to `--follow-ups` more turns when the visible checks stay red and `--max-tool-calls` calls; each attempt draws from a seed that is a function of the task and the attempt |
| `verify` | applies each patch to a clean copy, lays the hidden tests over it and runs the project's checks; an exposure audit fails any attempt whose trajectory names a hidden path, quotes the hidden tests or names the state root, whatever its patch did, and the stage counts those by kind |
| `teach` | only with `--teacher`: each task no graded attempt solved is given to the teacher, which is `reference` (the task's own reference change, the inverse of the seeded fault, replayed as an agent's actions through sven's tools: read the file, edit it by the patch's hunks) or a model (`remote:`/`local:`) that solves the task as an agent. Either is graded by the same workspace verifier and exposure audit as the policy's attempts and never grades itself; the exam's tasks are never given to it. A demonstration names the file without searching for it |
| `frontier` | keeps the tasks the policy solves sometimes (pass@k), and with a teacher the never-solved tasks whose demonstration was verified |
| `dataset` | the passing attempts, action by action (`sft-step`), with the tools the agent was offered (an action a labeller marked good is also admitted from an experience nobody verified or whose verdict failed, unless `--control outcome-verified` is given; the manifest records it); `--max-block TOKENS` leaves out and counts records longer than the card fits; the dataset is checked to hold no family of the exam |
| `train` | a candidate from the champion with earlier releases' records replayed (`--replay none|uniform|balanced`; `--rebuild` trains a fresh adapter on the base from every earlier release's records and the new ones instead of continuing the champion) |
| `release` | the gate, over the exam this run reserved (`--no-release` stops at the candidate): the champion and the candidate attempt the exam's tasks as agents, under the bounds and seeds of `solve`, graded as `verify` grades; paired outcomes are counted by family in the sign test; each earlier release is held to the exam of its own run; the anchor suite is held (a frozen suite with no coding-agent task is decided on the tasks it holds, and the report says it holds none); the candidate is served by `brain serve --adapter-manifest` and must be graded as it was in-process on at least three quarters of a sample of the exam. A blocked candidate is not released and the run stops with the reason of each check that failed; `practice` gives the gate the bounds it solved under, and `splinter release` on such a candidate probes with the default ones |

Where no container is configured, the commands the agent works with run on
the host, bounded but not isolated, and every environment record says
`isolation: none`; naming a docker client and a digest-pinned image
(`SPLINTER_DOCKER`, `SPLINTER_WORKSPACE_IMAGE`) runs them in a container with
no network instead. `SPLINTER_CARGO_HOME` gives the commands a
cargo home to build offline from, and `SPLINTER_WORKSPACE_SCRATCH` says where
trees are laid out. An attempt whose trajectory shows the hidden tests,
and an attempt the harness could not carry out, are set aside and counted in the
gate's report; neither is in a denominator. A release moves the alias and
rewrites `<state>/serving/<alias>.json`, the manifest `brain serve
--adapter-manifest` follows (a rollback repoints it); an alias at a full
checkpoint has no adapter to name, so its manifest is withdrawn. A candidate the
gate blocks, or cannot measure, is not released, and the run exits 1 with the
reason. `--test-command` takes the rest of the command line, so it comes last.

A task the policy never solves is not dropped when a teacher is named (`splinter practice
--teacher reference`, or `--teacher remote:<provider>/<model>`): the task's own reference change
is replayed as the actions of an agent through sven's real tools (read the file, edit it by the
patch's hunks), or a stronger model solves it as an agent. Either is graded by the same
workspace verifier and exposure audit as the policy's attempts and never grades itself, and the
demonstration teaches what to do once the place is known, not how to find it. `--replay
none|uniform|balanced` chooses what of the earlier releases' records is replayed, and
`--rebuild` trains a fresh adapter on the base from all of them instead of continuing the
champion.

## Confinement

On the host the kernel confines what the agent runs: a command (the solver's
shell, the checks) reaches the system read only, its own tree and a private
directory, and nothing else of the filesystem (no home directory, no state
root with the hidden tests, no sibling attempt's tree); it cannot connect
anywhere, loopback included (Landlock denies TCP, a seccomp filter refuses
every internet socket, UDP too); and its environment holds no variable named
like a secret (key, token, password, auth socket), even when an allowlist
names one. Without a kernel that can do this (Landlock ABI 4) the command is
refused, never run open. `--allow-workspace-network` opens the network,
`--allow-unconfined-workspace` the filesystem on a kernel that cannot confine,
and `--workspace-max-processes N` bounds the processes and threads of a
command; each is part of the environment record, so a task generated without
it names another environment. The agent's own file and search tools are
confined to the attempt's tree by sven's project root, for its delegated
children too. A command, a code call of `run_code` included, runs in a private user and pid
namespace with a `/proc` of its own, so no other process of the host is visible to it
(Landlock alone already refuses it another process's environment); a kernel that cannot
make one refuses the command. The exposure
audit reads a command's mention of a path outside the tree by what that call's own
result shows (the shell's `[exit code N]` line is not a result, and the other calls of
the same step are not consulted): `outside_workspace` (something came back) fails the
audit; `outside_workspace_denied` (the sandbox refused) and `outside_workspace_absent`
(no such path, or nothing returned) are recorded as `outside_probes` in the evidence and
do not. A call that names a hidden path and only got a refusal or "no such file" is
recorded the same way (`hidden_path_absent`); a hidden path in what a tool returned,
hidden content and the state root still fail it.

## How a solver is run

A solver of a repository works under the coding-agent prompt (look, reproduce, change,
verify, never repeat a call, stop when the tests pass) unless the model it is asked for brings
a prompt of its own (a policy trained to be a person). Each attempt records what it was held
to in its trajectory's `solve_profile`: the prompt's id and digest, the longest shell call
(1800 s), sven's note on a repeated result (on), the repetition bound and the tool-call cap,
and how the model was asked to sample. A served model is asked to sample with `--temperature`,
`--top-p`, `--top-k` and `--sampling-seed` on every request (`--thinking` lets it reason),
and none is sent unless given, so the server's own settings apply.

The repetition stop counts a call as repeated when it has the same tool and arguments once a
`description`, the order of keys and whitespace are set aside, and the same result once sven's
note on a repeated result is removed; it is decided before the next model request and labels
the stopped turn `repetition stop: ...`. A run that ends announcing a call it never made
(`stopped_without_conclusion`) is not a success and is not sent another turn.

## What a task's request says

The request of a task made from a seeded fault names its symptoms in the words of the tests
that catch it (a short docstring, else the test's name) and the test command of the project
with the test modules taken out of the tree left out. A name of fewer than three words
(`databricks`, `time`) is not enough to act on, so such a symptom is listed with what the
failed check observed: the two values of an equality, or the error the code raised, as one
line each. A traceback frame, a diff, the test's path and the call of the assertion are never
shown; those are the hidden test's code. A candidate fault is refused, and counted, when a
test that stays in the tree imports a module that would be withheld
(`hidden_imported_by_tree`) or when a short symptom cannot be rendered this way
(`symptom_not_actionable`). The values shown are the one disclosure a task makes of its
hidden tests, so the exposure audit counts the request as text the solver may see.

## Generations of a coding policy

`splinter generation run REPO --experiment NAME --plan PLAN.toml --generation N` measures the
champion on every protected task (the exam of every earlier generation), runs `practice` to a
candidate, measures both on the protected tasks and the new exam at equal settings, rejects the
candidate when a protected family falls by more than `--max-family-drop`, the mean falls by more
than `--max-mean-drop` or it gains no more than `--min-new-gain` on the new exam, and otherwise
releases it through the gate. Every generation, accepted or not, is in the experiment's ledger.
`--replay none|uniform|balanced|failure-targeted|learning-progress` and `--rebuild` choose how the earlier
experience is carried forward. `splinter generation matrix --experiment NAME` prints the ledger
and the full retention matrix of the lineage: the base as generation 0 and every generation on
every protected task, with ACC, BWT, forgetting, forward transfer and learning gain.

`--replay learning-progress` weights the replay by how fast the policy's pass rate on each earlier
generation's exam has been moving across the releases that were the champion (the panel
measurements of the ledger and of this generation), with the learning-progress signal described
below (`--lp-window`, `--lp-novelty`, `--lp-explore`; a panel measurement records no device time,
so the cost divisor is neutral). Each earlier generation's training families take the weight of
that generation's exam, and none gets less than the exploration share divided by the number of
generations. The recorded `replay_policy` carries the weights.

## Which tasks a budget reaches first

`practice` and `generation run` go through the admitted tasks in the order they were admitted
and stop at `--budget`. `--curriculum learning-progress` or `--curriculum gain-ranked` puts them
in the order of how fast the policy is still improving, per GPU second, on each family (the file a
fault was seeded in), in a `schedule` stage between `reserve` and `solve`; the solve and teach
stages then reach those families first. The pass@k frontier still decides afterwards which
solved tasks are kept for training.

A family's learning progress is the absolute difference between its verified pass rate in the
newest `--lp-window` releases and in the window before, read from the attempts the store already
holds (the same history `status` shows as concept mastery). It is divided by the family's GPU
seconds per attempt relative to the mean over families (a family with no recorded device time is
priced at the mean, never at zero) and a novelty bonus (`--lp-novelty`, shrinking with attempts) is
added, so a family nothing has attempted is not ignored. `learning-progress` draws every family
once, without replacement and in proportion to that weight, from `--curriculum-seed` (drawn and
recorded when absent), mixed with the uniform distribution by `--lp-explore` so no family gets less
than that share divided by the number of families. `gain-ranked` takes the families heaviest
first, with no draw. The report's `schedule` lists each family's progress, cost, novelty and
weight, the order, and `informative`: how many families were measured under two or more
releases. A family first met by this run has no history, so on a repository none of whose files an
earlier release attempted `informative` is 0 and the order is the novelty-weighted draw; the
signal discriminates from the second release a family is attempted under. The report's `yield`
gives the device seconds of the solve and teach attempts against the tasks kept, so the
strategies can be compared per GPU hour.

## Related

- Check a plan of which repositories a lineage trains on and which are sealed as the final
  exam with `splinter experiment check`; the frozen repositories this project ships are
  described in [Task families](../reference/tasks.md#repository-families).
- Import tasks written for the Harbor benchmark format, or export these as Harbor tasks:
  [Harbor tasks and trials](harbor-tasks.md).
- Try other agent loops on the same tasks: [Improve a harness](improving-a-harness.md).
