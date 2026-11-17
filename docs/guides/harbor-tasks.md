<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Harbor tasks and trials

A Harbor task is a directory (`instruction.md`, `task.toml`, `environment/` with a
Dockerfile or `workspace/` with the starting files themselves, `tests/test.sh` that
writes a reward, optionally `solution/solve.sh`);
a splinter workspace task is the same thing as a request, a tree, hidden files and
the commands that must fail before and pass after.

```sh
splinter tasks harbor discover ~/harbor-tasks        # which tasks can be imported, and why not
splinter tasks harbor import ~/harbor-tasks/calc --oracle require
splinter tasks harbor export TASKSET-ID --out out/   # splinter tasks as Harbor tasks
splinter experiences ingest-harbor TASKSET-ID trials/*   # Harbor trials as experiences
splinter tasks harbor episodes ~/practice --attempts 8 --jobs 4   # practice folders run as graded episodes
```

*Import* admits a task whose environment is a repository checkout: a Dockerfile with
`FROM`, `WORKDIR` and `COPY` (no `RUN`, `ADD`, stages or wildcards), or no Dockerfile and
a `workspace/` directory that is the tree (its working directory is `[environment] workdir`,
else `/workspace`), no prebuilt image, a shell `tests/test.sh`, one step, Linux. Anything
else is refused with the reason. A task folder that is exam material is refused too: one
marked `[metadata].bench_sealed = true`, one in a held-out tier (`[metadata].bench_tier`
`b` or `c`), and one whose `[metadata].bench_family` is in the `eval` set of the
`split.toml` beside it; an unreadable `split.toml` refuses the folder rather than clearing
it.
The task is held to the bar of every workspace task: its check fails on the tree the
solver is given on every run, and with `--oracle require` the change `solve.sh` makes
(run on the host with `/solution` and the working directory mapped to the tree) must
pass the check. The task is graded by its own `test.sh`, run with `/tests`,
`/logs/verifier` and the working directory mapped to host paths (and named in
`WORKSPACE_DIR`, `TESTS_DIR` and `LOGS_DIR`); it passes only when `reward` is exactly 1,
read the way Harbor's verifier reads `reward.json` or `reward.txt`, so partial credit is
not a pass. A verifier that leaves no reward, or one that is not a number, fails the check
with exit status 3, which tells a verifier that broke from a change it judged wrong. A task
splinter exported carries its check in `[metadata.splinter]` and is rebuilt exactly:
export then import gives the same task, with the same content address. Harbor's
verifier does not enforce the check's protected and pinned paths; splinter's does.
Each imported task is recorded with where it came from (the checkout's commit, whether
it was dirty, and a digest of the task directory) outside the task, so the address does
not depend on where it was read; `tasks harbor discover` reads a task repository at its
revision and writes nothing.

*Export* also writes a [comprehension drill](../reference/tasks.md#comprehension-drills):
a closed-book task with no repository, whose instruction adds that the answer goes to
`answer.txt`, whose `tests/test.sh` rewards 1 exactly when that file matches the reference
(whitespace runs count as one space) and whose reference is kept under `tests/`. A drill is
not imported back.

*Trials* are read from `result.json` (or `results.json`) and `agent/trajectory.json`
(ATIF; a trial without one is refused, never reconstructed). The experience's verdict
is Harbor's: pass at reward 1, fail at any other reward, none for a trial with an
exception or no reward. `verify` leaves such experiences alone: they carry no patch.

*Episodes* run a directory of practice task folders (or one folder) through sven's agent,
for experience to learn from:

```sh
splinter tasks harbor episodes ~/practice --policy policy:default --attempts 8 --jobs 4 \
    --deadline 10m --max-tool-calls 40 --out episodes.jsonl
```

The seed every attempt's draws derive from is drawn and printed unless `--seed` gives it;
the run records it. Every folder is read, checked for exam material and imported before any attempt starts;
a sealed or malformed folder refuses the whole collection by name, and a folder imported
before is found again by the digest of its files. Each attempt is a workspace episode as
`solve` runs one: a fresh copy of the starting files, the agent's shell confined to it,
the policy asked, with every request, to sample at temperature 1 with nothing cut away
and a seed of the attempt's own, whatever serves it (the record's `seeded`), and asked for
its policy trace, which sven keeps in the trajectory's step metrics:
the prompt and sampled token ids and each sampled token's log-probability, as the
provider returned them. Only after the agent has finished is its change graded, in
another fresh copy with the task's tests laid over it; the exposure audit fails an
attempt that reached for them.

Each attempt is stored as an experience, and a record of it (`--out` writes them one per
line, `--json` prints the report) names the task's address and its folder's digest, the
attempt and its seed, the policy, a digest of the configuration (deadline, caps, episode
protocol, coding prompt, sampling), the experience, how the agent's run ended
(`conclusion`: `success`, `timeout`, `budget_exhausted`, ...), its usage (model calls,
tool calls, tokens, wall time) and its trace (calls, how many carry ids and
log-probabilities, `provable` when all do). Its `outcome` is one of `passed` (reward 1),
`failed` (reward 0), `exposed` (reward 0), `verifier_error` or `verifier_timeout` (no
reward) and `infrastructure` (the episode could not run: no reward and no experience).
A verifier or infrastructure failure is never a reward of zero.

A rerun with the same seed, policy and settings skips every attempt that has a record
and redoes the ones the infrastructure stopped; more `--attempts` adds the missing ones.
A record is written only after its experience and verdicts are stored, so an attempt
killed before then is run again from the start. A campaign reaches the same episodes
through `episodes::SvenEpisodes`, its episode executor.
