<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Learn a repository from its own tests

Use the `repository-practice` template to teach the policy a codebase. The project's own tests
are the verifier: faults are seeded into the code, and a task exists only where the
tests catch the fault every time. The agent works in a copy of the tree and is graded
by tests it never sees.

The agent runs on [sven](https://swedishembedded.com/docs/sven); the commands it
runs are bounded as described under [Confinement](#confinement). Read
[limitations](../concepts/limitations.md) before pointing it at code you do not trust.

```bash
splinter init --template repository-practice my-repo
cd my-repo
splinter validate recipes/repository-practice.yaml
splinter run recipes/repository-practice.yaml --out runs/practice
```

`init` makes `project/` (a small example project with its tests) a git
repository and names its commit in the recipe; point `repository` and `revision` at your
own. The recipe's stages, in order:

| Stage | What it does |
|---|---|
| `practised` (`task.mutate`) | seeds faults by code in the files under test of the revision's tree and keeps a task only where the project's own tests catch it every time, the change reversed makes its check pass, and the catching tests live in files of their own, which are taken out of the tree the agent is given; a family (the file a fault was seeded in) is trained on or examined, never both (`eval_percent`) |
| `train-tasks`, `held-tasks` (`tasks.pin`) | pin the training split and the held-out split |
| `attempts` (`attempt.collect`) | the policy attempts each training task as an agent in its own copy of the tree, with sven's file, search and shell tools, several times, each attempt drawing from a seed |
| `sft` (`dataset.project`) | the passing attempts become a dataset |
| `lora` (`model.train`) | a LoRA trained on them |
| `eval` (`evaluation`) | the candidate and the baseline attempt the held-out tasks, paired, and compared by the evaluation's sign test |
| `release` (`promotion`) | decides from the evaluations whether the candidate is promoted |

[Learning factory](learning-factory.md) says what a run needs on the machine and what each
operation takes. The recipe is yours to change: its models, the number of attempts, the
size of the held-out split.

A corpus of corrections can be built without solving again from an arm that
already ran: `splinter corrections --attempts <EXPERIENCE-SET> --tasks
<TASK-SET> --teacher <REF> [--max N]` continues the failed or stuck attempts
of that set from the state the student actually reached, verifies every
continuation with the task's own hidden checks and prints the id of the
experience set holding the verified ones, which a `dataset.project` stage
and a `model.train` stage then take as any other attempts.

Where no container is configured, the commands the agent works with run on
the host, bounded but not isolated, and every environment record says
`isolation: none`; naming a docker client and a digest-pinned image
(`--docker`, `--workspace-image`) runs them in a container with
no network instead. `--cargo-home` gives the commands a
cargo home to build offline from, and `--workspace-scratch` says where
trees are laid out. An attempt whose trajectory shows the hidden tests,
and an attempt the harness could not carry out, are set aside and counted in the
report; neither is in a denominator.

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
`--workspace-max-processes N` bounds the processes and threads of a
command, `--workspace-memory-gib N` the address space of each of its processes and the memory the whole command holds resident (32 GiB per process and 24 GiB for the whole command by default; a command that passes the bound is killed and reported as stopped by a limit), and `--workspace-env NAME=VALUE` gives every command a variable (for
example `PYTHONDONTWRITEBYTECODE=1`, so a Python project leaves no bytecode in the tree
the solver reads); each is part of the environment record, so a task generated without
it names another environment. The solver's shell says where it works: every command
starts in the project directory, `/tmp` is not there, and a scratch directory outside the
tree (`$TMPDIR`, also `$HOME`) keeps its files between commands and never reaches the
patch. The harness keeps its own state (sven's audit log) out of what the solver reads:
`.sven` in the tree is a link to a directory the sandbox does not let a command open. The agent's own file and search tools are
confined to the attempt's tree by sven's project root, for its delegated
children too. A command, a code call of `run_code` included, runs in a private user and pid
namespace with a `/proc` of its own, so no other process of the host is visible to it
(Landlock alone already refuses it another process's environment); a kernel that cannot
make one refuses the command (a command run unconfined is put in the namespace where the host allows it, and any command dies with the process that supervises it, so nothing it started is left running). The exposure
audit reads a command's mention of a path outside the tree by what that call does with it
and what its own result shows. Results arrive in the order the calls finished, so each is
paired to its call by the call id it names (the shell's `[exit code N]` line is not a
result, and a failure report such as a Python or Lua traceback is not something the path
returned). A read is classed by its result: `outside_workspace` (something came back)
fails the audit; `outside_workspace_denied` (the sandbox refused) and
`outside_workspace_absent` (no such path, nothing returned or only a failure report) are
recorded as `outside_probes` in the evidence and do not. A write (a redirect, `tee`, the
write tool) reads nothing and makes the file the solver's own scratch
(`outside_workspace_write`), and so does a `cd` (`outside_workspace_cd`; a change into
another attempt's tree is a read); a path inside a here-document counts only as the
argument of a call. A search of the whole filesystem (`find /`, `grep -r ... /`) fails the
audit whatever it returned (`outside_workspace_search`): hunting for another copy of the
code or the tests is behaviour that is not trained. A call that names a hidden path and
only got a refusal or "no such file" is recorded as a probe (`hidden_path_absent`); a hidden
path in what a tool returned, hidden content and the state root still fail it.

## Running a repository's attempts on a brain endpoint

An `attempt.collect` stage and the release gate attempt tasks as agents, each
attempt a conversation of many requests. With `--brain-endpoint` the policy is served by brain and up
to `--attempt-concurrency` attempts run at once against it, each with the seed it has when run one
after another, so the experiences recorded are the same ones, in the order of the tasks and attempts
(see "Serving models from a brain endpoint" in the configuration page):

```text
splinter --brain-endpoint http://127.0.0.1:8805/v1 --brain-key-file KEY --attempt-concurrency 48 \
         --policy-base Qwen/Qwen3.8-27B-FP8 run recipes/practice.yaml --out runs/practice
```

The gate's candidate must be served too: start the endpoint with `--adapter-manifest FILE` and give
splinter the same file as `--brain-adapter-manifest FILE`; the gate offers each adapter there as it
grades it.

## How a solver is run

A solver of a repository works under the coding-agent prompt (look, reproduce, change,
verify, never repeat a call, stop when the tests pass) unless the model it is asked for brings
a prompt of its own (a policy trained to be a person). `coding-agent-1` is the default.
`--coding-prompt coding-agent-2` selects the second: it says the tests that name the symptoms
are hidden and the visible suite proves nothing, asks for a direct call of the function the
report names (`python3 -c`, `lua -e`) instead of a test run, forbids grepping the report's
words and searching outside the repository, and says to change a crashing script instead of
running it again. The flag applies to every repository solve of the run, the teacher's included. Each attempt records what it was held
to in its trajectory's `solve_profile`: the prompt's id and digest, the longest shell call
(1800 s), sven's note on a repeated result (on), the repetition bound and the tool-call cap,
and how the model was asked to sample. A served model is asked to sample with `--temperature`,
`--top-p`, `--top-k` and `--sampling-seed` on every request (`--thinking` lets it reason, and
`--thinking-budget N` bounds the thought of each response: the server closes it there and the
model goes on to its answer),
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
shown; those are the hidden test's code. The test files that catch a fault are taken out of the
solver's tree whole. When tests that stay import one of them (a base class every test of the
suite shares), that file stays and only the failing cases are cut out of it (for `unittest`, by
the parser's line ranges of each case with its decorators and the comments above it); the
hidden overlay puts the whole file back at verification, the file is protected from patches
and the audit counts only the removed text as hidden. A candidate fault is refused, and
counted, when the tests that stay still cannot all run in the tree
(`hidden_imported_by_tree`: a runner whose source is not cut, a case that is not a method of
its module) or when a short symptom cannot be rendered this way (`symptom_not_actionable`). The values shown are the one disclosure a task makes of its
hidden tests, so the exposure audit counts the request as text the solver may see.

The request is never enriched to make a task solvable: a title that does not say what a fault
broke stays as it is, and the task is refused instead. A fault can break a contract no title
states (the fix makes a function return a value again, or raise an error again, and no failing
test's title speaks of a result or an error), so the code already does everything the titles say
and the request leads nowhere. Such a candidate is refused and counted (`unstated_contract`).
The rule reads the request and the reference change only (Python source; the two classes are a
restored non-boolean `return <value>` and a restored `raise <Error>`; a title naming the result
or the error, or the exception, satisfies it), so `splinter tasks check-requests TASKSET-ID`
applies it to a task set that already exists: it writes the tasks that pass, unchanged and with
the same ids, as a task set of their own and reports how many were dropped for each class.

## Generations of a coding policy

A generation of a coding policy is one run of the `repository-practice` recipe whose champion is the previous
generation's model. The learning factory runs a series of them as forked recipe runs that name the previous
generation's model as their `champion:`; see `docs/guides/learning-factory.md`.

## Harder tasks: several faults at once

A solver that fixes nearly every single-fault task of a repository needs harder tasks of the
same kind to be measured against. `splinter tasks combine <TASKSET> --faults 3 --per-family 6`
makes them from the admitted tasks of a snapshot: each seeds faults in three different
files at once, is graded by all of the hidden tests that catch them, and is admitted only after
its tree fails the check, its reference change (the three reverse patches together) passes it on
two runs, and no one fault's reverse patch passes it alone. Because the tasks are made by the
same generator as the single-fault ones, the loop and an external benchmark (`tasks harbor
export`) use identical tasks. A task with a fault in a file the exam reserved is an exam task
and stays out of every training set, and the sets it writes (all, exam, training) are in the
report. [Tasks reference](../reference/tasks.md#combining-faults) has the rules and the refusal
reasons.

## Comprehension drills

A policy that fixes most repair tasks of a repository and fails some reliably often misreads
part of the code (it finds the fix and over-edits a neighbouring condition). `splinter tasks
drills REPO --rev REV` makes prerequisite tasks for that: each shows a short program that imports
the repository's module and calls one function on concrete inputs, and asks what it prints.

```bash
splinter tasks drills ./shop --rev v1.4 --per-function 3
splinter tasks drills ./shop --rev v1.4 --file shop/pricing.py
splinter tasks drills ./shop --rev v1.4 --for-tasks blake3:1ae0f1...
```

The answer is what the pinned interpreter prints against the clean snapshot of the revision, so
no model writes a task or its reference, and the inputs come from the repository's own tests
(arguments only, never an assertion or an expected value). Drills are tagged with the file and
function they exercise; the ones of a file the exam reserved are exam tasks and are
never trained on. `--for-tasks` drills exactly the functions the reference changes of a set of
repair tasks touch. [Tasks reference](../reference/tasks.md#comprehension-drills) has the rules,
the refusal reasons and the export.

## Related

- Check a plan of which repositories a lineage trains on and which are sealed as the final
  exam with `splinter experiment check`; the frozen repositories this project ships are
  described in [Task families](../reference/tasks.md#repository-families).
- Import tasks written for the Harbor benchmark format, or export these as Harbor tasks:
  [Harbor tasks and trials](harbor-tasks.md).
- Try other agent loops on the same tasks: [Improve a harness](improving-a-harness.md).
