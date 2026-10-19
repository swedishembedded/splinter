# Learning a repository from executable tasks

Outcome: at any time a person runs one verb on a git repository. Splinter derives
executable tasks from it, drops what it already holds (same snapshot, near-duplicates of
earlier tasks, earlier generations), solves them with sven agents, keeps what hidden checks
confirm, trains a new adapter on the clean body of verified work, gates it against the
champion, and releases it. The person then points sven at the released adapter served by
brain and gets an agent that works in that codebase: tool calls, edits, test runs, and the
conventions only the people working on the code know.

The same machinery is not code-specific: a workspace is a tree of files, a check is a command
over it. Documents, tools and persons keep sharing the experience store, curriculum, release
gate and lineage.

## Requirements the end state adds

- **Repeatable and cumulative.** A second run on a new revision, or on another repository,
  adds only tasks that are new (snapshot digest, family and dedup key); earlier tasks stay
  valid evidence of their own snapshot; training mixes in replay so earlier repositories are
  not lost. Retention is measured, not assumed.
- **Knowledge only the maintainer has.** Tasks are not only mutations. Sources include the
  repository's history (a fix and the test that came with it, reversed), and what the
  maintainer says in sessions (conventions, "never do X here"), which already travels through
  the claim intake and becomes tasks with verifiers. A task that no check can grade is not
  admitted as training evidence.
- **Small tool surface.** The agent keeps sven's existing tools. No tool is added unless a
  measured failure needs it. The tool block is part of every training record and of every
  served request, so its token cost is measured per tool and descriptions are shortened
  where they carry no behaviour. A defect in a tool definition (wrong schema, ambiguous
  argument, output that wastes context) is fixed in sven, in its own commit.

## Work, in order

1. [x] Workspace environment and command runs in the sandbox; core names for the environment,
   check and task kinds.
2. [x] Workspace verifier (fail-to-pass, pass-to-pass, protected and hidden paths) and an
   exposure audit between what the solver saw and what the verifier holds.
3. [x] Solving a task in a workspace through the existing worker; patch read back; follow-up
   turns from visible checks only.
4. [x] Training view that keeps the tool definitions the agent was offered, treats follow-up
   turns as context, and excludes compacted, unparsed or over-length trajectories with
   counted reasons. (The compaction and unparsed-argument exclusions read markers sven does
   not write yet; see the dependencies below.)
5. [~] Workspace task generation: profile and baseline qualification (twice, flaky refused),
   mutation-seeded faults, import of fixture families; admission reuses dedup across earlier
   generations. Remaining: faults derived from the repository's history (a fix and the test
   that came with it, reversed).
6. [x] The pipeline that composes them (tasks, reserve families before rollouts, solve,
   verify, frontier, dataset, train, release) as one recorded run (`practice`), with an SDK
   method and a CLI verb.
7. [x] Release for agentic suites: the gate's four checks over the exam `practice` reserved
   (paired outcomes of the champion and the candidate attempting its tasks as agents, counted by
   family; exposed attempts and attempts the harness could not carry out set aside and counted;
   each earlier release held to its own exam; the frozen anchor suite, decided on what it holds
   when it holds no coding-agent task and saying so), the serve check by agreement of verdicts
   with the in-process run, and brain's adapter manifest rewritten on every alias move so
   `brain serve --adapter-manifest` follows the alias. `practice` ends in the gate and says why
   a blocked candidate was not released.
8. [x] Tool-definition audit in sven: token cost of each tool definition measured and guarded by a
   footprint test; redundant wording removed; no tool added.
9. [x] Retire the duplicates in the coding-loop sample so it keeps only supervision and calls the
   shared stages: its dataset projector, model registry and promotion rule, training wrapper,
   private process runner and git checkout logic are deleted; a run captures the baseline as a
   snapshot, each attempt is `solve_in_workspace`, a candidate is judged by the workspace grading
   and audited by the exposure audit, and the sample depends on the SDK alone.
11. [x] A teacher for the tasks the policy never solves: `practice --teacher reference` replays a
    task's own reference change (the inverse of the seeded fault) as an agent's actions through
    sven's real tools (read the file, edit it by the patch's hunks), `--teacher remote:|local:`
    has a stronger model solve the task as an agent; both are graded by the same workspace
    verifier and exposure audit as the policy's attempts, never grade themselves, and are asked
    only about tasks outside the exam. The demonstration names the file without searching for it:
    it teaches the edit once the place is known, not the localisation (open: demonstrations that
    include the search an agent would have made, from a stronger model's verified trajectories).
10. [ ] Acceptance on the GPUs: plumbing on the small model, then the measured run on the 8B
    model with frozen exam families, then served through brain and used from sven.

## Not done yet (everything below is needed before this is finished)

Order matters; each line is one self-contained piece of work.

### Before the acceptance runs

- Re-point the frozen copies of sven and brain that Splinter builds against at the current
  main of each, rebuild the whole Splinter workspace, run `make lock` and the file-size and
  architecture gates.

### Acceptance on the GPUs (work item 10)

1. Fixtures: the twelve seeded-bug families of the loop sample imported as workspace tasks,
   plus mutation-seeded tasks from one small Python repository kept under `tasks/` as a
   frozen family (own code).
2. Plumbing run on the 0.6B model: `practice` end to end, every stage recorded, dataset
   passes brain's check, overfit test (30 steps on 4 records, held-out loss on them falls).
   It proves the plumbing, not learning; expect few or no verified trajectories from 0.6B,
   so the overfit records may come from 8B runs.
3. Measured run on the 8B model: pass@k on training families, SFT from verified
   trajectories, exam families reserved before any rollout, base and candidate at the same
   seeds, the gate's sign test over family clusters. Size from the measured training cost
   (windows near 4096 tokens, a few hundred records per epoch; longer windows need an int8
   base or are refused by the placer). Report pass@1, tool-call validity, exposure count,
   environment versus model failures, wall time. A negative result is reported as such.
4. Serve the released adapter with `brain serve --openai --adapter-manifest`, mixed with
   base requests on the one resident base, and solve one exam task from sven.
5. Second round: a later revision of the same repository and a second repository; measure
   retention on the first (non-inferiority margin in the paper protocol). The eight families'
   schedule (three generations, retention, two sealed families) is `tasks/e3-plan.toml`,
   validated by `splinter experiment check` and enforced per run by `practice --plan
   --generation`. Not yet enforced: the sealed families' tree ids are not kept in the state
   root, so a run without `--plan` can still read one.
6. Put the measured numbers into `papers/splinter-brain-sven-flywheel/data/results.csv`
   (nothing before it is measured), rebuild the figures, update the audit statuses, and
   settle with the maintainer whether a paper may hold results of a protocol still being run.

### Task generation still missing

- Faults derived from the repository's history (a fix and the test that came with it,
  reversed), with the same qualification as mutations.
- Test-selection for large suites (impact analysis) so a task grades only relevant tests.
- Mutation seeding covers Python unittest suites and `cargo test --offline` (read per test
  from libtest; doc-tests are not read). Missing: a pre-built `target` per snapshot (every
  attempt rebuilds the crate), the offline dependency path for crates with dependencies, and
  other test runners.
- Model-written faults and issue text that states the symptom without the fix, validated
  composites of several faults, and an oracle-strength classification of each task.
- Maintainer knowledge as tasks: conventions and "never do X here" from sessions, taken
  through the claim intake and given a verifier; a task nothing can grade is not admitted.
- Non-code workspaces (documents, configuration trees) as a second user of the same
  environment, to show the machinery is not code-specific.

### Isolation

- A container-backed workspace run (needs a reachable container runtime): until then every
  run has isolation `none` and the exposure audit detects but does not prevent. Needs the
  live container tests (currently ignored) to run, and the Rust offline path above.
- Two sven tools outside the coding set stay unguarded (`web_fetch` private-address filter,
  debugger path confinement); they are withheld from workers, so only a person using them
  is exposed.

### Dependencies in sven and brain that are still open

- sven: compaction and raw-argument markers are written by the SDK trajectory but the
  headless CI runner trajectory still shows `{}` for unparsed arguments; consumers of the
  served-model-changed event in the TUI and ACP frontends; `make check` formatting fails on
  three files outside this work.
- brain: per-record token lengths from the SDK so `overlong_records` is exact instead of an
  estimate; an unused import in the SDK decision module; two violations reported by the
  kernel-selection gate; D-Bus coverage and a sample note for multi-adapter serving; adapters
  for the qwen35 family are not servable; the training device is not selectable from
  Splinter's `FineTune` (deferred until a measurement shows it matters); no model between
  0.6B and 8B is on disk (a 1.7B or 4B would make rollouts and training cheaper).

### Documentation owed when the above lands

- Close F-021 in `agent-loop.md` only when a run on real weights trains a multi-step tool
  conversation (the specs use a test double).
- `splinter.md` section 5: mark the file-tools-in-a-directory and ATIF-as-chat items
  resolved, with the evidence.
- A knowledge entry for every defect found during the acceptance runs, with its number.

Dependencies in brain (per-answer masking of tool-call turns, tool definitions carried into
records, public adapter manifest writer, long-context memory measurement) and in sven
(compaction marker, offered tool set, raw arguments for unparsed tool calls) land in those
repositories first.

The measurement protocol and results are in the paper
`papers/splinter-brain-sven-flywheel` (its coding-task protocol and registration `prereg/coding-tasks.md`).
