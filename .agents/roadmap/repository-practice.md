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
8. [ ] Tool-definition audit in sven: token cost of each tool definition, fix wrong or
   redundant ones, no additions without a measured failure.
9. [ ] Retire the duplicates in the coding-loop sample so it keeps only supervision and calls the
   shared stages.
10. [ ] Acceptance on the GPUs: plumbing on the small model, then the measured run on the 8B
    model with frozen exam families, then served through brain and used from sven.

Dependencies in brain (per-answer masking of tool-call turns, tool definitions carried into
records, public adapter manifest writer, long-context memory measurement) and in sven
(compaction marker, offered tool set, raw arguments for unparsed tool calls) land in those
repositories first.

The measurement protocol and results are in the paper
`papers/training-code-skills-with-splinter`.
