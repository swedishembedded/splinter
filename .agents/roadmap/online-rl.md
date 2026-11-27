<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Online RL over any environment's task set

Scope: `train.online` over any bench environment's task set run as proper multi-turn agent episodes in each task's own environment, with sound version handling, observable training health, a frontier task selector, and a watchdog that rolls training back when it moves the policy the wrong way.

Everything here is environment-generic: tasks, tools, verifiers and signals are read from the task folders and
the records, never from an environment's name. Bench defines the environments, tasks and graders; Splinter runs the
online loop over a task set. Vim is the first validation example only. The controller core, its ledger and the learner
seam are the async-grpo ledger's (`async-grpo.md`); this ledger is what was built on them in lane rl-online
(2026-11-27). Every new behaviour is a recipe parameter whose default is the earlier behaviour, absent from a stage's
identity when it is the default.

## Done

- [x] **Tools a task declares, in online rollouts (A0).** `train.online` `agent: { agent: sven, environment, args,
      files, system_prompt, environment_variables, budget, timeout_policy }` runs each rollout on the execution worker
      as one attempt of its task, exactly as `attempt.collect` runs it (`run::online_worker`): the agent gets the
      tools, configuration and sandbox the task declares (its `[[environment.mcp_servers]]` and the agent
      configuration its image installs). The run sets each request's temperature, `--top-p 1`, a seed of the rollout's
      own and `--logprobs`/`--top-logprobs`; sven's own request log is filed on the trajectory and projected by the one
      projection every executor shares (`online::rollout::rollout_of`). The learner's server is reached through sven's
      `brain` driver (`ModelBinding::driver`, knowledge 250). A restarted rollout reads back what an earlier attempt
      finished (`ExecutionBackend::reconcile`). The in-process rollouts refuse a task that declares MCP servers (it
      would run without its tools). The tool schema is in the stage identity through the task folders' digests and the
      environment's. Tests: `make test/online/tools` (a fixture task whose only tool is an MCP server: offered exactly
      that tool, every call answered in the task's image, rollouts exact, graded by its `test.sh`, groups trained) and
      `make test/online/tools/tasks TASKS=<dir>` (any task set that declares tools; run on the bench Vim task
      `manual-0007c64a-update-menu-docs`: every offered tool was the task's `vim-*`, every call answered, four rollouts
      graded 0/1). No other bench environment declares tools today; the fixture is the second, non-Vim case.
      Cancellation: the worker cannot stop an attempt it runs, so a cancelled or rolled-back run waits for running
      attempts and a resumed run reads them back.
- [x] **Truncation (A1).** `splinter_core::factory::truncation`: one classifier of a limit that ended a run (context,
      output length, output budget, turns, deadline) for runs in this process and trials on the worker; the offline
      solve uses it too. Decision: a truncated rollout is a scored failure (reward 0) recorded with its cause, never
      retried. Its prefix: every completed call is a sample of the policy and stays training material (reward 0);
      a call cut at its own token limit is `truncated` in brain's record contract and brain trains none of such a
      rollout, so it is left out of the group's trainable members as before. A conversation the server refused for its
      context keeps no trajectory on the in-process path (sven returns none on a provider error), so it is a failure
      that trains nothing. Report: `rollouts.truncations` by cause. Knowledge 249.
- [x] **Version pinning and epochs (A2).** `pin_versions` (off): only rollouts one version decoded first to last are
      trained; a straddling rollout is scored and counted (`rollouts.version_mixed`), a group left without differing
      trainable rewards filtered `version-mixed`. Size `publish_every` so a version outlives an episode. Brain serves
      only the newest release of a published id (`PolicyBank::publish/identity`), so pinning an episode to the version
      it started on would need brain to keep earlier releases addressable (`id@vN`); not built. Every group records its
      epoch (ledger column added in place, read as 0 when absent); `rolled_back` is a terminal state; a controller in a
      new epoch rolls back every earlier-epoch group not yet terminal and never sends it, and the selector observes only
      its own epoch. A resumed controller no longer counts another stage's groups of the same ledger.
- [x] **groups.jsonl (A3).** `online::group_log`: one synced line each time a group is sent, applied, committed,
      filtered, stale or rolled back (epoch, learner version, span, rewards, every rollout's outcome and cause); a torn
      last line is dropped and missing transitions recovered from the ledger on open, each transition once through
      kills. The run's own records (rollback audits) go to the same log. In `work/<stage>/` while running, in the
      product after.
- [x] **Training health (A4).** `online::health`: typed step figures from brain's frames (KL, entropy, clip fractions,
      ratio min/max/mean, grad norm, staleness, wait, queue) and the trainer's skipped steps (`StreamEvent::Applied`
      optional `skipped`); per policy version (a rollout's oldest version) the reward and the shares of zero-variance
      groups, truncations, infrastructure failures, false completions and repeated calls. Definitions, from records
      alone: false completion = a scored rollout whose agent ended the run itself (its last agent step calls no tool),
      no limit ending it, that the verifier failed; repeated call = a tool call with the same tool and arguments as an
      earlier call of the rollout (`Behaviour`, recorded on every rollout outcome). `report.json` `health`;
      `splinter observe` prints the per-version lines for every online stage, live from the group log.
- [x] **Frontier selector (A5).** `selector: frontier` (default `round_robin`): draws in proportion to the expected
      reward variance under a decayed Beta posterior (decay 0.75 per group), a floor of 0.1 of the largest weight for a
      never-passed task, retirement by the campaign rule (`RETIRE_AT` 0.9 on 8 rollouts); the run ends when every task
      retired. Draws derive from the seed and the group's place; the posterior is replayed from the ledger on resume (own
      epoch only). Knob variable `online.zero_variance_share` for `splinter probe`. Scripted run (three tasks: always,
      never, sometimes passed; 60 groups of 4): round robin 0.67 of groups zero-variance, frontier 0.27.
- [x] **Watchdog (A6).** `online::watchdog` (off): invariants with registered bounds pause intake on a new breach and
      ask for an evaluation (never roll back); CUSUM drift detectors on change from the promoted versions' mean (entropy
      falling; false completions, repeated calls rising; reward falling) ask for an evaluation at once; decisions only by
      a paired comparison of frozen copies (adapters by digest, `Trainer::published_adapter`, served by
      `Trainer::serve`) of the newest version and the anchor on a fresh draw of the `monitor` input (each task at most
      `max_uses` per epoch), test martingales on discordant pairs (regression at `1/alpha`, non-inferiority at
      `1/beta`, comparison k held to half of what is left), the sealed `confirm` input read only before a promotion.
      Rollback = new epoch (`run::online_train`): running rollouts stopped, untrained groups of the epoch rolled back, a
      fresh learner from the anchor's adapter under a key of its own with `params.lr` times `cooldown` per rollback,
      halt after `max_rollbacks`, `epochs.json` kept so a stopped controller resumes in its epoch; audit (trigger,
      evidence, candidate, anchor, alarms, groups discarded) in `groups.jsonl` and `report.json` `watchdog.audits`.
      Tests: the layers against scripted telemetry (`online::watchdog::tests`), the false-rollback rate under no
      difference over 2000 seeded comparisons (at most alpha spent), and a driven run that regresses from version 3,
      rolls back, and trains only its next epoch's groups (`it::online::watchdog`).

- [x] **Validation (A7 a, b).** GPU-free: the tests named above, plus the false-alarm check end to end
      (`a_watched_run_whose_policy_does_not_regress_is_evaluated_and_never_rolled_back`, splinter-pipelines
      `it::online_factory`: training, monitor and confirm pins, a scripted streaming job and policy; every published
      version evaluated against the anchor through frozen copies fetched by digest and served; no audit). The CPU toy
      stack with every default (`make test/online`, brain 900a65bc1 built on the CPU, port outside the shared ones)
      passed: 12 groups started, 8 committed, 4 filtered, versions 1..6, the controller and brain each killed and
      resumed, no group trained twice; `splinter observe` printed its per-version health from the run. Two defects of
      the stack's own checks were fixed on the way: it read a published version as a list (`p[0]`, a `KeyError` on
      every run since versions became objects), and it kept an earlier invocation's brain event logs in a reused work
      directory, so brain's commits of another run failed "every group brain committed is an experience".

- [x] **Episode pinning (round 2).** brain (15cad2670, b1536a552, c824e324d): a stream's `retain_versions` N also
      serves each published version under `<publish_adapter>-v<version>` with its version and lineage (the N newest;
      older withdrawn, requests already accepted finish), keeps each in the artifact store (`artifact_digest` in the
      published frame), and `POST /v1/jobs/{id}/hold` keeps a referenced version served past the retention. Splinter:
      `pin_episodes` asks retention (max_staleness + 2, at most 16) and starts each rollout on the newest version's id
      (in-process through the stage's policy resolver, on the worker as the attempt's model). The watchdog evaluates
      versions by those ids (an earlier copy of a published adapter served by the trainer would have replaced the
      training's own release, which carries the same card id), holds its anchor and the candidate under evaluation,
      and a rollback starts the next run from the anchor's stored adapter. Tests: brain `stream::a_stream_retaining_*`
      (retention, hold, store), splinter `a_run_that_pins_its_episodes_*` (driver),
      `a_rollout_pinned_to_a_version_asks_it_for_every_call_of_its_episode` (two calls across a publication, span 3..3).
      CPU stack `make test/online/pinned` (toy, publish_every 1): 12 groups, 48 rollouts, every span one version, 8
      versions each under its own id. The toy's episodes are one call, so multi-turn pinning is shown by the in-process
      test, not the stack.
- [x] **Cancel running worker attempts.** `ExecutionBackend::execute_cancellable`: the worker program and everything it
      started are killed (process group and descendants); a cancelled or rolled-back online run kills its attempts.
      Test `a_cancelled_batch_kills_the_worker_and_everything_it_started`.
- [x] **Selector ageing by version.** `TaskSelector::observe_at`: frontier evidence weighs 0.9 per policy version
      published since it was counted.
- [x] **Merged sessions online.** `sessions: merged` writes an exact, append-only session as one `SessionRecord`
      (`params.max_block` required, `pack_answers` set; `params.loss_agg` passes through), else one record per call
      counted by `Unmergeable::reason()` in `rollouts.sessions`. CPU stack `make test/online/merged` (toy, 12 groups):
      13 sessions merged, 35 stayed per call (not exact: replies the output bound cut), 1 group of merged sessions
      trained. A per-call record longer than `max_block` is refused by brain whole, as before: the block must hold the
      longest call, so `longer_than_block` falls back usefully only for multi-call sessions.
- [x] **The GPU bench run, prepared.** `make test/online/bench` (scripts/online/bench-task-set.sh): 12 training, 8
      monitor and 4 confirm tasks, disjoint, from a bench environment's training folders (a path naming an exam is
      refused; sealed folders are refused on import), train.online with `agent: sven` and the environment file,
      `selector: frontier`, `pin_episodes`, the watchdog; prints groups, rollouts, versions, per-version health and
      the watchdog's decisions. `make test/online/bench/prepare` validated against the Vim manual set
      (`splinter validate`: 24 tasks).

## Measured

- Loss weighting of a long episode: with one record per model call, brain's stream objective trains each call as a
  sequence, so "under every aggregation an episode of many calls weighs by its calls" (brain `stream_pg.rs` module
  docs at 900a65bc1): a 60-call failed episode carries its negative advantage on 60 sequences against one for a
  one-call success. Session merging (one sequence per append-only session) landed upstream meanwhile (brain 900a65bc1,
  splinter dbcc8fe58); under the per-sequence mean every episode then weighs alike.

## Open

- [ ] A7 (c): a CPU fault-injection run on the real stack (a reward-sign flip or poisoned grader on a task subset
      after warm-up, past brain's step guard) that must go alarm -> evaluation -> rollback -> recovery, and a clean run
      that must not roll back. The pieces are tested GPU-free (above); the toy stack's tasks (`tasks/toy-answers`) need
      monitor and confirm collections and a grader that can be poisoned by a run parameter. Command once built:
      `make test/online` with a recipe carrying `watchdog` and `monitor`/`confirm` inputs.
- [ ] The end-to-end check as "a bench environment's task set + splinter train.online" on a served model: a recipe
      `tasks.pin` of the bench task set -> `train.online` with `agent: { agent: sven, environment:
      environments/<env>/environment.yaml }` (bench's environment file), run with `splinter run <recipe> --brain-endpoint
      ... --brain-artifact-store ... --brain-adapter-dir ...`. Validated with scripted models only.
- [ ] Watchdog evaluations on capacity of their own. Not built: a version exists only on the learner's server (its
      bank, under its version id); another server could serve it only from the stored adapter, whose card names the
      published id, and two versions of one card id cannot be served side by side there. Needs brain to serve a stored
      adapter under an id the caller names (an additive `--serve-adapters` rule or an offer route). The evaluations
      run at most `max_concurrent_attempts` rollouts of a version at once on the learner's server meanwhile.
