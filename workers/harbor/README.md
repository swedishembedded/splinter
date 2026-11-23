<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# splinter-harbor-worker

The execution backend of the learning factory: it runs `splinter.execution/1` specifications as one native
Harbor job (`harbor==0.20.0`, pinned exactly) and reports a
`splinter.attempt/1` report for every attempt. It never grades; the controller's one classifier reads the evidence.

```
make worker/install        # workers/harbor/.venv with the pinned worker and its test dependencies
make test/worker           # the suite: proot sandboxes, stub models, no network (needs sven on PATH)

splinter-harbor-worker run --specs SPECS.jsonl --tasks TASK_ROOT --out OUT_DIR \
    [--concurrency N] [--files FILES_ROOT] [--image-cache DIR]
splinter-harbor-worker reconcile --out OUT_DIR
splinter-harbor-worker play --spec SPEC.json --tasks TASK_ROOT --out OUT_DIR \
    [--files FILES_ROOT] [--image-cache DIR] [--continue TRIAL_DIR]
```

`run` exits 0 when every attempt has a report (a reward-0 trial or a model error is still a report), 1 when the worker
itself failed or an attempt has none, 2 when a specification or an argument is refused before Harbor starts.

## Play: a person drives the agent

`play` runs exactly one ExecutionSpec (one JSON object in SPEC.json) as one Harbor trial under `OUT_DIR/harbor/job/<trial>`
with the same staging, sandbox, gateway and agent plumbing as `run`, except that the agent is started interactively on a
terminal attached to yours: you type into the agent's own interface (Sven's TUI, `sven --interactive`; Claude Code's) and
end the session by quitting it. The terminal is raw while the agent runs, window resizes are forwarded, and it is restored
however the session ends. Then the verifier runs and the worker writes the AttemptReport and `events.jsonl` exactly as
`run` does. `OUT_DIR` must be new (a person's session is never resumed), and the agent's time limit (the task's, or the
specification's `timeout_seconds`) bounds the session. Harbor's own progress display and log lines are kept off your screen
(they are in the trial's logs).

```
splinter-harbor-worker play --spec spec.json --tasks tasks --out out/play-1
splinter-harbor-worker play --spec spec.json --tasks tasks --out out/play-2 --continue out/play-1/harbor/job/<trial>
```

* **Refused with exit 2:** standard input or output is not a TTY; the agent is `oracle` or `nop` (only `sven` and
  `claude-code` have an interface to play); the provider is `docker` (only `userns` and `proot` give the agent a terminal
  of the sandbox's own); the usual specification refusals. It works under `proot`, where there are no user namespaces and
  no docker.
* **Marking.** A reader tells a guided trial from an automatic one by `config.agent.kwargs.human_guided: true` in the
  trial's `result.json`, which is `evidence.result.config.agent.kwargs.human_guided` in the report: the field Splinter's
  Harbor ingestion reads. Sven also marks each turn you type in its trajectory (`extra.sven.origin: human`), and the
  worker writes `agent/guidance.json` (`splinter.guidance/1`): who sat there, how many turns were typed and after how many
  agent steps. Claude Code leaves no trajectory, so its guidance record has `trace: false` and the configuration is the marker.
* **`--continue TRIAL_DIR`** goes on with the conversation of an earlier played Sven trial (`TRIAL_DIR` is its
  `OUT_DIR/harbor/job/<trial>`, beside that `OUT_DIR/specs.jsonl`; the specification must be for the same slot). The new
  trial is the next attempt of that slot: Sven is started on a copy of the old trajectory (`--trace`) and its trajectory is
  the whole conversation, the first message is not sent again, and `guidance.json` records `continued_from` (trial,
  attempt, trajectory digest). The old trial is only read. The workspace starts fresh from the task: the conversation
  continues, the files the earlier session changed do not come with it. Sven drops the human marks of the turns it reloads;
  the worker restores them (`continued_from.marks_restored`).

## Output

```
OUT_DIR/harbor/job/            the Harbor job (harbor view OUT_DIR/harbor)
OUT_DIR/attempts/<pass>/<task id, / as __>/<slot>-<attempt>.json     one AttemptReport per attempt
OUT_DIR/events.jsonl           {at, kind: started|attempt_finished|finished|error, slot, attempt, detail}, flushed
OUT_DIR/specs.jsonl            the specifications, kept so reconcile needs nothing else
OUT_DIR/staged/                the staged tasks (network and image written into task.toml) and agent binaries
```

Every path in a report is relative to `OUT_DIR`. A report is a pure function of its trial directory, so `reconcile`
rewrites the missing ones byte for byte after a crash, runs nothing, and names the attempts with no finished trial.
Running `run` again on the same `OUT_DIR` resumes the Harbor job: a trial that finished is not run again.

## What a specification is mapped to

* **Paths.** `task_dir` is relative to `TASK_ROOT`; agent `files`, the `system_prompt` and mount `source`s are relative to
  `FILES_ROOT` (default `TASK_ROOT`). A path that leaves its root is refused; a `sha256:` digest is verified.
* **Agents.** `sven` and `claude-code` (the executable named by `SPLINTER_SVEN_BIN` / `SPLINTER_CLAUDE_BIN`, else on
  `PATH`, staged read-only at `/splinter/bin`), `oracle` and `nop` (Harbor's own). `agent.args` are an argv, never a
  shell line; an argument equal to a listed file's path (or `--option=path`) names its uploaded copy under
  `/splinter/files`. `agent.env` is the only environment the agent gets besides its isolated HOME: `from_env` values are
  read from the worker's environment when the trial starts and never written to a file. A literal under a credential-like
  name is refused. The attempt's seed is given as `SPLINTER_SEED`, not as a flag (Sven refuses a sampling seed a model
  cannot honour; a recipe that wants one passes it in `args`).
* **Model.** With `model`, the worker runs one gateway for the job. The agent gets the trial's HMAC token as its only
  credential; the gateway forwards to `model.url` with the token read from the worker's `model.token_env` variable,
  rewrites the model, sends `accept-encoding: identity` and logs each request to `<trial>/gateway/requests.jsonl`
  (`evidence.requests`). Error text is stripped of control characters. Without `model` the agent uses its own provider
  configuration (from `args` and `files`).
* **Sandboxes.** `docker` is Harbor's own. `userns` and `proot` run the task's own image (`docker_image` or
  `environment/Dockerfile`) from the image cache; `proot` works where user namespaces and docker do not
  (`PROOT_NO_SECCOMP=1`, with a translation self-check before every trial).
* **Network.** `open`, `none` and `model-gateway-only` become the staged `task.toml`'s network mode and, for the
  gateway, Harbor's agent-phase allowlist (and the gateway port on `userns`). `proot` can enforce only `open`; a
  combination a provider cannot enforce is refused. `docker` needs `SPLINTER_GATEWAY_HOST`, an address of this host the
  containers can reach. `proot` and `userns` cannot enforce `cpus` or `memory_mib`; the start event lists them.
* **Mounts.** Bind mounts at the target given; read-only where the provider can make them (`proot` refuses a read-only
  mount).

## Not verified here

`userns` needs unprivileged user namespaces and `docker` a daemon; this container has neither, so only what they decide
before any process starts is tested. A real `claude` binary was not run; a scripted stand-in speaks its protocol, also in `play`. `play` was run here only
on `proot` (real Sven's TUI and a scripted agent on a pseudo-terminal); the `userns` terminal path is the one `bench task play`
ran and was ported, not run.
