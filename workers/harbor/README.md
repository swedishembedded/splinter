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
```

`run` exits 0 when every attempt has a report (a reward-0 trial or a model error is still a report), 1 when the worker
itself failed or an attempt has none, 2 when a specification or an argument is refused before Harbor starts.

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
before any process starts is tested. A real `claude` binary was not run; a scripted stand-in speaks its protocol.
