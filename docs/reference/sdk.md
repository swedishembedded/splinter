<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Rust SDK

Everything the command line does, a program can do through one dependency, the `splinter-sdk` crate.
The command line and every program under `samples/` depend on the SDK and on nothing else of Splinter's;
if they need more, the SDK is missing something.

```rust
use splinter_sdk::learn::LearnRequest;
use splinter_sdk::Splinter;

let splinter = Splinter::from_env()?;
let learned = splinter.learn(&LearnRequest {
    sources: vec!["./manual.md".into()],
    goal: Some("the console and power limits".into()),
    ..LearnRequest::default()
})?;
```

## The handle

A `Splinter` owns what a process shares: the runtime with its loaded models, the trainer, and where
progress is reported. It hands each command a context of its own, so a policy alias that moves between two
commands is seen by the second. Every command that writes state runs as a recorded run (`Splinter::run` is the
one place that wrapping lives), readable and cancellable from any process. `Splinter::from_env` reads the
configuration from the environment as described in [configuration](../getting-started/configuration.md).

## Jobs

`Splinter::jobs()` returns a `Jobs` handle for work that runs outside the
process: `submit`, `list`, `reconcile`, `retry`, `show` and `cancel`. The
launcher is the local one, the controller's name and the attempt root come from
the configuration (`Config::jobs`), and a submission is idempotent by the job
specification's digest: resubmitting the same specification names the same job
and starts nothing twice. What each call returns is the durable job record
(`store::jobs::JobRecord`), not a process handle.

## What a caller works with

- **The commands**, one module each, with their requests and reports: `learn`, `practice`, `ask`, `sources`,
  `sessions`, `claims`, `absorb`, `tasks`, `timeline`, `solving`, `verify`, `critique`, `datasets`, `train`,
  `release`, `eval`, `lineage`, `status`, `doctor`, `state`, `router` and the rest. `absorb::consolidate` takes a
  generation of a policy's claims into the weights.
- **What a command works with:** `Config`, `Context`, `Runtime`, the roles (`roles`), the recorded run (`runs`) and the pipeline
  engine (`pipeline`) a new pipeline is defined on.
- **The layers a caller reaches into**, each under one name: `vocabulary` (what Splinter talks about), `store` (durable
  state), `measure` (verifiers, the release gate), `data` (training data from experience), `knowledge` (intake and
  task generation), `agent` (the sven adapter), `model` (the brain adapter) and `sandbox`.

The crate documents every public item (`#![warn(missing_docs)]`); build the reference with `cargo doc -p splinter-sdk
--no-deps`. The samples under `samples/` are programs written on the SDK alone; see the [examples](../examples/index.md).
