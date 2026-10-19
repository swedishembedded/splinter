<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Splinter

Splinter turns agent experience into better local models. You tell it what to learn in plain language, a
document, a repository, a command-line tool or your own sessions with an agent, and it writes tasks from that
material, has the model attempt them, verifies the attempts with checks the model cannot influence, trains a LoRA
adapter on what was verified, and releases the result only when held-out evidence shows it improved and nothing
else got worse. A release is an ordinary [brain](https://swedishembedded.com/docs/brain) adapter with a manifest:
it serves on plain `brain serve`, and [sven](https://swedishembedded.com/docs/sven) can use it through its brain
provider. Neither needs Splinter installed.

```text
   task --> experience --> evidence --> dataset --> training --> evaluation --> release
```

## Where to start

- New here: [Installation](getting-started/installation.md), [Requirements](getting-started/requirements.md), then
  [Your first learning run](getting-started/first-learning-run.md).
- Teach it something: [documents and commands](guides/documents.md), [a repository](guides/repositories.md),
  [your own agent sessions](guides/agent-sessions.md), [a person's way of thinking](guides/personas.md).
- Check what it learned: [Measure what a candidate learned](guides/evaluation.md) and
  [Release, serve and roll back](guides/releases.md).
- Understand it: [Architecture](concepts/architecture.md), [the learning loop](concepts/learning-loop.md),
  [Verification](concepts/verification.md), [Release policy](concepts/release-policy.md) and, before you trust a
  result, [Limitations](concepts/limitations.md).
- Look something up: the [command reference](reference/cli.md), [configuration](getting-started/configuration.md),
  [output and exit status](reference/output.md), [storage](reference/storage.md) and the [Rust SDK](reference/sdk.md).

## How the manual is organised

| Section | Holds |
|---|---|
| Getting started | Installing, requirements, a first run you can follow, configuration |
| Guides | One page per workflow: what to run, in what order, and what you get |
| Concepts | Why it works the way it does: architecture, the loop, experience, verification, provenance, release policy, limitations |
| Reference | Precise interfaces: the generated command reference, tasks, datasets, evaluation, storage, output, SDK |
| Examples | The runnable samples, which stay next to their code |
| Contributing | Building, testing, the gates, and how this documentation is produced |

## Three projects, one stack

| Project | Its role |
|---|---|
| Splinter | Decides what is learned: which tasks to write, which experience qualifies as training data, whether a result is released |
| [sven](https://swedishembedded.com/docs/sven) | Runs the agent: tools, sessions, delegation and a trustworthy record of what the agent did |
| [brain](https://swedishembedded.com/docs/brain) | Runs the model: inference, training, evaluation and serving on the hardware you have |

Splinter is built on the other two and replaces neither; see [Architecture](concepts/architecture.md). It is released
under the Apache-2.0 license.
