<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Splinter

A learning agent with its own model. Tell it what to learn, in plain
language; it learns it, measures that it did, and releases a better version
of itself.

```
splinter "Learn all the facts about the STM32 reference manual from <path-to-manual>"
splinter "Learn what we can do with brain command line options"
splinter "Explain brain command line flags without interacting with brain"
```

Splinter is built on two standalone projects and replaces neither:

- **sven** runs the agent: tools, sessions, delegation, and a trustworthy
  record of what the agent did.
- **brain** runs the model: inference, training, evaluation and serving.

What Splinter releases is an ordinary brain adapter with a manifest. It runs
on plain `brain serve`, and plain sven can use it through its brain
provider; neither needs Splinter installed.

## Status

Early. See `AGENTS.md` for how the repository is organised.

## License

Apache-2.0. See `LICENSE`.
