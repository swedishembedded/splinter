<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Test

```bash
make test                          # every test; no model, GPU or network needed
make test/crate/splinter-pipelines # one package; `make test/crate` lists them
make test/families                 # the repository task families: baselines, conventions, fault detection
make test/e2e                      # lists the end-to-end targets
```

`make test` loads no model. The end-to-end targets need built brain and sven binaries and a real model, and say so:

- `make test/e2e/brain-sven` has brain serve Qwen3-0.6B and sven complete a tool-using task with it, judged on a
  number only a file holds. It runs on the CPU backend and exits 3, never 0, when a binary or the model is missing, so a
  skipped run is never read as a pass. Name the binaries with `BRAIN_BIN`, `SVEN_BIN` and the store with `MODELS_DIR`.
- `make test/e2e/turn` runs one full turn on a small model: practice on a tiny fixture, experiences with verdicts, a
  trained candidate, the release gate, and the candidate served and driven by sven. Its spec, run against fakes, is
  `make test/e2e/turn/spec`.

Write the test from the specification first and see it fail, then implement. Tests assert observable behavior at a real
boundary: a test that returns early reports a pass, and a gate rejects a test that skips itself silently. The tests that
document a user-facing walk-through run the real binary against an empty home, so the documentation cannot drift from the
program; see [documentation](documentation.md).
