<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Examples

Runnable material that stays next to its code. Start with the first-run example, which needs one page of text and a small
model. The samples are controlled measurements: each states one question, how it was measured and what it found, and
none of them is a claim that the loop makes a model generally better (see [limitations](../concepts/limitations.md)).

| Example | The question it asks |
|---|---|
| [First run](../../examples/first-run/README.md) | Can a 0.6B model be taught a one-page manual for a device that does not exist? The walk-through is [your first learning run](../getting-started/first-learning-run.md). |
| [Repository task families](../../tasks/README.md) | The frozen catalog of eight projects `splinter practice` learns from their own tests, and the hidden-world tasks. |
| [absorb](../../samples/absorb/README.md) | Can a model be corrected by its user in conversation? |
| [absorb-story](../../samples/absorb-story/README.md) | What does the nightly loop keep, replace and drop in the situations it will meet? |
| [generations](../../samples/generations/README.md) | Does a policy go on learning when each generation is taken into the weights? |
| [jefferson](../../samples/jefferson/README.md) | Can Splinter, given one sentence and a directory, learn to think like Thomas Jefferson? |
| [adams](../../samples/adams/README.md) | Can a small model learn how Samuel Adams worked from his own papers? |
| [synthetic-author](../../samples/synthetic-author/README.md) | Can Splinter learn beliefs it was never told, against ground truth that cannot have been seen in pretraining? |
| [tool-syntax](../../samples/tool-syntax/README.md) | Does verified experience teach a small model to look before it acts? |
| [repo-replay](../../samples/repo-replay/README.md) | Can a git history be replayed by an agent, commit by commit, into verified supervised data? |
| [agent-loop](../../samples/agent/loop/README.md) | Can a local model complete a repository task unaided, and can a supervisor audit it? |
| [health](../../samples/health/README.md) | Does a candidate risk model replace the one in place only when held-out evidence says it should? |
| [lifecourse](../../samples/lifecourse/README.md) | Can a model read one health examination and say what is likely in fifteen years? |
| [persona-chat](../../samples/persona-chat/README.md) | Talk to a model, a trained persona or not, through tools you define. |
| [voice-loop](../../samples/voice-loop/README.md) | Can a persona be talked to and talk back? |

Each sample is a standalone program on the [SDK](../reference/sdk.md) alone; `make samples` lists the build and run targets.
