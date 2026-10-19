<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# First run: teach a model a document it has never seen

`manual.md` describes a device that does not exist (the Quillon QX-7), so no model can know its console
speed, boot jumper, power limits or watchdog from training. `anchor.jsonl` is a twenty-question suite of
general knowledge that the release gate asks before and after training, to refuse a candidate that got worse at it.

The walk-through, with the commands, the output to expect from each step and the ways each can fail, is
[Your first learning run](../../docs/getting-started/first-learning-run.md). Run its commands from the
repository root after `make build`.
