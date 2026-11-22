<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 213. A dataset built for a base is read by that base's trainer at build time

**Lesson:** a record the trainer refuses must be found when the dataset is built, by the trainer's own check, never
when a run starts. `dataset build --from BASE` now has every record read by BASE's trainer (brain's `check_dataset`
through the service, the one check there is; splinter renders no template). A refused record is left out as
`trainer_refused`, with the first reason in the build report, and is never counted as `over_block`. A build that names
its base and has no brain service to ask is refused; only a build without `--from` may still fall back to the
character floor, and it says so.

## Evidence (2026-11-22, lane PLAY, SD1)

SD1's fold-a dataset was built with `--from local:Qwen/Qwen3-4B --max-block 16384` and no brain endpoint, so the
build fell back to the floor (a warning in the text report, nothing in `--json`). Both trainings then failed at start:
`line 127: the final message does not follow the prompt the model is asked from: the template renders the
conversation so far differently with and without it`. Before the fix, a build that could ask the service also left
such a record out, but counted it as `over_block`. Spec: `crates/pipelines/tests/it/dataset_split.rs`, a record the
scripted trainer refuses and a build that names its base without a service.
