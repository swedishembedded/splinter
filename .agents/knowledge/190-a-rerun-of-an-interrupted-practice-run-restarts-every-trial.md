<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 190. A rerun of an interrupted practice run restarts every trial

**Lesson:** `splinter tasks harbor practice` finds its run again by seed, but a run that was killed before it finished is
staged again and every trial runs anew; the finished trials of the killed attempt are not kept. Run long practice and
training commands fully detached (their own session, not a child of a wrapper that a time limit can kill), and size a
batch so that losing it is cheap.

Seen 2026-11-20 (item 19): a tool's two-hour background limit killed the round-1 practice driver after about 470 of 640
trials. The rerun with the same seeds restaged all five draws and ran 640 trials again (about 1 GPU-h lost). The same limit
killed a `splinter train` client at step 38 of 72. Splinter then cancelled brain's job (`DELETE /v1/jobs/{id}`), and the
rerun started at step 1. `--detach-on-interrupt` keeps the brain job running in that case.
